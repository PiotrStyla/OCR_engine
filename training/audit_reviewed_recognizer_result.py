"""CPU-only integrity and paired-metric audit of reviewed Colab training results."""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile

from training.full_page_pilot import digest, write_json, write_rows
from training.run_reviewed_recognizer_colab import normalize

MODEL = 'recognizer-reviewed-colab-v1-model.zip'
EVIDENCE = 'recognizer-reviewed-colab-v1-evidence.zip'


def stream_digest(stream):
    sha = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
        sha.update(chunk)
    return sha.hexdigest()


def weights_digest(stream, size):
    import numpy as np

    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError('Missing safetensors header')
    length = struct.unpack('<Q', prefix)[0]
    if not 0 < length <= 16 * 1024**2:
        raise ValueError('Invalid safetensors header length')
    raw = stream.read(length)
    if len(raw) != length:
        raise ValueError('Truncated safetensors header')
    header = json.loads(raw)
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    if not tensors or any('lora_' in name for name in tensors):
        raise ValueError('Expected full merged weights, not an adapter')
    cursor, parameters = 0, 0
    for tensor in sorted(tensors.values(), key=lambda v: v['data_offsets'][0]):
        start, end = tensor['data_offsets']
        count = math.prod(tensor['shape'])
        if tensor['dtype'] != 'F32' or start != cursor or end - start != count * 4:
            raise ValueError('Unexpected dtype, shape or non-contiguous tensor offsets')
        cursor = end
        parameters += count
    if size != 8 + length + cursor:
        raise ValueError('Safetensors payload length mismatch')
    sha = hashlib.sha256(prefix + raw)
    read = 0
    for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
        sha.update(chunk)
        read += len(chunk)
        if len(chunk) % 4 or not np.isfinite(np.frombuffer(chunk, dtype='<f4')).all():
            raise ValueError('Non-finite or truncated float32 weights')
    if read != cursor:
        raise ValueError('Truncated weight payload')
    return sha.hexdigest(), {'tensors': len(tensors), 'parameters': parameters,
        'dtype': 'F32', 'all_weights_finite': True, 'merged_weights': True}


def verify_zip(archive, checksum_name='checksums.json', check_weights=False):
    names = archive.namelist()
    if len(names) != len(set(names)) or any(Path(n).name != n or '\\' in n for n in names):
        raise ValueError('Duplicate or non-root archive members')
    expected = json.loads(archive.read(checksum_name))
    if set(names) != set(expected) | {checksum_name}:
        raise ValueError('Checksum coverage mismatch')
    weights = None
    for name, sha in expected.items():
        with archive.open(name) as stream:
            if check_weights and name == 'model.safetensors':
                actual, weights = weights_digest(stream, archive.getinfo(name).file_size)
            else:
                actual = stream_digest(stream)
        if actual != sha:
            raise ValueError('Checksum mismatch: ' + name)
    if check_weights and weights is None:
        raise ValueError('Missing full weights')
    return expected, weights


def paired_metrics(baseline, candidate):
    from jiwer import process_characters, wer

    def index(rows):
        result = {(r['domain'], r['id']): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate prediction IDs')
        return result

    left, right = index(baseline), index(candidate)
    if left.keys() != right.keys():
        raise ValueError('Prediction ID mismatch')
    lines, metrics = [], {}
    for key in sorted(left):
        if left[key]['reference'] != right[key]['reference']:
            raise ValueError('Reference changed between baseline and candidate')
        ref = normalize(left[key]['reference'])
        b = process_characters(ref, normalize(left[key]['prediction']))
        c = process_characters(ref, normalize(right[key]['prediction']))
        lines.append({'domain': key[0], 'id': key[1], 'reference': left[key]['reference'],
            'baseline': left[key]['prediction'], 'candidate': right[key]['prediction'],
            'baseline_cer': b.cer, 'candidate_cer': c.cer,
            'baseline_errors': b.substitutions + b.deletions + b.insertions,
            'candidate_errors': c.substitutions + c.deletions + c.insertions,
            'cer_delta': c.cer - b.cer})
    for domain in sorted({r['domain'] for r in lines}):
        subset = [r for r in lines if r['domain'] == domain]
        refs = [normalize(r['reference']) for r in subset]
        result = {'lines': len(subset), 'improved_lines': sum(r['cer_delta'] < 0 for r in subset),
            'regressed_lines': sum(r['cer_delta'] > 0 for r in subset),
            'equal_cer_lines': sum(r['cer_delta'] == 0 for r in subset)}
        for variant in ('baseline', 'candidate'):
            hyps = [normalize(r[variant]) for r in subset]
            chars = process_characters(refs, hyps)
            result[variant] = {'cer': chars.cer, 'wer': wer(refs, hyps),
                'lines': len(subset), 'character_errors': chars.substitutions + chars.deletions + chars.insertions,
                'replacement_characters': sum(r[variant].count('\ufffd') for r in subset)}
        result['cer_delta_percentage_points'] = 100 * (result['candidate']['cer'] - result['baseline']['cer'])
        metrics[domain] = result
    refs = [normalize(r['reference']) for r in lines]
    combined = {v: process_characters(refs, [normalize(r[v]) for r in lines]).cer
                for v in ('baseline', 'candidate')}
    return metrics, combined, lines


def audit(source, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve existing audit; select a new output directory')
    with zipfile.ZipFile(source) as outer:
        outer_sums, _ = verify_zip(outer, 'result-checksums.json')
        if set(outer_sums) != {MODEL, EVIDENCE}:
            raise ValueError('Unexpected result package')
        with zipfile.ZipFile(io.BytesIO(outer.read(EVIDENCE))) as evidence:
            evidence_sums, _ = verify_zip(evidence)
            def obj(name):
                return json.loads(evidence.read(name))
            def rows(name):
                return [json.loads(line) for line in evidence.read(name).decode('utf-8').splitlines()]
            environment, run = obj('environment.json'), obj('training-run.json')
            manifests, config = obj('input-manifests.json'), obj('corpus-config.json')
            if run['source_commit'] != environment['code_revision']:
                raise ValueError('Code revision mismatch')
            if config['base_revision'] != environment['base_revision']:
                raise ValueError('Base model revision mismatch')
            frozen_config = Path(__file__).resolve().parents[1]/'experiments/2026-10-06/recognizer-reviewed-colab-v1/config.json'
            if hashlib.sha256(evidence.read('corpus-config.json')).hexdigest() != digest(frozen_config):
                raise ValueError('Frozen recipe mismatch')
            def record_key(r):
                return r['id'], r['image_sha256'], r['text_sha256']
            expected_train = manifests['reviewed'] * config['training']['historical_repeats'] + manifests['synthetic']
            recipe = config['training']
            expected_arguments = {'num_train_epochs': recipe['epochs'],
                'per_device_train_batch_size': recipe['batch_size'],
                'gradient_accumulation_steps': recipe['gradient_accumulation_steps'],
                'learning_rate': recipe['learning_rate'], 'seed': recipe['seed'],
                'load_best_model_at_end': True, 'metric_for_best_model': 'cer', 'fp16': True}
            if any(run['training_arguments'].get(k) != v for k, v in expected_arguments.items()):
                raise ValueError('Training arguments differ from frozen recipe')
            expected_val = [{**r, 'id': domain+'__'+r['id']} for domain in
                ('historical-development', 'ordinary-development') for r in manifests[domain]]
            if Counter(map(record_key, expected_train)) != Counter(map(record_key, run['train'])):
                raise ValueError('Training input lineage mismatch')
            if Counter(map(record_key, expected_val)) != Counter(map(record_key, run['validation'])):
                raise ValueError('Validation input lineage mismatch')
            if {r['image_sha256'] for r in run['train']} & {r['image_sha256'] for r in run['validation']}:
                raise ValueError('Exact train/development image overlap')
            baseline, candidate = rows('baseline-predictions.jsonl'), rows('candidate-predictions.jsonl')
            metrics, combined, lines = paired_metrics(baseline, candidate)
            for variant in ('baseline', 'candidate'):
                supplied = obj(variant+'-metrics.json')
                for domain, values in metrics.items():
                    if {k: values[variant][k] for k in ('cer', 'wer', 'lines')} != supplied[domain]:
                        raise ValueError('Recomputed metrics differ from report')
            lookup = {(domain, r['id']): r for domain in
                ('historical-development', 'ordinary-development') for r in manifests[domain]}
            if set(lookup) != {(r['domain'], r['id']) for r in baseline}:
                raise ValueError('Predictions do not cover development inputs exactly')
            for row in baseline:
                if hashlib.sha256(row['reference'].encode('utf-8')).hexdigest() != lookup[row['domain'], row['id']]['text_sha256']:
                    raise ValueError('Prediction reference hash differs from input label')
            selection = obj('training-selection.json')
            result = obj('result.json')
            if result['sota_claim'] or result['production_promoted'] or result['independent_benchmark']:
                raise ValueError('Unexpected promotion claim')
            for domain, key in [('historical-development', 'historical_cer_delta'),
                                ('ordinary-development', 'ordinary_cer_delta')]:
                delta = metrics[domain]['candidate']['cer'] - metrics[domain]['baseline']['cer']
                if not math.isclose(delta, result[key], abs_tol=1e-12):
                    raise ValueError('Reported CER delta mismatch')
            state_bytes = None
            with outer.open(MODEL) as model_stream, zipfile.ZipFile(model_stream) as model:
                model_sums, weight_stats = verify_zip(model, check_weights=True)
                for model_name, evidence_name in [('run.json', 'training-run.json'),
                    ('selection.json', 'training-selection.json'), ('best_metrics.json', 'training-best_metrics.json')]:
                    if model.read(model_name) != evidence.read(evidence_name):
                        raise ValueError('Model/evidence metadata mismatch')
                state_bytes = model.read('trainer_state.json')
                state = json.loads(state_bytes)
                if state['global_step'] != 294 or state['epoch'] != 3.0:
                    raise ValueError('Training did not complete the frozen three-epoch recipe')
                if state['best_model_checkpoint'] != selection['best_checkpoint'] or state['best_metric'] != selection['best_cer']:
                    raise ValueError('Checkpoint selection mismatch')
            report = {'schema': 'slayer-reviewed-recognizer-result-audit-v1',
                'archive_sha256': digest(source), 'inner_archive_sha256': outer_sums,
                'verified_members': {'outer': 3, 'evidence': len(evidence_sums)+1, 'model': len(model_sums)+1},
                'weights': weight_stats, 'environment': environment,
                'metrics': metrics, 'combined_strict_cer': combined,
                'selection': selection, 'training_steps': state['global_step'],
                'trainer_selection_cer_reproduced': math.isclose(selection['best_cer'], combined['candidate'], abs_tol=1e-12),
                'training_epochs': state['epoch'], 'training_examples_per_epoch': len(run['train']),
                'development_examples': len(run['validation']),
                'decision': 'retain-experimental-candidate-do-not-replace-baseline',
                'reason': 'Small historical development gain accompanies ordinary-print regression and a new replacement character.',
                'independent_benchmark': False, 'sota_claim': False, 'production_promoted': False,
                'inference_repeated_locally': False,
                'limitations': ['Only nine historical development lines; checkpoint selection uses development data.',
                    'Line crops do not evaluate page layout, segmentation or reading order.',
                    'Trainer CER uses tokenizer-decoded labels/cleanup, unlike the raw-reference strict audit.',
                    'Trainer eval_loss is not used for selection; aligned evaluation loss requires separate verification.']}
            output.mkdir(parents=True)
            (output/EVIDENCE).write_bytes(outer.read(EVIDENCE))
            (output/'trainer-state.json').write_bytes(state_bytes)
            write_json(output/'audit.json', report)
            write_rows(output/'line-comparison.jsonl', lines)
            write_json(output/'checksums.json', {p.name: digest(p) for p in output.iterdir()})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.output), ensure_ascii=False, indent=2))
