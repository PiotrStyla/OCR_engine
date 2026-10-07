"""CPU-only audit of all V2 checkpoints, merged predictions and baseline selection."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import zipfile

from training.audit_reviewed_recognizer_result import paired_metrics, verify_zip
from training.full_page_pilot import digest, write_json, write_rows
from training.merge_recognizer_reviewed_pool import verified_package
from training.reviewed_recognizer_selection import DOMAINS, eligibility, select, summarize

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'experiments/2026-10-07/recognizer-reviewed-colab-v2/config.json'
CODE_REVISION = 'ee90eb4426232f0a03937787c98291b59ce8398a'


def assert_close(actual, expected, label, tolerance=1e-12):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError('Metric mismatch: ' + label)


def record_key(row):
    return row['id'], row['image_sha256'], row['text_sha256']


def verify_pair_files(directory, records):
    directory = Path(directory)
    if {path.stem for path in directory.glob('*.png')} != {row['id'] for row in records}:
        raise ValueError('Local pair coverage mismatch')
    for row in records:
        if (digest(directory / (row['id'] + '.png')) != row['image_sha256']
                or digest(directory / (row['id'] + '.txt')) != row['text_sha256']):
            raise ValueError('Local pair file checksum mismatch')


def verify_references(rows, baseline, manifests):
    paired_metrics(baseline, rows)
    expected = {(domain, row['id']): row for domain in DOMAINS for row in manifests[domain]}
    if len(expected) != sum(len(manifests[domain]) for domain in DOMAINS):
        raise ValueError('Duplicate development input IDs')
    if {(row['domain'], row['id']) for row in rows} != set(expected):
        raise ValueError('Development prediction coverage mismatch')
    for row in rows:
        text_hash = hashlib.sha256(row['reference'].encode('utf-8')).hexdigest()
        if text_hash != expected[row['domain'], row['id']]['text_sha256']:
            raise ValueError('Reference hash differs from frozen input')


def verify_training_lineage(run, variant, recipe, manifests, replay, code_revision):
    expected_train = manifests['reviewed'] * recipe['historical_repeats'] + replay
    expected_val = [{**row, 'id': domain + '__' + row['id']}
                    for domain in DOMAINS for row in manifests[domain]]
    if Counter(map(record_key, run['train'])) != Counter(map(record_key, expected_train)):
        raise ValueError('Training lineage mismatch')
    if Counter(map(record_key, run['validation'])) != Counter(map(record_key, expected_val)):
        raise ValueError('Validation lineage mismatch')
    if run['source_commit'] != code_revision:
        raise ValueError('Training code revision mismatch')
    expected_args = dict(num_train_epochs=recipe['epochs'],
        per_device_train_batch_size=recipe['batch_size'],
        gradient_accumulation_steps=recipe['gradient_accumulation_steps'],
        learning_rate=variant['learning_rate'], seed=recipe['seed'], data_seed=recipe['seed'],
        metric_for_best_model='selection_score', load_best_model_at_end=True, fp16=True)
    if any(run['training_arguments'].get(key) != value for key, value in expected_args.items()):
        raise ValueError('Training recipe mismatch')
    if (run['use_4bit'] or run['include_mlp'] or run['max_target_length'] != recipe['max_target_length']
            or run['gradient_accumulation_steps'] != recipe['gradient_accumulation_steps']):
        raise ValueError('Training protocol mismatch')


def verify_checkpoint_metrics(scores, gate, logged):
    expected = dict(eval_cer=scores['combined']['cer'], eval_wer=scores['combined']['wer'],
                    eval_eligible=int(gate['eligible']))
    for domain in DOMAINS:
        for metric in ('cer', 'wer', 'replacement_characters'):
            expected['eval_' + domain + '_' + metric] = scores[domain][metric]
    for key, value in expected.items():
        assert_close(logged[key], value, key)
    score = scores['combined']['cer'] + (0 if gate['eligible'] else 1e6)
    assert_close(logged['eval_selection_score'], score, 'selection_score', 1e-9)
    if not math.isfinite(logged['eval_loss']) or logged['eval_loss'] < 0:
        raise ValueError('Invalid reported evaluation loss')
    return score


def audit(source, output, input_root=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('Preserve the previous audit; choose a new output directory')
    comparisons, variants = [], []
    with zipfile.ZipFile(source) as archive:
        sums, _ = verify_zip(archive)
        def obj(name):
            return json.loads(archive.read(name))
        def rows(name):
            return [json.loads(line) for line in archive.read(name).decode('utf-8').splitlines() if line.strip()]
        if 'failure.json' in archive.namelist():
            raise ValueError('This is a failed run, not completed V2 evidence: ' + str(obj('failure.json')))
        cfg, environment = obj('experiment-config.json'), obj('environment.json')
        if cfg != json.loads(CONFIG.read_text(encoding='utf-8')):
            raise ValueError('Frozen V2 configuration mismatch')
        if hashlib.sha256(archive.read('experiment-config.json')).hexdigest() != environment['config_sha256']:
            raise ValueError('Environment/config digest mismatch')
        if environment['code_revision'] != CODE_REVISION:
            raise ValueError('Unexpected V2 runtime revision')
        corpus_sums, corpus_cfg = obj('corpus-checksums.json'), obj('corpus-config.json')
        if hashlib.sha256(archive.read('corpus-config.json')).hexdigest() != cfg['input_config_sha256']:
            raise ValueError('Frozen input policy mismatch')
        for name in ('config.json', 'report.json', 'manifest.jsonl', 'source-ledger.jsonl'):
            if hashlib.sha256(archive.read('corpus-' + name)).hexdigest() != corpus_sums[name]:
                raise ValueError('Preserved corpus metadata checksum mismatch')
        if input_root is not None:
            verified_package(input_root)
            for name in ('config.json', 'report.json', 'manifest.jsonl', 'source-ledger.jsonl', 'checksums.json'):
                if archive.read('corpus-' + name) != (Path(input_root) / name).read_bytes():
                    raise ValueError('Original reviewed package metadata differs from evidence')
        manifests, replays = obj('input-manifests.json'), obj('replay-manifests.json')
        if {name: len(value) for name, value in manifests.items()} != {
                'reviewed': 70, 'synthetic': 2000, DOMAINS[0]: 9, DOMAINS[1]: 75}:
            raise ValueError('Input count drift')
        verify_pair_files(REPO / 'benchmarks/real-lines-v1/pairs', manifests[DOMAINS[1]])
        hashes = [row['image_sha256'] for group in manifests.values() for row in group]
        if len(hashes) != len(set(hashes)):
            raise ValueError('Exact input duplicates or train/development overlap')
        original = rows('corpus-manifest.jsonl')
        for split, domain in [('train', 'reviewed'), ('development', DOMAINS[0])]:
            pool = [row for row in original if row['split'] == split]
            if Counter(map(record_key, pool)) != Counter(map(record_key, manifests[domain])):
                raise ValueError('Reviewed input manifest drift')
            if any(row['collection'] in corpus_cfg['excluded_collections'] for row in pool):
                raise ValueError('Excluded work family used')
        pages = {row['page_id'] for row in original if row['split'] == 'train'}
        if pages & {row['page_id'] for row in original if row['split'] == 'development'}:
            raise ValueError('Reviewed page overlap')
        ordered = sorted(manifests['synthetic'], key=lambda row: hashlib.sha256(
            (cfg['synthetic_order_prefix'] + row['id']).encode()).hexdigest())
        for count in (500, 2000):
            if Counter(map(record_key, replays[str(count)])) != Counter(map(record_key, ordered[:count])):
                raise ValueError('Deterministic replay subset drift')
        tokens = rows('tokenizer-audit.jsonl')
        if Counter((row['domain'], row['id']) for row in tokens) != Counter(
                (domain, row['id']) for domain, group in manifests.items() for row in group):
            raise ValueError('Tokenizer audit coverage mismatch')
        if any(not 0 < row['tokens'] <= cfg['training']['max_target_length'] for row in tokens):
            raise ValueError('Invalid audited target length')
        preflight = obj('adapter-preflight.json')
        if (preflight['status'] != 'ok' or not preflight['finite_gradients']
                or not preflight['merge_and_generation'] or not preflight['generation_evaluation_aligned_loss']
                or preflight['updates_used_in_training'] or preflight['updated_adapter_tensors'] <= 0):
            raise ValueError('Incomplete adapter/evaluation preflight')
        sample = next((row for row in manifests['reviewed'] if row['id'] == preflight['training_sample_id']), None)
        if not sample or any(preflight[key] != sample[key] for key in ('image_sha256', 'text_sha256')):
            raise ValueError('Preflight did not use a frozen training sample')
        baseline_rows = rows('baseline-predictions.jsonl')
        verify_references(baseline_rows, baseline_rows, manifests)
        baseline = summarize(baseline_rows)
        if baseline != obj('baseline-metrics.json') or baseline != obj('selection-policy.json')['baseline']:
            raise ValueError('Baseline metric mismatch')
        for variant in cfg['variants']:
            identifier = variant['id']
            run, state = obj(identifier + '-run.json'), obj(identifier + '-trainer_state.json')
            verify_training_lineage(run, variant, cfg['training'], manifests,
                                    replays[str(variant['replay_count'])], CODE_REVISION)
            if run['selection_policy'] != obj('selection-policy.json'):
                raise ValueError('Checkpoint policy changed during training')
            count = len(run['train'])
            steps_per_epoch = math.ceil(math.ceil(count / cfg['training']['batch_size']) /
                                        cfg['training']['gradient_accumulation_steps'])
            if state['epoch'] != cfg['training']['epochs'] or state['global_step'] != steps_per_epoch * cfg['training']['epochs']:
                raise ValueError('Training did not complete all configured epochs')
            epochs = [item for item in state['log_history'] if 'eval_cer' in item]
            if len(epochs) != cfg['training']['epochs']:
                raise ValueError('Missing epoch evaluations')
            checkpoints = []
            for index, logged in enumerate(epochs, 1):
                checkpoint_rows = rows(f'{identifier}-checkpoint-evaluation-{index:03d}.jsonl')
                verify_references(checkpoint_rows, baseline_rows, manifests)
                scores = summarize(checkpoint_rows)
                gate = eligibility(scores, baseline)
                score = verify_checkpoint_metrics(scores, gate, logged)
                checkpoints.append(dict(epoch=logged['epoch'], step=logged['step'], metrics=scores,
                    eligible=gate['eligible'], reasons=gate['reasons'], selection_score=score,
                    reported_aligned_eval_loss=logged['eval_loss']))
            winner = min(checkpoints, key=lambda row: row['selection_score'])
            selected = obj(identifier + '-selection.json')
            if (selected['best_checkpoint'] != state['best_model_checkpoint']
                    or selected['metric_name'] != 'selection_score'
                    or Path(state['best_model_checkpoint']).name != 'checkpoint-' + str(winner['step'])):
                raise ValueError('Best checkpoint identity mismatch')
            assert_close(state['best_metric'], winner['selection_score'], 'best state score', 1e-9)
            assert_close(selected['best_metric'], winner['selection_score'], 'selected score', 1e-9)
            reevaluated = rows(f'{identifier}-checkpoint-evaluation-004.jsonl')
            verify_references(reevaluated, baseline_rows, manifests)
            reevaluated_metrics = summarize(reevaluated)
            verify_checkpoint_metrics(reevaluated_metrics, eligibility(reevaluated_metrics, baseline),
                                      obj(identifier + '-best_metrics.json'))
            candidate_rows = rows(identifier + '-predictions.jsonl')
            verify_references(candidate_rows, baseline_rows, manifests)
            metrics = summarize(candidate_rows)
            if metrics != obj(identifier + '-metrics.json'):
                raise ValueError('Merged candidate metrics mismatch')
            domain_metrics, combined, paired = paired_metrics(baseline_rows, candidate_rows)
            comparisons += [dict(row, variant=identifier) for row in paired]
            drift, _, merge_rows = paired_metrics(reevaluated, candidate_rows)
            changed = [row for row in merge_rows if row['baseline'] != row['candidate']]
            training_summary = next(item for item in state['log_history'] if 'train_loss' in item)
            variants.append(dict(id=identifier, metrics=metrics, domain_comparison=domain_metrics,
                combined_strict_cer=combined, **eligibility(metrics, baseline),
                epochs=state['epoch'], optimizer_steps=state['global_step'], examples_per_epoch=count,
                best_checkpoint=selected['best_checkpoint'], checkpoints=checkpoints,
                reported_train_loss=training_summary['train_loss'],
                merge_stage_prediction_changes=dict(changed_lines=len(changed),
                    changed_predictions=changed, domain_comparison=drift)))
        selection = select(baseline, [dict(id=row['id'], metrics=row['metrics']) for row in variants])
        reported = obj('selection.json')
        if any(reported[key] != value for key, value in selection.items()):
            raise ValueError('Final baseline/candidate selection does not reproduce')
        if (reported['base_revision'] != corpus_cfg['base_revision']
                or reported['input_dataset_revision'] != cfg['dataset_revision']
                or reported['input_dataset_sha256'] != cfg['dataset_sha256']):
            raise ValueError('Selected base/input provenance mismatch')
    report = dict(schema='slayer-reviewed-recognizer-v2-evidence-audit-v1',
        archive_sha256=digest(source), archive_bytes=source.stat().st_size, verified_members=len(sums) + 1,
        audit_implementation_sha256=digest(Path(__file__)),
        metric_implementation_sha256=digest(REPO / 'training/reviewed_recognizer_selection.py'),
        audit_jiwer_version=importlib.metadata.version('jiwer'),
        environment=environment, baseline=baseline, variants=variants, selection=selection,
        epoch_checkpoints_audited=sum(len(row['checkpoints']) for row in variants),
        eligible_epoch_checkpoints=sum(checkpoint['eligible'] for row in variants for checkpoint in row['checkpoints']),
        preflight_reported_aligned_evaluation_passed=True, model_weights_in_this_archive=False,
        original_reviewed_input_package_verified=input_root is not None,
        ordinary_input_files_verified=True,
        model_weights_verified=False, inference_repeated_locally=False,
        limitations=['Evidence ZIP contains no weights or image payloads; local input files were compared where available, not GPU tensors.',
                     'Synthetic image payloads and trained model weights were not independently verified in this evidence-only audit.',
                     'Only nine historical development lines; all checkpoint selection reuses development data.',
                     'Line crops do not evaluate full-page segmentation, reading order, or SOTA.',
                     'Evaluation losses are reported values, not independently recomputed without model weights.'])
    output.mkdir(parents=True)
    shutil.copyfile(source, output / source.name)
    write_json(output / 'audit.json', report)
    write_rows(output / 'line-comparison.jsonl', comparisons)
    write_json(output / 'checksums.json', {path.name: digest(path) for path in output.iterdir() if path.is_file()})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, help='Optional original reviewed package for image/metadata checksum verification')
    args = parser.parse_args()
    result = audit(args.archive, args.output, args.input_root)
    print(json.dumps({key: result[key] for key in ('verified_members', 'epoch_checkpoints_audited',
        'eligible_epoch_checkpoints', 'selection')}, ensure_ascii=False, indent=2))
