"""Audit returned full v7 validation and independently reproduce its scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import zipfile

from training.audit_full_page_comparison import audit as audit_v5
from training.benchmark_pages import evaluate
from training.full_page_pilot import digest, read_rows, safe_relative, write_json, write_rows
from training import full_page_validation_4mp as validation


ROOT = Path(__file__).resolve().parents[1]
ENGINE = 'qwen3-vl-4b'


def audit(archive_path, dataset, baseline_archive, config_path, output, code_revision):
    archive_path, dataset, baseline_archive, config_path, output = map(
        Path, (archive_path, dataset, baseline_archive, config_path, output))
    if output.exists():
        raise FileExistsError('Use a new audit directory')
    if len(code_revision) != 40 or any(c not in '0123456789abcdef' for c in code_revision):
        raise ValueError('Use the full published inference revision')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validation.validate_profile(config)
    code = {'code_revision': code_revision}
    for key, name in [('runner_sha256', 'full_page_pilot.py'),
                      ('comparison_sha256', 'full_page_comparison.py'),
                      ('validation_sha256', 'full_page_validation_4mp.py')]:
        code[key] = hashlib.sha256(subprocess.check_output(
            ['git', 'show', f'{code_revision}:training/{name}'], cwd=ROOT)).hexdigest()
    with zipfile.ZipFile(archive_path) as stream:
        members = stream.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or sum(item.file_size for item in members) > 100_000_000:
            raise ValueError('Duplicate or oversized evidence members')
        for item in members:
            name = safe_relative(item.filename)
            if (item.is_dir() or stat.S_ISLNK(item.external_attr >> 16)
                    or (Path(name).suffix not in ('.json', '.jsonl', '.csv', '.log')
                        and name != 'cer-comparison.png')):
                raise ValueError('Unexpected evidence member')
        payload = {name: stream.read(name) for name in names}
        checksums = json.loads(payload['checksums.json'])
        if set(checksums) != set(names)-{'checksums.json'} or any(
                hashlib.sha256(payload[name]).hexdigest() != sha for name, sha in checksums.items()):
            raise ValueError('Evidence checksum mismatch or incomplete coverage')
    decode = lambda name: json.loads(payload[name])
    rows = lambda name: [json.loads(line) for line in payload[name].splitlines() if line.strip()]
    if decode('config.json') != config:
        raise ValueError('Frozen configuration mismatch')
    for filename, keys in [('code-provenance.json', ('code_revision', 'runner_sha256', 'comparison_sha256')),
                           ('validation-code-provenance.json', ('code_revision', 'validation_sha256'))]:
        if any(decode(filename).get(key) != code[key] for key in keys):
            raise ValueError('Pinned code provenance mismatch')
    spec = config['models'][ENGINE]
    probes = [json.loads(line) for line in payload['preflight.log'].decode('utf-8').splitlines()
              if line.startswith('{')]
    if len(probes) != 1 or probes[0].get('packages') != dict(item.split('==') for item in spec['packages']):
        raise ValueError('Import preflight package mismatch')
    for name, expected in [('manifest.jsonl', config['dataset']['manifest_sha256']),
                           (config['baseline']['source'], config['dataset']['baseline_sha256'])]:
        if digest(dataset/name) != expected or hashlib.sha256(payload[f'dataset/{name}']).hexdigest() != expected:
            raise ValueError('Frozen source manifest/bundle baseline mismatch')
    references = read_rows(dataset/'manifest.jsonl')
    if (len(references) != config['dataset']['pages'] or len({row['id'] for row in references}) != len(references)
            or any(row['split'] != 'validation' or row.get('final_test') is not False
                   or row.get('eligible_for_training') is not False for row in references)):
        raise ValueError('Unexpected page scope')
    from PIL import Image
    for reference in references:
        image = dataset/safe_relative(reference['image'])
        if digest(image) != reference['sha256']:
            raise ValueError('Image checksum mismatch')
        with Image.open(image) as source:
            if source.size != (reference['width'], reference['height']):
                raise ValueError('Image dimension mismatch')
    expected_inputs = [{key: row[key] for key in ('id', 'image', 'sha256', 'width', 'height')}
                       | {'source_regions': []} for row in references]
    if rows('dataset/inference-inputs.jsonl') != expected_inputs:
        raise ValueError('Reference-free inference inputs mismatch')
    input_hash = hashlib.sha256(payload['dataset/inference-inputs.jsonl']).hexdigest()
    staging = decode('dataset/staging-provenance.json')
    if (staging['bundle_sha256'] != config['dataset']['archive_sha256'] or staging['inputs_sha256'] != input_hash
            or staging['reference_text_sent_to_model'] is not False
            or staging['source_geometry_sent_to_model'] is not False):
        raise ValueError('Staging provenance mismatch')
    retained = config['retained_baseline']
    if (digest(baseline_archive) != retained['archive_sha256'] or hashlib.sha256(
            payload['dataset/qwen-v5-1mp.jsonl']).hexdigest() != retained['raw_predictions_sha256']):
        raise ValueError('Retained baseline checksum mismatch')
    baseline_provenance = decode('baseline-provenance.json')
    for key, value in {'archive_sha256': retained['archive_sha256'],
                       'raw_predictions_sha256': retained['raw_predictions_sha256'],
                       'inference_code_revision': retained['inference_code_revision'],
                       'v5_metrics_recomputed': True, 'baseline_rerun': False,
                       'same_session_comparison': False, 'runtime_measurements_comparable': False}.items():
        if baseline_provenance.get(key) != value:
            raise ValueError('Baseline provenance mismatch')
    prefix = 'predictions/qwen3-vl-4b'
    if decode(f'{prefix}/identity.json') != {'engine': ENGINE, 'spec': spec, 'input_sha256': input_hash,
                                           'runner_sha256': code['runner_sha256']}:
        raise ValueError('Worker identity mismatch')
    environment = decode(f'{prefix}/environment.json')
    packages = {key.lower(): value for key, value in environment['packages'].items()}
    for package in spec['packages']:
        name, version = package.split('==')
        if name != 'sentencepiece' and packages.get(name.lower()) != version:
            raise ValueError('Worker package version mismatch')
    if environment['reference_text_sent_to_model'] is not False:
        raise ValueError('Worker reference leakage attestation mismatch')
    candidate = rows(f'{prefix}/{ENGINE}-full-page.jsonl')
    by_id = {row['id']: row for row in references}
    if len(candidate) != len(references) or {row['id'] for row in candidate} != set(by_id):
        raise ValueError('Incomplete or duplicate candidate coverage')
    for prediction in candidate:
        if prediction['status'] != 'ok':
            continue
        trace = prediction['generation_trace']
        tokens = trace['generated_token_ids']
        eos = bool(tokens and tokens[-1] in trace['eos_token_ids'])
        capped = len(tokens) >= spec['max_new_tokens'] and not eos
        if (prediction['generated_tokens'] != len(tokens) or prediction['token_limit_reached'] != capped
                or prediction['finish_reason'] != ('eos' if eos else ('length' if capped else 'unknown'))):
            raise ValueError('Token/termination mismatch')
        geometry = prediction['input_geometry']
        grid, patch, merge = geometry['image_grid_thw'], geometry['patch_size'], geometry['merge_size']
        reference = by_id[prediction['id']]
        expected = {'processed_width': grid[0][2]*patch, 'processed_height': grid[0][1]*patch,
            'processed_pixels': grid[0][1]*grid[0][2]*patch*patch,
            'visual_tokens': sum(t*h*w//(merge*merge) for t, h, w in grid),
            'original_width': reference['width'], 'original_height': reference['height'],
            'min_pixels': spec['min_pixels'], 'max_pixels': spec['max_pixels']}
        if (any(geometry.get(key) != value for key, value in expected.items())
                or not spec['min_pixels'] <= geometry['processed_pixels'] <= spec['max_pixels']):
            raise ValueError('Processed input geometry mismatch')
    output.mkdir(parents=True)
    baseline_check = audit_v5(baseline_archive, dataset, validation.V5_CONFIG,
                              output/'baseline-audit', retained['inference_code_revision'])
    if baseline_provenance['members_verified'] != baseline_check['members_verified']:
        raise ValueError('Baseline audit member count mismatch')
    audit_directory = safe_relative(baseline_provenance['audit_directory'])
    if not audit_directory.startswith('baseline/audit-'):
        raise ValueError('Unexpected nested baseline audit directory')
    if decode(f'{audit_directory}/audit.json') != baseline_check:
        raise ValueError('Nested baseline audit report mismatch')
    with zipfile.ZipFile(baseline_archive) as stream:
        for name, data in payload.items():
            if name.startswith(audit_directory+'/evidence/'):
                original_name = name[len(audit_directory+'/evidence/'):]
                if original_name not in stream.namelist() or stream.read(original_name) != data:
                    raise ValueError('Nested baseline evidence differs from original v5')
    extracted = output/'evidence'
    for name, data in payload.items():
        target = extracted/safe_relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # Recompute separately; never overwrite the returned metrics or predictions.
    local_dataset = output/'verified-dataset'
    local_dataset.mkdir()
    for name in ('manifest.jsonl', 'inference-inputs.jsonl', 'qwen-v5-1mp.jsonl'):
        (local_dataset/name).write_bytes(payload[f'dataset/{name}'])
    for reference in references:
        destination = local_dataset/safe_relative(reference['image'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset/reference['image'], destination)
    recomputed = validation.compare(config, local_dataset, extracted/f'{prefix}/{ENGINE}-full-page.jsonl',
                                    output/'recomputed')
    if recomputed != decode('scores/metrics.json'):
        raise ValueError('Recomputed metrics mismatch')
    for name in ('per-page.jsonl', 'input-and-glyph-diagnostics.json', 'scoring-config.json'):
        path = output/'recomputed'/name
        actual = read_rows(path) if name.endswith('.jsonl') else json.loads(path.read_text(encoding='utf-8'))
        expected = rows(f'scores/{name}') if name.endswith('.jsonl') else decode(f'scores/{name}')
        if actual != expected:
            raise ValueError('Recomputed diagnostics mismatch')
    if (output/'recomputed/qwen3-vl-4b-projected.jsonl').read_bytes() != payload['scores/qwen3-vl-4b-projected.jsonl']:
        raise ValueError('Scored projection mismatch')
    baseline = rows('dataset/qwen-v5-1mp.jsonl')
    labels = config['comparison_profile']['labels']
    results = {label: {row['id']: row for row in recomputed['reports'][label]['results']} for label in labels}
    directions = {kind: [] for kind in ('lower', 'higher', 'equal')}
    for reference in references:
        key = reference['id']
        difference = results[labels[1]][key]['cer']-results[labels[0]][key]['cer']
        directions['lower' if difference < 0 else ('higher' if difference > 0 else 'equal')].append(key)
    # Explanatory only: remove the old capped page symmetrically, never from primary scores.
    old_caps = {row['id'] for row in baseline if row.get('token_limit_reached')}
    uncapped_reference = [row for row in references if row['id'] not in old_caps]
    write_rows(local_dataset/'old-uncapped-diagnostic.jsonl', uncapped_reference)
    secondary = {}
    for label, predictions in zip(labels, (baseline, candidate)):
        path = output/'recomputed'/f'{label}-old-uncapped.jsonl'
        write_rows(path, [row for row in predictions if row['id'] not in old_caps])
        secondary[label] = evaluate(local_dataset/'old-uncapped-diagnostic.jsonl', path)
    report = {'schema': 'slayer-full-page-validation-4mp-v7-audit',
        'source_archive_sha256': digest(archive_path), 'members_verified': len(names),
        'metrics_recomputed': True, 'pages': len(references), 'code_provenance': code,
        'runtime_reported': decode('runtime.json'), 'worker_environment': environment,
        'import_preflight_recorded': probes[0], 'process_result': decode('process-result.json'),
        'retained_baseline_independently_audited': True, 'reports': recomputed['reports'],
        'per_page_CER_change': directions,
        'old_uncapped_secondary_diagnostic': {'primary_metric': False, 'outcome_selected': True,
            'excluded_from_secondary_only': sorted(old_caps), 'reports': secondary},
        'glyph_counts': {char: {'reference': sum(row['text'].count(char) for row in references),
                               'baseline': sum(row['text'].count(char) for row in baseline),
                               'candidate': sum(row['text'].count(char) for row in candidate)}
                        for char in ('\u017f', '\u00e1', '\u0247', 'Po\u017f\u0142a\u0142')},
        'glyph_counts_are_recall': False,
        'page_inference_seconds': sum(row['elapsed_seconds'] for row in candidate),
        'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in candidate),
        'baseline_rerun': False, 'same_session_comparison': False,
        'runtime_measurements_comparable': False, 'gold_pages': 0,
        'model_promotion': False, 'automatic_teacher_promotion': False, 'sota_claim': False,
        'claim_boundary': recomputed['claim_boundary']}
    shutil.copyfile(archive_path, output/'source-evidence.zip')
    write_json(output/'audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'dataset', 'baseline-archive', 'config', 'output', 'code-revision'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = audit(args.archive, args.dataset, args.baseline_archive, args.config, args.output, args.code_revision)
    print(json.dumps({key: result[key] for key in ('members_verified', 'pages', 'metrics_recomputed',
        'per_page_CER_change', 'glyph_counts', 'page_inference_seconds', 'peak_allocated_bytes')},
        ensure_ascii=False, indent=2))
