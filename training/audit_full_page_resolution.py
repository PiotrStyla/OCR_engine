"""Audit returned v6 evidence and recompute the paired scores without model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import zipfile

from training.benchmark_pages import evaluate
from training.full_page_pilot import digest, read_rows, safe_relative, write_json, write_rows
from training.full_page_resolution import ARMS, ENGINE, PAGE_IDS, arm_configs, compare


ROOT = Path(__file__).resolve().parents[1]


def audit(archive_path, dataset, config_path, output, code_revision):
    archive_path, dataset, config_path, output = map(Path, (archive_path, dataset, config_path, output))
    if output.exists():
        raise FileExistsError('Use a new audit directory')
    if len(code_revision) != 40 or any(c not in '0123456789abcdef' for c in code_revision):
        raise ValueError('Use the full published inference revision')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    arms = arm_configs(config)
    expected_code = {'code_revision': code_revision}
    for key, filename in [('runner_sha256', 'full_page_pilot.py'),
                          ('comparison_sha256', 'full_page_comparison.py'),
                          ('resolution_sha256', 'full_page_resolution.py')]:
        expected_code[key] = hashlib.sha256(subprocess.check_output(
            ['git', 'show', f'{code_revision}:training/{filename}'], cwd=ROOT)).hexdigest()
    with zipfile.ZipFile(archive_path) as stream:
        members = stream.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or sum(item.file_size for item in members) > 100_000_000:
            raise ValueError('Duplicate or oversized evidence members')
        for item in members:
            name = safe_relative(item.filename)
            if (item.is_dir() or stat.S_ISLNK(item.external_attr >> 16)
                    or (Path(name).suffix not in ('.json', '.jsonl', '.log', '.csv')
                        and name != 'cer-comparison.png')):
                raise ValueError('Unexpected evidence member')
        checksums = json.loads(stream.read('checksums.json'))
        if set(checksums) != set(names)-{'checksums.json'}:
            raise ValueError('Evidence checksum coverage mismatch')
        payload = {name: stream.read(name) for name in names}
        if any(hashlib.sha256(payload[name]).hexdigest() != sha for name, sha in checksums.items()):
            raise ValueError('Evidence checksum mismatch')
    decode = lambda name: json.loads(payload[name])
    rows = lambda name: [json.loads(line) for line in payload[name].splitlines() if line.strip()]
    if decode('config.json') != config:
        raise ValueError('Frozen config mismatch')
    probes = [json.loads(line) for line in payload['preflight.log'].decode('utf-8').splitlines()
              if line.startswith('{')]
    required = dict(item.split('==') for item in config['models'][ENGINE]['packages'])
    if len(probes) != 1 or probes[0].get('packages') != required:
        raise ValueError('Recorded import-preflight package versions mismatch')
    for filename, keys in [('code-provenance.json', ('code_revision', 'runner_sha256', 'comparison_sha256')),
                           ('resolution-code-provenance.json', ('code_revision', 'resolution_sha256'))]:
        if any(decode(filename).get(key) != expected_code[key] for key in keys):
            raise ValueError('Pinned code provenance mismatch')
    for filename, expected in [('manifest.jsonl', config['dataset']['manifest_sha256']),
                               (config['baseline']['source'], config['dataset']['baseline_sha256'])]:
        if (digest(dataset/filename) != expected
                or hashlib.sha256(payload[f'dataset/{filename}']).hexdigest() != expected):
            raise ValueError('Frozen source manifest/baseline mismatch')
    source = read_rows(dataset/'manifest.jsonl')
    if (len(source) != config['dataset']['pages'] or len({row['id'] for row in source}) != len(source)
            or any(row['split'] != 'validation' or row.get('final_test') is not False
                   or row.get('eligible_for_training') for row in source)):
        raise ValueError('Unexpected source page scope')
    by_id = {row['id']: row for row in source}
    selected = [by_id[key] for key in PAGE_IDS]
    if rows('dataset/resolution-manifest.jsonl') != selected:
        raise ValueError('Subset reference mismatch')
    for row in source:
        if digest(dataset/row['image']) != row['sha256']:
            raise ValueError('Source image checksum mismatch')
    sanitize = lambda records: [{key: row[key] for key in ('id', 'image', 'sha256', 'width', 'height')}
                               | {'source_regions': []} for row in records]
    if (rows('dataset/resolution-inputs.jsonl') != sanitize(selected)
            or rows('dataset/inference-inputs.jsonl') != sanitize(source)):
        raise ValueError('Reference-free inference inputs mismatch')
    staging = decode('dataset/staging-provenance.json')
    if (staging['bundle_sha256'] != config['dataset']['archive_sha256']
            or staging['inputs_sha256'] != hashlib.sha256(payload['dataset/inference-inputs.jsonl']).hexdigest()
            or staging['reference_text_sent_to_model'] is not False
            or staging['source_geometry_sent_to_model'] is not False):
        raise ValueError('Staging provenance mismatch')
    input_hash = hashlib.sha256(payload['dataset/resolution-inputs.jsonl']).hexdigest()
    selection = decode('dataset/resolution-selection.json')
    if (selection['selection'] != config['selection'] or selection['pages'] != 3
            or selection['source_manifest_sha256'] != config['dataset']['manifest_sha256']
            or selection['subset_manifest_sha256'] != hashlib.sha256(
                payload['dataset/resolution-manifest.jsonl']).hexdigest()
            or selection['inference_inputs_sha256'] != input_hash
            or selection['reference_text_sent_to_model'] is not False
            or selection['reference_geometry_sent_to_model'] is not False):
        raise ValueError('Selection provenance mismatch')
    environments, predictions = {}, {}
    for arm, arm_config in arms.items():
        prefix = f'arms/{arm}/predictions'
        spec = arm_config['models'][ENGINE]
        if decode(f'arms/{arm}/config.json') != arm_config:
            raise ValueError('Arm configuration mismatch')
        if decode(f'{prefix}/identity.json') != {
                'engine': ENGINE, 'spec': spec, 'input_sha256': input_hash,
                'runner_sha256': expected_code['runner_sha256']}:
            raise ValueError('Arm worker identity mismatch')
        environment = decode(f'{prefix}/environment.json')
        packages = {key.lower(): value for key, value in environment['packages'].items()}
        for package in spec['packages']:
            name, version = package.split('==')
            if name != 'sentencepiece' and packages.get(name.lower()) != version:
                raise ValueError('Worker package version mismatch')
        if environment['reference_text_sent_to_model'] is not False:
            raise ValueError('Worker reference leakage attestation mismatch')
        candidate = rows(f'{prefix}/{ENGINE}-full-page.jsonl')
        if len(candidate) != 3 or {row['id'] for row in candidate} != set(PAGE_IDS):
            raise ValueError('Incomplete or duplicate arm coverage')
        for prediction in candidate:
            if prediction['status'] != 'ok':
                continue
            if prediction['format'] != 'plain':
                raise ValueError('Expected raw plain-text output')
            trace = prediction['generation_trace']
            tokens = trace['generated_token_ids']
            eos = bool(tokens and tokens[-1] in trace['eos_token_ids'])
            capped = len(tokens) >= spec['max_new_tokens'] and not eos
            reason = 'eos' if eos else ('length' if capped else 'unknown')
            if (prediction['generated_tokens'] != len(tokens) or prediction['token_limit_reached'] != capped
                    or prediction['finish_reason'] != reason):
                raise ValueError('Token/termination mismatch')
            geometry = prediction['input_geometry']
            grid, patch, merge = geometry['image_grid_thw'], geometry['patch_size'], geometry['merge_size']
            reference = by_id[prediction['id']]
            expected_geometry = {
                'processed_width': grid[0][2]*patch, 'processed_height': grid[0][1]*patch,
                'processed_pixels': grid[0][1]*grid[0][2]*patch*patch,
                'visual_tokens': sum(t*h*w//(merge*merge) for t, h, w in grid),
                'original_width': reference['width'], 'original_height': reference['height'],
                'min_pixels': spec['min_pixels'], 'max_pixels': spec['max_pixels']}
            if (any(geometry.get(key) != value for key, value in expected_geometry.items())
                    or not spec['min_pixels'] <= geometry['processed_pixels'] <= spec['max_pixels']):
                raise ValueError('Processed input geometry mismatch')
        environments[arm], predictions[arm] = environment, candidate
    output.mkdir(parents=True)
    extracted = output/'evidence'
    for name, data in payload.items():
        target = extracted/safe_relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # Keep returned scores immutable; recompute against separately verified local images.
    recompute_work = output/'recomputed'
    shutil.copytree(extracted/'arms', recompute_work/'arms')
    local_dataset = output/'verified-dataset'
    local_dataset.mkdir()
    for filename in ('manifest.jsonl', 'resolution-manifest.jsonl', 'resolution-inputs.jsonl'):
        (local_dataset/filename).write_bytes(payload[f'dataset/{filename}'])
    for row in selected:
        destination = local_dataset/safe_relative(row['image'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset/row['image'], destination)
    recomputed = compare(config, local_dataset, recompute_work)
    if recomputed != decode('scores/metrics.json'):
        raise ValueError('Recomputed metrics mismatch')
    if read_rows(recompute_work/'scores/per-page.jsonl') != rows('scores/per-page.jsonl'):
        raise ValueError('Recomputed per-page diagnostics mismatch')
    body_rows = [row for row in selected if row['id'] != PAGE_IDS[0]]
    write_rows(local_dataset/'body-only-manifest.jsonl', body_rows)
    body_reports = {}
    for arm, candidate in predictions.items():
        path = recompute_work/f'{arm}-body-only.jsonl'
        write_rows(path, [row for row in candidate if row['id'] != PAGE_IDS[0]])
        body_reports[arm] = evaluate(local_dataset/'body-only-manifest.jsonl', path)
    report = {'schema': 'slayer-full-page-resolution-v6-audit',
        'source_archive_sha256': digest(archive_path), 'members_verified': len(names),
        'metrics_recomputed': True, 'pages_per_arm': 3, 'code_provenance': expected_code,
        'runtime_reported': decode('runtime.json'), 'worker_environments': environments,
        'import_preflight_recorded': probes[0],
        'package_verification_scope': 'All pinned model packages checked in recorded import preflight; '
                                      'all packages recorded per worker independently checked.',
        'process_results': {arm: decode(f'arms/{arm}/process-result.json') for arm in ARMS},
        'reports': recomputed['reports'], 'paired': recomputed['paired'],
        'body_only_diagnostic': {'primary_metric': False,
            'scope': 'Two dense-text pages, title page excluded only for explanatory diagnostics',
            'reports': body_reports},
        'glyph_counts': {char: {'reference': sum(row['text'].count(char) for row in selected),
                               **{arm: sum(row['text'].count(char) for row in candidate)
                                  for arm, candidate in predictions.items()}}
                        for char in ('\u017f', '\u00e1', '\u0247', 'Po\u017f\u0142a\u0142')},
        'glyph_counts_are_recall': False,
        'page_inference_seconds': {arm: sum(row['elapsed_seconds'] for row in candidate)
                                   for arm, candidate in predictions.items()},
        'peak_allocated_bytes': {arm: max(row['peak_allocated_bytes'] for row in candidate)
                                 for arm, candidate in predictions.items()},
        'gold_pages': 0, 'model_promotion': False, 'automatic_teacher_promotion': False,
        'sota_claim': False, 'claim_boundary': recomputed['claim_boundary']}
    shutil.copyfile(archive_path, output/'source-evidence.zip')
    write_json(output/'audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'dataset', 'config', 'output', 'code-revision'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = audit(args.archive, args.dataset, args.config, args.output, args.code_revision)
    print(json.dumps(result, ensure_ascii=False, indent=2))
