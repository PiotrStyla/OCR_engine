"""Full validation at the frozen 4MP profile, compared with audited retained v5."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import quote
from urllib.request import urlopen
from uuid import uuid4

from training.audit_full_page_comparison import audit
from training.full_page_comparison import compare as compare_profiles
from training.full_page_pilot import digest, read_rows, safe_relative, write_json


ROOT = Path(__file__).resolve().parents[1]
V5_CONFIG = ROOT/'experiments/2026-10-03/full-page-comparison-v5/config.json'
ENGINE = 'qwen3-vl-4b'


def validate_profile(config):
    previous = json.loads(V5_CONFIG.read_text(encoding='utf-8'))
    models = copy.deepcopy(previous['models'])
    models[ENGINE]['max_pixels'] = 4194304
    if config['models'] != models or config['dataset'] != previous['dataset']:
        raise ValueError('v7 must keep the exact v5 data/model profile except max_pixels')
    if config['retained_baseline']['inference_code_revision'] != '0d74a667f94901516dd9d3f75f9e478ca63502be':
        raise ValueError('Use the audited v5 baseline revision')
    return previous


def stage_baseline(config, dataset, work, opener=urlopen):
    previous = validate_profile(config)
    dataset, work = Path(dataset), Path(work)
    spec = config['retained_baseline']
    url = (f'https://huggingface.co/datasets/{spec["repo"]}/resolve/{spec["revision"]}/'
           f'{quote(safe_relative(spec["path"]), safe="/")}')
    baseline_root = work/'baseline'
    baseline_root.mkdir(parents=True, exist_ok=True)
    archive = baseline_root/'source-evidence.zip'
    if not archive.exists():
        with opener(url, timeout=120) as response:
            data = response.read(5_000_001)
        if hashlib.sha256(data).hexdigest() != spec['archive_sha256']:
            raise ValueError('Retained baseline archive checksum mismatch')
        archive.write_bytes(data)
    if digest(archive) != spec['archive_sha256']:
        raise ValueError('Cached baseline archive checksum mismatch')
    # Reuse the independent v5 auditor, including safe ZIP parsing and frozen-code checks.
    audit_dir = baseline_root/f'audit-{uuid4().hex}'
    report = audit(archive, dataset, V5_CONFIG, audit_dir, spec['inference_code_revision'])
    source = audit_dir/'evidence/predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl'
    if digest(source) != spec['raw_predictions_sha256']:
        raise ValueError('Audited baseline prediction checksum mismatch')
    destination = dataset/'qwen-v5-1mp.jsonl'
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise ValueError('Existing staged baseline differs; use a fresh workspace')
    if not destination.exists():
        shutil.copyfile(source, destination)
    baseline_model = previous['models'][ENGINE]
    if {**baseline_model, 'max_pixels': 4194304} != config['models'][ENGINE]:
        raise ValueError('More than the pixel budget changed')
    provenance = {'archive_url': url, 'archive_sha256': digest(archive),
        'raw_predictions_sha256': digest(destination), 'members_verified': report['members_verified'],
        'v5_metrics_recomputed': report['metrics_recomputed'], 'baseline_rerun': False,
        'inference_code_revision': spec['inference_code_revision'],
        'only_model_spec_change': 'max_pixels', 'same_session_comparison': False,
        'runtime_measurements_comparable': False, 'audit_directory': audit_dir.relative_to(work).as_posix()}
    write_json(work/'baseline-provenance.json', provenance)
    return provenance


def compare(config, dataset, candidate, output):
    validate_profile(config)
    dataset, candidate, output = map(Path, (dataset, candidate, output))
    baseline = dataset/'qwen-v5-1mp.jsonl'
    if digest(baseline) != config['retained_baseline']['raw_predictions_sha256']:
        raise ValueError('Staged v5 baseline checksum mismatch')
    predictions = read_rows(candidate) if candidate.exists() else []
    if any(row.get('format', 'plain') != 'plain' for row in predictions):
        raise ValueError('v7 requires raw plain-text predictions')
    scoring_config = copy.deepcopy(config)
    scoring_config['baseline'] = {'source': baseline.name, 'engine': ENGINE, 'rerun': False}
    scoring_config['dataset']['baseline_sha256'] = config['retained_baseline']['raw_predictions_sha256']
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'scoring-config.json', scoring_config)
    summary = compare_profiles(scoring_config, dataset, candidate, output)
    references = read_rows(dataset/'manifest.jsonl')
    by_id = {row['id']: row for row in predictions}
    baseline_by_id = {row['id']: row for row in read_rows(baseline)}
    geometry = []
    for reference in references:
        key = reference['id']
        current = by_id.get(key, {})
        previous = baseline_by_id[key]
        current_geometry = current.get('input_geometry', {})
        old_geometry = previous.get('input_geometry', {})
        one, four = old_geometry.get('processed_pixels'), current_geometry.get('processed_pixels')
        geometry.append({'id': key, 'candidate_status': current.get('status', 'missing'),
            'one_mp_input': old_geometry, 'four_mp_input': current_geometry,
            'processed_pixels_increased': four > one if isinstance(one, int) and isinstance(four, int) else None,
            'reference_long_s_count': reference['text'].count('\u017f'),
            'one_mp_long_s_count': previous['text'].count('\u017f'),
            'four_mp_long_s_count': current.get('text', '').count('\u017f'),
            'reference_poslal_count': reference['text'].count('Po\u017f\u0142a\u0142'),
            'four_mp_poslal_count': current.get('text', '').count('Po\u017f\u0142a\u0142')})
    write_json(output/'input-and-glyph-diagnostics.json', {
        'pages': geometry, 'character_counts_are_recall': False,
        'same_session_comparison': False, 'runtime_measurements_comparable': False})
    return summary
