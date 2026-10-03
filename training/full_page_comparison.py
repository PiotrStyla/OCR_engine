"""Stage the frozen review bundle and compare a new profile with retained Ovis."""
from __future__ import annotations

import csv
import hashlib
from itertools import groupby
import json
from pathlib import Path
import re
import stat
from urllib.parse import quote
from urllib.request import urlopen
import zipfile

from training.benchmark_pages import evaluate
from training.full_page_pilot import (
    digest, markdown_text, read_rows, safe_relative, validate_inputs, write_json, write_rows,
)


def stage_bundle(config, output, opener=urlopen):
    from PIL import Image
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = config['dataset']
    url = (f'https://huggingface.co/datasets/{spec["repo"]}/resolve/'
           f'{spec["revision"]}/{quote(safe_relative(spec["path"]), safe="/")}')
    archive = output/'input-bundle.zip'
    if not archive.exists():
        with opener(url, timeout=120) as response:
            data = response.read(100_000_001)
        if hashlib.sha256(data).hexdigest() != spec['archive_sha256']:
            raise ValueError('Bundle checksum mismatch')
        archive.write_bytes(data)
    if digest(archive) != spec['archive_sha256']:
        raise ValueError('Cached bundle checksum mismatch')
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or sum(item.file_size for item in members) > 250_000_000:
            raise ValueError('Duplicate or oversized bundle members')
        for item in members:
            path = safe_relative(item.filename)
            if (item.is_dir() or stat.S_ISLNK(item.external_attr >> 16)
                    or Path(path).suffix not in ('.json', '.jsonl', '.png', '.xml')):
                raise ValueError('Unsafe bundle member')
        checksums = json.loads(stream.read('checksums.json'))
        if set(checksums) != set(names)-{'checksums.json'}:
            raise ValueError('Incomplete bundle checksums')
        # Validate the entire archive before materializing any reference/image files.
        for name, expected in checksums.items():
            if hashlib.sha256(stream.read(name)).hexdigest() != expected:
                raise ValueError('Bundle member checksum mismatch')
        for name in names:
            target = output/safe_relative(name)
            data = stream.read(name)
            if target.exists() and target.read_bytes() != data:
                raise ValueError('Existing staged file differs; use a new directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)
    manifest = output/'manifest.jsonl'
    baseline = output/config['baseline']['source']
    if digest(manifest) != spec['manifest_sha256'] or digest(baseline) != spec['baseline_sha256']:
        raise ValueError('Frozen manifest/baseline mismatch')
    rows = read_rows(manifest)
    baseline_rows = read_rows(baseline)
    if (len(rows) != spec['pages'] or len({row['id'] for row in rows}) != len(rows)
            or len(baseline_rows) != len(rows)
            or {row['id'] for row in baseline_rows} != {row['id'] for row in rows}):
        raise ValueError('Expected all unique paired validation pages')
    inputs = []
    for row in rows:
        if (row['split'] != 'validation' or row.get('final_test') is not False
                or row['reference_status'] != 'single-review-draft-not-gold'
                or row.get('eligible_for_training') is not False):
            raise ValueError('Unexpected page split or reference eligibility')
        image = output/safe_relative(row['image'])
        if digest(image) != row['sha256']:
            raise ValueError('Image checksum mismatch')
        with Image.open(image) as source:
            if source.size != (row['width'], row['height']):
                raise ValueError('Image dimension mismatch')
        inputs.append({key: row[key] for key in ('id', 'image', 'sha256', 'width', 'height')}
                      | {'source_regions': []})
    write_rows(output/'inference-inputs.jsonl', inputs)
    validate_inputs(output/'inference-inputs.jsonl')
    write_json(output/'staging-provenance.json', {
        'bundle_url': url, 'bundle_sha256': digest(archive), 'pages': len(rows),
        'manifest_sha256': digest(manifest), 'baseline_sha256': digest(baseline),
        'inputs_sha256': digest(output/'inference-inputs.jsonl'),
        'reference_text_sent_to_model': False, 'source_geometry_sent_to_model': False,
        'gold_pages': 0, 'training_eligibility': False})
    return {'pages': len(rows), 'gold_pages': 0, 'inputs': str(output/'inference-inputs.jsonl')}


def repetition_flags(text):
    char_run = max((len(list(group)) for char,group in groupby(text) if not char.isspace()), default=0)
    words = re.findall(r'\w+', text)
    word_run = max((len(list(group)) for _,group in groupby(words)), default=0)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_run = max((len(list(group)) for _,group in groupby(lines)), default=0)
    return {'longest_character_run': char_run, 'longest_word_run': word_run,
            'longest_identical_line_run': line_run,
            'suspicious_repetition': char_run >= 64 or word_run >= 32 or line_run >= 4,
            'method': 'Consecutive repetition heuristic; not a complete loop detector'}


def compare(config, dataset, candidate, output):
    dataset, candidate, output = map(Path, (dataset, candidate, output))
    output.mkdir(parents=True, exist_ok=True)
    manifest = dataset/'manifest.jsonl'
    baseline = dataset/config['baseline']['source']
    if (digest(manifest) != config['dataset']['manifest_sha256']
            or digest(baseline) != config['dataset']['baseline_sha256']):
        raise ValueError('Comparison inputs differ from frozen configuration')
    references = read_rows(manifest)
    raw = read_rows(candidate) if candidate.exists() else []
    projected = [{**row, 'raw_text': row['text'],
                  'text': markdown_text(row['text']) if row.get('format') == 'markdown' else row['text']}
                 for row in raw]
    candidate_score = output/'qwen3-vl-4b-projected.jsonl'
    write_rows(candidate_score, projected)
    reports, page_table = {}, []
    profile = config.get('comparison_profile', {})
    labels = profile.get('labels', ['ovis-retained', 'qwen3-vl-4b'])
    if (not isinstance(labels, (list, tuple)) or len(labels) != 2
            or not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != 2):
        raise ValueError('Comparison needs two distinct nonempty labels')
    for label,path in [(labels[0], baseline), (labels[1], candidate_score)]:
        predictions = read_rows(path)
        report = evaluate(manifest, path)
        supplied = {row['id']: row for row in predictions}
        report.update(reference_status='single-review-draft-not-gold',
                      eligible_for_model_promotion=False, content_scope_verified=False,
                      token_limit_pages=sum(bool(row.get('token_limit_reached')) for row in predictions),
                      eos_pages=sum(row.get('finish_reason') == 'eos' for row in predictions),
                      cer_macro=sum(row['cer'] for row in report['results'])/report['pages'])
        report['runtime_measurements_comparable'] = False
        for reference,result in zip(references, report['results']):
            prediction = supplied.get(reference['id'], {})
            raw_text = prediction.get('raw_text', prediction.get('text', ''))
            page_table.append({'engine': label, 'id': reference['id'], **result,
                'token_limit_reached': prediction.get('token_limit_reached'),
                'finish_reason': prediction.get('finish_reason', 'missing'),
                'generated_tokens': prediction.get('generated_tokens'),
                'elapsed_seconds': prediction.get('elapsed_seconds'),
                'peak_allocated_bytes': prediction.get('peak_allocated_bytes'),
                'raw_long_s_count': raw_text.count('\u017f'), 'raw_a_acute_count': raw_text.count('\u00e1'),
                **repetition_flags(raw_text)})
        reports[label] = report
    summary = {'schema': profile.get('schema', 'slayer-full-page-comparison-v5-result'), 'pages': len(references),
        'reports': reports, 'scope_policy': config['scope_policy'],
        'reference_status': 'single-review-draft-not-gold', 'gold_pages': 0,
        'baseline_rerun': False, 'all_pages_included': True,
        'model_promotion': False, 'sota_claim': False,
        'spatial_reading_order_metrics_available': False,
        'omission_hallucination_accuracy_available': False,
        'comparison_kind': profile.get('kind', 'Complete inference profiles; prompts, quantization and runtimes differ'),
        'claim_boundary': profile.get('claim_boundary',
                          'Provisional v2 references with unadjudicated peripheral text and ordering. '
                          'No controlled model-only causal claim. No failures or capped outputs removed.')}
    write_json(output/'metrics.json', summary)
    write_rows(output/'per-page.jsonl', page_table)
    with (output/'per-page.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(page_table[0]))
        writer.writeheader()
        writer.writerows(page_table)
    return summary


def package_comparison(work, archive_name='full-page-comparison-v5-evidence.zip'):
    if (not isinstance(archive_name, str) or '/' in archive_name or '\\' in archive_name
            or not archive_name.endswith('.zip') or archive_name.startswith('.')):
        raise ValueError('Evidence archive name must be a plain ZIP basename')
    work = Path(work)
    files = [path for path in work.rglob('*') if path.is_file()
             and not any(part.startswith('.') for part in path.relative_to(work).parts)
             and (path.suffix in ('.json', '.jsonl', '.log', '.csv')
                  or path == work/'cer-comparison.png')
             and 'model' not in path.relative_to(work).parts]
    archive = work/archive_name
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            stream.write(path, path.relative_to(work).as_posix())
        stream.writestr('checksums.json', json.dumps(
            {path.relative_to(work).as_posix(): digest(path) for path in files}, indent=2))
    return archive
