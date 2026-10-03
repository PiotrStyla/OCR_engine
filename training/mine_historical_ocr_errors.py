"""Mine diagnostic historical-glyph mismatches, never create training labels."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import zipfile

from rapidfuzz.distance import Levenshtein

from training.build_annotation_review import build as build_review
from training.full_page_pilot import digest, read_rows, safe_relative, write_json, write_rows

TARGETS = '\u017f\u00e1\u0247'
WORD = 'Po\u017f\u0142a\u0142'


def utf16_length(text):
    return len(text.encode('utf-16-le')) // 2


def mismatches(reference, candidate):
    """One minimum-edit alignment is a review proposal, not spatial/glyph truth."""
    found = []
    for kind, start, end, other_start, other_end in Levenshtein.opcodes(reference, candidate):
        before, after = reference[start:end], candidate[other_start:other_end]
        if kind == 'equal' or not any(char in before + after for char in TARGETS):
            continue
        context_start, context_end = max(0, start - 35), min(len(reference), end + 35)
        context = reference[context_start:context_end]
        found.append({'kind': kind, 'reference_span': [start, end],
            'candidate_span': [other_start, other_end],
            'offset': utf16_length(reference[:start]), 'length': utf16_length(before),
            'reference': before, 'candidate': after,
            'reference_context': context,
            'candidate_context': candidate[max(0, other_start-35):other_end+35],
            'targets': sorted(set(before + after) & set(TARGETS)),
            'priority_word_context': WORD in context,
            'spatial_alignment_verified': False,
            'review_status': 'unreviewed', 'eligible_for_training': False})
    return found


def prepare(audit_dir, output):
    audit_dir, output = Path(audit_dir), Path(output)
    if output.exists():
        raise FileExistsError('Use a new mining directory')
    audit = json.loads((audit_dir/'audit.json').read_text(encoding='utf-8'))
    if (audit.get('schema') != 'slayer-full-page-validation-4mp-v7-audit'
            or audit.get('metrics_recomputed') is not True
            or audit.get('gold_pages') != 0
            or digest(audit_dir/'source-evidence.zip') != audit['source_archive_sha256']):
        raise ValueError('Expected verified v7 evidence and unchanged source ZIP')
    evidence = audit_dir/'evidence'
    manifest = audit_dir/'verified-dataset/manifest.jsonl'
    candidate_path = evidence/'predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl'
    baseline_path = evidence/'dataset/qwen-v5-1mp.jsonl'
    with zipfile.ZipFile(audit_dir/'source-evidence.zip') as stream:
        checksums = json.loads(stream.read('checksums.json'))
        for path, name in [(manifest, 'dataset/manifest.jsonl'),
                           (candidate_path, 'predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl'),
                           (baseline_path, 'dataset/qwen-v5-1mp.jsonl')]:
            data = stream.read(name)
            if hashlib.sha256(data).hexdigest() != checksums[name] or path.read_bytes() != data:
                raise ValueError('Local source differs from checksum-bound ZIP')
    rows = read_rows(manifest)
    baseline = read_rows(baseline_path)
    candidate = read_rows(candidate_path)
    ids = {row['id'] for row in rows}
    if (len(rows) != audit['pages'] or len(ids) != len(rows)
            or any(row['split'] != 'validation' or row['eligible_for_training'] is not False
                   or row['final_test'] is not False for row in rows)):
        raise ValueError('Mining is restricted to non-training validation diagnostics')
    for predictions in (baseline, candidate):
        if len(predictions) != len(rows) or {row['id'] for row in predictions} != ids:
            raise ValueError('Incomplete or duplicate prediction coverage')
    baseline, candidate = ({row['id']: row for row in predictions}
                           for predictions in (baseline, candidate))
    for row in rows:
        image = manifest.parent/safe_relative(row['image'])
        if digest(image) != row['sha256']:
            raise ValueError('Source scan checksum mismatch')
    diagnostics, queue = {}, []
    for row in rows:
        prediction = candidate[row['id']]
        items = mismatches(row['text'], prediction['text'])
        items.sort(key=lambda item: (not item['priority_word_context'], item['reference_span']))
        for item in items:
            identity = json.dumps([row['id'], digest(candidate_path), item['reference_span'],
                                   item['candidate_span']], separators=(',', ':'))
            item['id'] = hashlib.sha256(identity.encode()).hexdigest()
            queue.append({'page_id': row['id'], 'collection': row['collection'],
                'reference_status': row['reference_status'], 'image_sha256': row['sha256'], **item})
        diagnostics[row['id']] = {'items': items, 'candidate_text': prediction['text'],
            'candidate_status': prediction['status'], 'baseline_text': baseline[row['id']]['text'],
            'candidate_label': 'Qwen v7 4MP', 'baseline_label': 'Qwen v5 1MP'}
    # Priority affects presentation only; every mined edit remains in the queue.
    queue.sort(key=lambda item: (not item['priority_word_context'], item['page_id'], item['reference_span']))
    ordered = sorted(rows, key=lambda row: (
        not any(item['priority_word_context'] for item in diagnostics[row['id']]['items']),
        -len(diagnostics[row['id']]['items']), row['id']))
    output.mkdir(parents=True)
    (output/'images').mkdir()
    review_rows = []
    for row in ordered:
        destination = output/safe_relative(row['image'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest.parent/row['image'], destination)
        review_rows.append(dict(row))
    write_rows(output/'review-manifest.jsonl', review_rows)
    write_rows(output/'hard-examples.jsonl', queue)
    write_json(output/'diagnostics.json', diagnostics)
    build_review(output/'review-manifest.jsonl', output/'review', diagnostics=diagnostics)
    shutil.copyfile(manifest, output/'original-manifest.jsonl')
    shutil.copyfile(candidate_path, output/'raw-v7-predictions.jsonl')
    shutil.copyfile(baseline_path, output/'raw-v5-predictions.jsonl')
    report = {'schema': 'slayer-historical-glyph-mining-v1', 'pages': len(rows),
        'pages_with_mismatches': sum(bool(value['items']) for value in diagnostics.values()),
        'candidate_edit_spans': len(queue),
        'word_context_edit_spans': sum(item['priority_word_context'] for item in queue),
        'targets': list(TARGETS), 'gold_pages': 0, 'training_examples_created': 0,
        'normalization': 'none; raw text preserved; offsets include browser UTF-16',
        'alignment': 'RapidFuzz Levenshtein opcodes; one minimum-edit alignment, not glyph recall',
        'rapidfuzz_version': importlib.metadata.version('rapidfuzz'),
        'source_archive_sha256': audit['source_archive_sha256'],
        'source_manifest_sha256': digest(manifest),
        'candidate_predictions_sha256': digest(candidate_path),
        'baseline_predictions_sha256': digest(baseline_path),
        'source_audit_sha256': digest(audit_dir/'audit.json'),
        'eligible_for_training': False, 'spatial_alignment_verified': False,
        'limits': 'Validation-only diagnostic review. No automatic crops, gold, training labels or SOTA claim.'}
    write_json(output/'report.json', report)
    write_json(output/'checksums.json', {path.relative_to(output).as_posix(): digest(path)
                                       for path in output.rglob('*') if path.is_file()})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.audit, args.output), ensure_ascii=False, indent=2))
