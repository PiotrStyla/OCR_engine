"""Apply scoped, explicit text confirmations without changing page review status."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

from training.adjudicate_reviews import text_hash
from training.build_annotation_review import issues
from training.full_page_pilot import digest, read_rows, safe_relative, write_json, write_rows


def apply(source, confirmation, output):
    source, confirmation, output = map(Path, (source, confirmation, output))
    if output.exists():
        raise FileExistsError('Use a new confirmation directory')
    packet = json.loads(confirmation.read_text(encoding='utf-8'))
    if (packet.get('schema') != 'polocrbench-scoped-text-confirmation-v1'
            or packet.get('approved') is not True or not packet.get('user_message')):
        raise ValueError('Explicit scoped confirmation required')
    if packet['parent_manifest_sha256'] != digest(source/'manifest.jsonl'):
        raise ValueError('Parent manifest mismatch')
    checksums = json.loads((source/'checksums.json').read_text(encoding='utf-8'))
    for name, expected in checksums.items():
        if digest(source/safe_relative(name)) != expected:
            raise ValueError('Parent file checksum mismatch')
    rows = read_rows(source/'manifest.jsonl')
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows) or not packet.get('corrections'):
        raise ValueError('Unique pages and nonempty corrections required')
    if len({item['id'] for item in packet['corrections']}) != len(packet['corrections']):
        raise ValueError('Unique correction IDs required')
    changes = []
    for item in packet['corrections']:
        row = by_id[item['page_id']]
        if row['reference_status'] != 'single-review-draft-not-gold':
            raise ValueError('Expected a provisional single-review draft')
        before, after = item['before'], item['after']
        if not before or before == after or row['text'].count(before) != 1:
            raise ValueError('Correction must match exactly one unchanged context')
        row.setdefault('pre_confirmation_text', row['text'])
        old_hash = text_hash(row['text'])
        row['text'] = row['text'].replace(before, after, 1)
        row.setdefault('confirmed_text_correction_ids', []).append(item['id'])
        changes.append({**item, 'before_page_sha256': old_hash,
                        'after_page_sha256': text_hash(row['text'])})
    output.mkdir(parents=True)
    for name in checksums:
        src = source/safe_relative(name)
        target = output/safe_relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    for name in ('manifest.jsonl', 'draft-report.json', 'diagnostic-comparison.json'):
        if (source/name).exists():
            shutil.copyfile(source/name, output/('parent-'+name))
    shutil.copyfile(confirmation, output/'text-confirmation.json')
    write_rows(output/'manifest.jsonl', rows)
    report = json.loads((source/'draft-report.json').read_text(encoding='utf-8'))
    report.update(schema='polocrbench-confirmed-draft-v2',
                  parent_manifest_sha256=digest(source/'manifest.jsonl'),
                  draft_manifest_sha256=digest(output/'manifest.jsonl'),
                  confirmation_sha256=digest(confirmation), confirmed_changes=changes,
                  confirmation_scope='Listed text contexts only; not full-page verification')
    for result in report['results']:
        row = by_id[result['id']]
        result.update(draft_text_sha256=text_hash(row['text']),
                      draft_unicode_issues=len(issues(row['text'])),
                      changed=row['text'] != row['source_reference_text'])
    report['draft_unicode_issues'] = sum(item['draft_unicode_issues'] for item in report['results'])
    report['changed_pages'] = sum(item['changed'] for item in report['results'])
    predictions = output/'retained-projected-predictions.jsonl'
    if predictions.exists():
        from training.benchmark_pages import evaluate
        comparison = json.loads((source/'diagnostic-comparison.json').read_text(encoding='utf-8'))
        comparison['draft'] = evaluate(output/'manifest.jsonl', predictions)
        comparison['draft_manifest_sha256'] = report['draft_manifest_sha256']
        write_json(output/'diagnostic-comparison.json', comparison)
        report['diagnostic_cer_micro']['draft'] = comparison['draft']['cer_micro']
        report['diagnostic_wer_micro']['draft'] = comparison['draft']['wer_micro']
    write_json(output/'draft-report.json', report)
    files = [path for path in output.rglob('*') if path.is_file()]
    write_json(output/'checksums.json', {path.relative_to(output).as_posix(): digest(path) for path in files})
    archive = output/'full-page-validation-v4-confirmed-draft-v2.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for path in files+[output/'checksums.json']:
            stream.write(path, path.relative_to(output).as_posix())
    return {'archive': str(archive), 'archive_sha256': digest(archive),
            'manifest_sha256': report['draft_manifest_sha256'], 'changes': len(changes)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--confirmation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(apply(args.source, args.confirmation, args.output), indent=2))
