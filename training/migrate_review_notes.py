"""Explicitly promote user-confirmed transcription notes to a new review draft."""
import argparse
import json
from pathlib import Path
import shutil

from training.adjudicate_reviews import validate
from training.build_annotation_review import build
from training.stage_impact_benchmark import digest, read_rows


def migrate(manifest, review, output, *, notes_are_transcriptions=False):
    if not notes_are_transcriptions:
        raise ValueError('Explicit confirmation that notes are transcriptions is required')
    manifest, review, output = Path(manifest), Path(review), Path(output)
    if output.exists():
        raise FileExistsError('Use a new draft directory')
    rows = read_rows(manifest)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Expected unique nonempty source records')
    packet = json.loads(review.read_text(encoding='utf-8-sig'))
    events = validate(packet, rows, digest(manifest))
    latest = {e['page_id']: e for e in events}
    for row in rows:
        source = (manifest.parent / row['image']).resolve()
        if not source.is_relative_to(manifest.parent.resolve()) or digest(source) != row['sha256']:
            raise ValueError('Invalid image path or checksum')
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    candidates, changes = [], []
    for index, row in enumerate(rows):
        event = latest.get(row['id'])
        use_note = bool(event and event['note'].strip())
        text = event['note'] if use_note else event['after'] if event else row['text']
        relative = f'images/{index:04d}{Path(row["image"]).suffix}'
        shutil.copyfile(manifest.parent / row['image'], output / relative)
        candidates.append({**row, 'image': relative, 'text': text,
                           'original_text': row['text'], 'review_status': 'draft-needs-confirmation',
                           'eligible_for_evaluation': False,
                           'source_review_decision': event['decision'] if event else None,
                           'source_review_event_id': event['id'] if event else None})
        changes.append({'id': row['id'], 'from_note': use_note, 'changed': text != row['text'],
                        'original_text': row['text'], 'review_text': event['after'] if event else None,
                        'note': event['note'] if event else None, 'draft_text': text,
                        'source_decision': event['decision'] if event else None})
    target = output / 'manifest.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in candidates), encoding='utf-8')
    shutil.copyfile(review, output / 'original-review.json')
    shutil.copyfile(manifest, output / 'original-manifest.jsonl')
    report = {'schema': 'review-note-migration-v1', 'status': 'draft-not-gold',
              'source_manifest_sha256': digest(manifest), 'review_sha256': digest(review),
              'output_manifest_sha256': digest(target), 'records': len(rows),
              'notes_promoted': sum(r['from_note'] for r in changes),
              'changed_records': sum(r['changed'] for r in changes),
              'policy': 'User explicitly confirmed notes are transcriptions. Preserve text verbatim; no approval events synthesized. Latest means export event order, as in the review UI.',
              'changes': changes}
    (output / 'migration.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    build(target, output / 'review')
    checks = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()}
    (output / 'checksums.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--notes-are-transcriptions', action='store_true')
    args = parser.parse_args()
    result = migrate(args.manifest, args.review, args.output, notes_are_transcriptions=args.notes_are_transcriptions)
    print(json.dumps({k: v for k, v in result.items() if k != 'changes'}, indent=2))


if __name__ == '__main__':
    main()
