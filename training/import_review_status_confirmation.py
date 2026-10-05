"""Record explicit status-only confirmations without editing review history/text."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from training.adjudicate_reviews import reviewer_key, text_hash, validate
from training.full_page_pilot import digest, read_rows, safe_relative, write_json


def apply(manifest, review, confirmation, output):
    manifest, review, confirmation, output = map(Path, (manifest, review, confirmation, output))
    if output.exists():
        raise FileExistsError('Use a new confirmation directory')
    rows = read_rows(manifest)
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Expected unique source pages')
    packet = json.loads(review.read_text(encoding='utf-8-sig'))
    events = validate(packet, rows, digest(manifest))
    if len({reviewer_key(event['reviewer']) for event in events}) != 1:
        raise ValueError('Expected exactly one reviewer for a scoped status confirmation')
    if not events or any(event['before'] != event['after'] for event in events):
        raise ValueError('This importer is status-only; text edits require separate handling')
    authorization = json.loads(confirmation.read_text(encoding='utf-8'))
    if (authorization.get('schema') != 'polocrbench-review-status-confirmation-v1'
            or authorization.get('approved') is not True
            or not isinstance(authorization.get('user_message'), str)
            or not authorization['user_message'].strip()
            or authorization.get('source_manifest_sha256') != digest(manifest)
            or authorization.get('source_review_sha256') != digest(review)
            or authorization.get('from_decision') != 'needs-review'
            or authorization.get('to_decision') != 'verified'):
        raise ValueError('Explicit source-bound status confirmation required')
    targets = authorization.get('page_ids')
    if (not isinstance(targets, list) or not targets
            or not all(isinstance(identifier, str) for identifier in targets)
            or len(targets) != len(set(targets))):
        raise ValueError('Unique nonempty confirmation page IDs required')
    latest = {event['page_id']: event for event in events}
    if any(identifier not in latest or latest[identifier]['decision'] != 'needs-review'
           for identifier in targets):
        raise ValueError('Confirmation targets must have a latest needs-review event')
    for row in rows:
        if (row.get('eligible_for_training') is not False or row.get('final_test') is not False
                or row.get('split') != 'validation'
                or digest(manifest.parent/safe_relative(row['image'])) != row['sha256']):
            raise ValueError('Expected unchanged non-training validation sources')
    statuses = []
    for row in rows:
        event = latest.get(row['id'])
        original = event['decision'] if event else 'unreviewed'
        statuses.append({'id': row['id'], 'original_decision': original,
            'effective_decision': 'verified' if row['id'] in targets else original,
            'evidence_event_id': event['id'] if event else None,
            'reviewer': event['reviewer'] if event else None,
            'explicit_user_confirmation': row['id'] in targets,
            'text_sha256': text_hash(row['text']), 'image_sha256': row['sha256'],
            'eligible_for_training': False, 'reference_status': 'single-review-not-gold'})
    report = {'schema': 'polocrbench-review-status-confirmation-result-v1',
        'source_manifest_sha256': digest(manifest), 'source_review_sha256': digest(review),
        'confirmation_sha256': digest(confirmation), 'pages': len(rows), 'events': len(events),
        'decisions': dict(Counter(row['effective_decision'] for row in statuses)),
        'confirmed_pages': targets, 'changed_texts': 0, 'predictions_modified': False,
        'gold_pages': 0, 'training_examples_created': 0, 'sota_claim': False,
        'original_review_unchanged': True, 'review_events_synthesized': False,
        'user_message': authorization['user_message'],
        'claim_boundary': 'Explicit single-review status confirmation, not independent adjudication.',
        'results': statuses}
    output.mkdir(parents=True)
    shutil.copyfile(review, output/'original-review.json')
    shutil.copyfile(confirmation, output/'status-confirmation.json')
    shutil.copyfile(manifest, output/'source-manifest-snapshot.jsonl')
    write_json(output/'review-status-report.json', report)
    write_json(output/'checksums.json', {path.name: digest(path) for path in output.iterdir()
                                       if path.is_file()})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'review', 'confirmation', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = apply(args.manifest, args.review, args.confirmation, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'results'}, indent=2))
