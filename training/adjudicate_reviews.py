"""Validate review exports and build a versioned candidate without changing sources."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
import unicodedata
from pathlib import Path

from training.build_annotation_review import issues
from training.stage_impact_benchmark import digest, read_rows

DECISIONS = {'needs-review', 'proposed', 'verified'}


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def reviewer_key(name):
    return ' '.join(unicodedata.normalize('NFC', name).casefold().split())


def event_time(value):
    if not isinstance(value, str):
        raise ValueError('Invalid event timestamp')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Invalid event timestamp') from error
    if result.tzinfo is None:
        raise ValueError('Event timestamps must include a timezone')
    return result.astimezone(timezone.utc)


def validate(packet, rows, manifest_hash):
    if (not isinstance(packet, dict) or packet.get('schema') != 'polocrbench-review-patch-v1'
            or packet.get('manifest_sha256') != manifest_hash or not isinstance(packet.get('events'), list)):
        raise ValueError('Review schema or source manifest mismatch')
    pages = {row['id']: row for row in rows}
    texts = {row['id']: row['text'] for row in rows}
    seen = set()
    last_times = {}
    for event in packet['events']:
        if not isinstance(event, dict):
            raise ValueError('Invalid event')
        page = pages.get(event.get('page_id'))
        identifier = event.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in seen or page is None:
            raise ValueError('Unknown page or duplicate/invalid event ID')
        if (event.get('image_sha256') != page['sha256'] or
                event.get('original_text_sha256') != text_hash(page['text']) or
                event.get('before') != texts[page['id']]):
            raise ValueError('Review source identity or text history mismatch')
        if (not isinstance(event.get('after'), str) or len(event['after']) > 200000 or
                not isinstance(event.get('reviewer'), str) or not reviewer_key(event['reviewer']) or
                len(event['reviewer']) > 100 or not isinstance(event.get('note'), str) or
                len(event['note']) > 2000 or event.get('decision') not in DECISIONS):
            raise ValueError('Invalid event fields')
        timestamp = event_time(event.get('timestamp'))
        author_page = (reviewer_key(event['reviewer']), page['id'])
        if author_page in last_times and timestamp < last_times[author_page]:
            raise ValueError('Reviewer chronology moves backwards')
        last_times[author_page] = timestamp
        seen.add(identifier)
        texts[page['id']] = event['after']
    return packet['events']


def adjudicate(manifest, reviews, output):
    manifest, output = Path(manifest), Path(output)
    rows = read_rows(manifest)
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Expected unique nonempty source pages')
    if output.exists():
        raise FileExistsError('Use a new candidate directory')
    source_hash = digest(manifest)
    # Validate all inputs before creating any output.
    for row in rows:
        if digest(manifest.parent / row['image']) != row['sha256']:
            raise ValueError(f"Image checksum mismatch: {row['id']}")
    all_events, inputs = {}, []
    for review in map(Path, reviews):
        packet = json.loads(review.read_text(encoding='utf-8'))
        for event in validate(packet, rows, source_hash):
            if event['id'] in all_events and all_events[event['id']] != event:
                raise ValueError('Conflicting records for the same event ID')
            all_events[event['id']] = event
        inputs.append({'sha256': digest(review), 'events': len(packet['events'])})

    decisions = []
    for row in rows:
        per_reviewer = {}
        for event in all_events.values():
            if event['page_id'] == row['id']:
                per_reviewer.setdefault(reviewer_key(event['reviewer']), []).append(event)
        latest, ambiguous = [], False
        for candidates in per_reviewer.values():
            newest = max(event_time(event['timestamp']) for event in candidates)
            tied = [event for event in candidates if event_time(event['timestamp']) == newest]
            if len({(event['decision'], event['after']) for event in tied}) > 1:
                ambiguous = True
            latest.append(sorted(tied, key=lambda event: event['id'])[0])
        if not latest:
            status = 'unreviewed'
        elif ambiguous:
            status = 'ambiguous-reviewer-history'
        elif any(event['decision'] != 'verified' for event in latest):
            status = 'pending'
        elif len({event['after'] for event in latest}) > 1:
            status = 'conflict'
        elif len(latest) < 2:
            status = 'single-review'
        else:
            status = 'agreed'
        after = latest[0]['after'] if status == 'agreed' else row['text']
        decisions.append({'id': row['id'], 'status': status,
                          'reviewers': [event['reviewer'] for event in latest],
                          'evidence_event_ids': [event['id'] for event in latest],
                          'before_sha256': text_hash(row['text']), 'after_sha256': text_hash(after),
                          'changed': after != row['text'], 'remaining_flags': len(issues(after)),
                          'text': after})

    output.mkdir(parents=True)
    (output / 'images').mkdir()
    candidate = []
    for index, (row, decision) in enumerate(zip(rows, decisions)):
        suffix = Path(row['image']).suffix
        relative = f'images/{index:04d}{suffix}'
        shutil.copyfile(manifest.parent / row['image'], output / relative)
        candidate.append({**row, 'image': relative, 'text': decision['text']})
    target = output / 'manifest.jsonl'
    target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in candidate),
                      encoding='utf-8', newline='\n')
    report = {'schema': 'polocrbench-adjudication-v1', 'release_status': 'candidate-not-published',
              'source_manifest_sha256': source_hash, 'candidate_manifest_sha256': digest(target),
              'review_inputs': inputs, 'pages': len(rows),
              'agreed_pages': sum(d['status'] == 'agreed' for d in decisions),
              'changed_pages': sum(d['changed'] for d in decisions),
              'remaining_flags': sum(d['remaining_flags'] for d in decisions),
              'policy': 'All latest reviewer decisions must be verified and identical; at least two distinct normalized reviewer names.',
              'limitations': 'Names are self-reported; identity and independence are not authenticated. Flagged text still needs annotation-policy review.',
              'results': [{k: v for k, v in d.items() if k != 'text'} for d in decisions]}
    (output / 'adjudication.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                           encoding='utf-8', newline='\n')
    (output / 'review-evidence.json').write_text(json.dumps({
        'schema': 'polocrbench-review-evidence-v1', 'manifest_sha256': source_hash,
        'events': list(all_events.values())}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    artifacts = [target, output / 'adjudication.json', output / 'review-evidence.json']
    (output / 'checksums.sha256').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in artifacts),
                                           encoding='utf-8', newline='\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--reviews', nargs='*', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = adjudicate(args.manifest, args.reviews, args.output)
    print(json.dumps({key: report[key] for key in ('release_status', 'pages', 'agreed_pages', 'changed_pages', 'remaining_flags')}, indent=2))


if __name__ == '__main__':
    main()
