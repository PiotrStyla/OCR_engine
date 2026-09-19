import json

import pytest
from PIL import Image

from training.adjudicate_reviews import adjudicate, text_hash
from training.stage_impact_benchmark import digest


def source(tmp_path):
    image = tmp_path / 'page.png'
    Image.new('L', (10, 10), 255).save(image)
    manifest = tmp_path / 'source.jsonl'
    row = {'id': 'page', 'image': image.name, 'sha256': digest(image), 'text': 'Old\ufffd'}
    manifest.write_text(json.dumps(row), encoding='utf-8')
    return manifest, row


def event(row, reviewer, *, after='Correct', decision='verified', identifier=None,
          before=None, timestamp='2026-09-19T10:00:00Z'):
    return {'id': identifier or reviewer, 'page_id': row['id'], 'reviewer': reviewer,
            'timestamp': timestamp, 'note': '', 'decision': decision,
            'before': row['text'] if before is None else before, 'after': after,
            'original_text_sha256': text_hash(row['text']), 'image_sha256': row['sha256']}


def export(tmp_path, manifest, name, events):
    path = tmp_path / name
    path.write_text(json.dumps({'schema': 'polocrbench-review-patch-v1',
                               'manifest_sha256': digest(manifest), 'events': events}), encoding='utf-8')
    return path


def test_two_agreeing_reviewers_create_traceable_candidate(tmp_path):
    manifest, row = source(tmp_path)
    original = manifest.read_bytes()
    paths = [export(tmp_path, manifest, name + '.json', [event(row, name)]) for name in ('A', 'B')]
    output = tmp_path / 'candidate'
    report = adjudicate(manifest, paths, output)
    assert report['agreed_pages'] == report['changed_pages'] == 1
    candidate = json.loads((output / 'manifest.jsonl').read_text(encoding='utf-8'))
    assert candidate['text'] == 'Correct'
    assert digest(output / candidate['image']) == row['sha256']
    assert manifest.read_bytes() == original
    for line in (output / 'checksums.sha256').read_text().splitlines():
        checksum, name = line.split('  ')
        assert digest(output / name) == checksum
    with pytest.raises(FileExistsError):
        adjudicate(manifest, paths, output)


@pytest.mark.parametrize('other,decision,expected', [
    ('Different', 'verified', 'conflict'), ('Correct', 'proposed', 'pending'),
    ('Correct', 'needs-review', 'pending'),
])
def test_conflicts_and_pending_decisions_never_change_source(tmp_path, other, decision, expected):
    manifest, row = source(tmp_path)
    a = export(tmp_path, manifest, 'a.json', [event(row, 'A')])
    b = export(tmp_path, manifest, 'b.json', [event(row, 'B', after=other, decision=decision)])
    output = tmp_path / 'candidate'
    report = adjudicate(manifest, [a, b], output)
    assert report['results'][0]['status'] == expected
    assert report['changed_pages'] == 0
    assert json.loads((output / 'manifest.jsonl').read_text(encoding='utf-8'))['text'] == row['text']


def test_duplicate_exports_and_reviewer_aliases_do_not_count_twice(tmp_path):
    manifest, row = source(tmp_path)
    a = export(tmp_path, manifest, 'a.json', [event(row, 'Anna')])
    b = export(tmp_path, manifest, 'b.json', [event(row, '  ANNA  ')])
    report = adjudicate(manifest, [a, a, b], tmp_path / 'candidate')
    assert report['results'][0]['status'] == 'single-review'
    assert report['changed_pages'] == 0


def test_later_withdrawal_supersedes_previous_approval(tmp_path):
    manifest, row = source(tmp_path)
    first = event(row, 'A')
    second = event(row, 'A', identifier='withdrawal', before='Correct', decision='needs-review',
                   timestamp='2026-09-19T11:00:00Z')
    a = export(tmp_path, manifest, 'a.json', [first, second])
    b = export(tmp_path, manifest, 'b.json', [event(row, 'B')])
    assert adjudicate(manifest, [a, b], tmp_path / 'candidate')['results'][0]['status'] == 'pending'


def test_equal_timestamp_conflicting_review_is_not_arbitrarily_chosen(tmp_path):
    manifest, row = source(tmp_path)
    a = export(tmp_path, manifest, 'a.json', [event(row, 'A')])
    b = export(tmp_path, manifest, 'b.json', [event(row, 'A', identifier='alternative', after='Other')])
    report = adjudicate(manifest, [a, b], tmp_path / 'candidate')
    assert report['results'][0]['status'] == 'ambiguous-reviewer-history'


@pytest.mark.parametrize('field,value', [
    ('before', 'forged'), ('image_sha256', 'wrong'), ('original_text_sha256', 'wrong'),
    ('timestamp', '2026-09-19T12:00:00'), ('reviewer', '  '), ('decision', 'approved'),
])
def test_invalid_evidence_rejected_before_writes(tmp_path, field, value):
    manifest, row = source(tmp_path)
    item = event(row, 'A'); item[field] = value
    review = export(tmp_path, manifest, 'invalid.json', [item])
    with pytest.raises(ValueError):
        adjudicate(manifest, [review], tmp_path / 'candidate')
    assert not (tmp_path / 'candidate').exists()


def test_empty_reviews_preserve_all_pages_as_unreviewed(tmp_path):
    manifest, row = source(tmp_path)
    report = adjudicate(manifest, [], tmp_path / 'candidate')
    assert report['agreed_pages'] == report['changed_pages'] == 0
    assert report['results'][0]['status'] == 'unreviewed'
    assert report['remaining_flags'] == 1


def test_backdated_withdrawal_is_rejected(tmp_path):
    manifest, row = source(tmp_path)
    review = export(tmp_path, manifest, 'a.json', [event(row, 'A'),
        event(row, 'A', identifier='withdraw', before='Correct', decision='needs-review',
              timestamp='2026-09-18T10:00:00Z')])
    with pytest.raises(ValueError, match='chronology'):
        adjudicate(manifest, [review], tmp_path / 'candidate')


def test_foreign_manifest_and_modified_event_id_rejected(tmp_path):
    manifest, row = source(tmp_path)
    a = export(tmp_path, manifest, 'a.json', [event(row, 'A')])
    b = export(tmp_path, manifest, 'b.json', [event(row, 'A', after='Other')])
    with pytest.raises(ValueError, match='same event ID'):
        adjudicate(manifest, [a, b], tmp_path / 'candidate')
    packet = json.loads(a.read_text())
    packet['manifest_sha256'] = 'wrong'
    a.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match='manifest mismatch'):
        adjudicate(manifest, [a], tmp_path / 'candidate')
