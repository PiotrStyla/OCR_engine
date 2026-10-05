import json

import pytest

from training.full_page_pilot import digest, write_json
from training.import_review_status_confirmation import apply
from tests.test_adjudicate_reviews import event, export, source


def inputs(tmp_path):
    manifest, row = source(tmp_path)
    row.update(split='validation', eligible_for_training=False, final_test=False)
    manifest.write_text(json.dumps(row), encoding='utf-8')
    review = export(tmp_path, manifest, 'review.json',
                    [event(row, 'A', decision='needs-review', after=row['text'])])
    confirmation = tmp_path/'confirmation.json'
    packet = {'schema': 'polocrbench-review-status-confirmation-v1', 'approved': True,
        'user_message': 'These pages were visually verified.',
        'source_manifest_sha256': digest(manifest), 'source_review_sha256': digest(review),
        'from_decision': 'needs-review', 'to_decision': 'verified', 'page_ids': ['page']}
    write_json(confirmation, packet)
    return manifest, review, confirmation, packet


def test_confirmation_preserves_export_and_text(tmp_path):
    manifest, review, confirmation, _ = inputs(tmp_path)
    old_review, old_manifest = review.read_bytes(), manifest.read_bytes()
    output = tmp_path/'result'
    report = apply(manifest, review, confirmation, output)
    assert report['decisions'] == {'verified': 1}
    assert report['results'][0]['original_decision'] == 'needs-review'
    assert report['changed_texts'] == report['gold_pages'] == report['training_examples_created'] == 0
    assert report['review_events_synthesized'] is False
    assert (output/'original-review.json').read_bytes() == review.read_bytes() == old_review
    assert (output/'source-manifest-snapshot.jsonl').read_bytes() == manifest.read_bytes() == old_manifest
    assert all(digest(output/name) == sha for name, sha in json.loads((output/'checksums.json').read_text()).items())


@pytest.mark.parametrize(('key', 'value'), [('approved', False), ('user_message', ''),
    ('source_review_sha256', '0'*64), ('source_manifest_sha256', '0'*64),
    ('page_ids', ['missing']), ('page_ids', ['page', 'page']), ('to_decision', 'gold')])
def test_unapproved_or_wrong_scope_rejected(tmp_path, key, value):
    manifest, review, confirmation, packet = inputs(tmp_path)
    packet[key] = value
    write_json(confirmation, packet)
    with pytest.raises(ValueError):
        apply(manifest, review, confirmation, tmp_path/'result')
    assert not (tmp_path/'result').exists()


def test_text_edits_are_not_status_confirmation(tmp_path):
    manifest, review, confirmation, packet = inputs(tmp_path)
    original = json.loads(review.read_text())
    original['events'][0]['after'] = 'new text'
    write_json(review, original)
    packet['source_review_sha256'] = digest(review)
    write_json(confirmation, packet)
    with pytest.raises(ValueError, match='status-only'):
        apply(manifest, review, confirmation, tmp_path/'result')
