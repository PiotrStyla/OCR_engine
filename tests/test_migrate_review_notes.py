import json
import pytest
from tests.test_adjudicate_reviews import source, event, export
from training.migrate_review_notes import migrate


def test_explicit_confirmation_required(tmp_path):
    with pytest.raises(ValueError):
        migrate('unused', 'unused', tmp_path / 'out')


def test_verbatim_notes_preserve_evidence_without_approval(tmp_path):
    manifest, row = source(tmp_path)
    e = event(row, 'A', decision='proposed')
    e['note'] = '  Transcription a\u0301  '
    review = export(tmp_path, manifest, 'review.json', [e])
    original = manifest.read_bytes()
    result = migrate(manifest, review, tmp_path / 'out', notes_are_transcriptions=True)
    out = json.loads((tmp_path / 'out/manifest.jsonl').read_text(encoding='utf-8'))
    assert out['text'] == e['note']
    assert out['source_review_decision'] == 'proposed'
    assert out['review_status'] == 'draft-needs-confirmation'
    assert not out['eligible_for_evaluation']
    assert out['original_text'] == row['text']
    assert result['notes_promoted'] == 1
    assert manifest.read_bytes() == original
    assert (tmp_path / 'out/original-review.json').read_bytes() == review.read_bytes()


def test_empty_note_uses_actual_edit_and_keeps_pending(tmp_path):
    manifest, row = source(tmp_path)
    e = event(row, 'A', after='Edited', decision='needs-review')
    review = export(tmp_path, manifest, 'review.json', [e])
    result = migrate(manifest, review, tmp_path / 'out', notes_are_transcriptions=True)
    assert result['notes_promoted'] == 0
    out = json.loads((tmp_path / 'out/manifest.jsonl').read_text(encoding='utf-8'))
    assert out['text'] == 'Edited'
    assert out['source_review_decision'] == 'needs-review'
    with pytest.raises(FileExistsError):
        migrate(manifest, review, tmp_path / 'out', notes_are_transcriptions=True)
