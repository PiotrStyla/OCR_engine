import hashlib
import json

import pytest

from training.transcription_eval import evaluate


def write_case(tmp_path, references, predictions):
    image = tmp_path / 'page.png'
    image.write_bytes(b'fixture')
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps({
        'id': str(i), 'image': image.name, 'sha256': digest, 'text': text,
    }, ensure_ascii=False) for i, text in enumerate(references)), encoding='utf-8')
    supplied = tmp_path / 'predictions.jsonl'
    supplied.write_text('\n'.join(json.dumps(row, ensure_ascii=False)
                                 for row in predictions), encoding='utf-8')
    return manifest, supplied


@pytest.mark.parametrize('status', ['missing', 'error'])
def test_failures_stay_in_structure_denominator(tmp_path, status):
    predictions = [{'id': '0', 'status': 'ok', 'text': '# Title'}]
    if status == 'error':
        predictions.append({'id': '1', 'status': 'error', 'text': '# Title'})
    paths = write_case(tmp_path, ['# Title', '# Title'], predictions)
    result = evaluate(*paths)
    assert result['structure_similarity'] == 0.5
    assert result['cer_micro'] == 0.5
    assert result['errors_or_missing'] == 1
    assert result['results'][1]['structure']['structure_similarity'] == 0
    assert result['protocol_version'] == 'polocrbench-transcription-v1.1'


def test_failed_blank_page_is_not_successful_blank_page(tmp_path):
    paths = write_case(tmp_path, ['', ''], [{'id': '0', 'status': 'ok', 'text': ''}])
    result = evaluate(*paths)
    assert result['structure_similarity'] == 0.5


def test_all_missing_pages_score_zero(tmp_path):
    assert evaluate(*write_case(tmp_path, ['# Title'], []))['structure_similarity'] == 0


def test_jsonl_unicode_separators_and_normalization(tmp_path):
    paths = write_case(tmp_path, ['\u201eZa\u017co\u0301\u0142\u0107\u201d\u2028tekst\u2029dalej'], [
        {'id': '0', 'status': 'ok', 'text': '"Za\u017c\u00f3\u0142\u0107"\u2028tekst\u2029dalej'},
    ])
    result = evaluate(*paths)
    assert result['cer_micro'] == 0
    assert result['structure_similarity'] == 1


def test_structure_can_be_disabled(tmp_path):
    result = evaluate(*write_case(tmp_path, ['text'], []), with_structure=False)
    assert 'structure_similarity' not in result
    assert 'structure' not in result['results'][0]
