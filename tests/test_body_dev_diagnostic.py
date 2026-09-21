import io
import json
import zipfile

import pytest
from training.kaggle_body_dev_diagnostic import digest, load_input, metrics


def row(id='a', decision='proposed', text='abc'):
    return {'id': id, 'text': text, 'collection': 'dev', 'eligible_for_evaluation': False,
            'image': 'images/0000.png', 'sha256': digest(b'image'), 'source_review_decision': decision}


def package(rows):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        z.writestr('manifest.json', json.dumps(rows))
        z.writestr('provenance.json', json.dumps({'scope': 'diagnostic-only'}))
        z.writestr('images/0000.png', b'image')
    return stream.getvalue()


def test_separate_uncertain_subset_and_error_denominator():
    rows = [row(), row('b', 'needs-review')]
    preds = [{'id': 'a', 'text': 'abc', 'status': 'ok'}, {'id': 'b', 'text': '', 'status': 'error'}]
    result = metrics(rows, preds)
    assert result['all_draft_lines']['lines'] == 2
    assert result['all_draft_lines']['cer'] == .5
    assert result['all_draft_lines']['errors'] == 1
    assert result['without_needs_review']['cer'] == 0


def test_lowercase_is_separate():
    result = metrics([row()], [{'id': 'a', 'text': 'ABC', 'status': 'ok'}])['all_draft_lines']
    assert result['cer'] == 1
    assert result['lowercase_cer_diagnostic'] == 0


def test_historical_acute_is_not_stripped():
    reference = 'W \u015bwietne b\u0142\u00e1waty.'
    prediction = 'W \u015bwietne b\u0142awaty.'
    result = metrics([row(text=reference)], [{'id': 'a', 'text': prediction, 'status': 'ok'}])['all_draft_lines']
    assert result['cer'] == pytest.approx(1 / 18)
    assert result['lowercase_cer_diagnostic'] == pytest.approx(1 / 18)
    composed = reference.replace('\u00e1', 'a\u0301')
    assert metrics([row(text=reference)], [{'id': 'a', 'text': composed, 'status': 'ok'}])['all_draft_lines']['cer'] == 0


def test_long_s_is_not_modernized():
    result = metrics([row(text='\u017f')], [{'id': 'a', 'text': 's', 'status': 'ok'}])['all_draft_lines']
    assert result['cer'] == 1
    assert result['lowercase_cer_diagnostic'] == 1


def test_empty_subset_and_alignment():
    assert metrics([row(decision='needs-review')], [{'id': 'a', 'text': '', 'status': 'error'}])['without_needs_review']['cer'] is None
    with pytest.raises(ValueError):
        metrics([row()], [{'id': 'b', 'text': '', 'status': 'error'}])


def test_zip_integrity():
    data = package([row()])
    assert len(load_input(data, digest(data))[0]) == 1
    with pytest.raises(ValueError):
        load_input(data, 'wrong')


@pytest.mark.parametrize('change', [{'collection': 'NA2_FT'}, {'eligible_for_evaluation': True},
                                   {'sha256': 'wrong'}, {'source_review_decision': None}, {'text': ''}])
def test_reject_invalid_input(change):
    data = package([{**row(), **change}])
    with pytest.raises(ValueError):
        load_input(data, digest(data))


def test_duplicate_lines():
    data = package([row(), row()])
    with pytest.raises(ValueError):
        load_input(data, digest(data))


def test_private_notebook_roundtrip(tmp_path):
    import ast
    import base64
    from PIL import Image
    from training.build_private_body_notebook import build
    image = tmp_path / 'line.png'
    Image.new('RGB', (12, 8), 'white').save(image)
    record = {**row(), 'image': 'line.png', 'sha256': digest(image.read_bytes()),
              'original_text': 'abc', 'page_id': 'page', 'review_status': 'draft-needs-confirmation'}
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps(record), encoding='utf-8')
    summary = build(manifest, tmp_path / 'private')
    notebook = json.loads((tmp_path / 'private/kaggle_body_dev_PRIVATE.ipynb').read_text(encoding='utf-8'))
    script = ''.join(notebook['cells'][-1]['source'])
    tree = ast.parse(script)
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id in {'INPUT_BASE64', 'INPUT_SHA256'}}
    rows, images, _ = load_input(base64.b64decode(assignments['INPUT_BASE64']), assignments['INPUT_SHA256'])
    assert rows[0]['text'] == 'abc'
    assert images[0] == image.read_bytes()
    assert summary['lines'] == 1
    assert notebook['cells'][-1]['outputs'] == []
