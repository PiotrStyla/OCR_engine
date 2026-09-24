import json
import zipfile
import io
import pytest
from training.geometry_comparison import pair_inputs, paired_metrics
from training.kaggle_body_dev_diagnostic import metrics


def fixture():
    row = {'id': 'line', 'text': 'abc', 'original_text': 'abc', 'source_review_decision': 'proposed',
           'collection': 'dev', 'page_id': 'page', 'geometry_status': 'auto-proposal'}
    return row


def test_pair_preserves_both_arms_and_metrics():
    row = fixture()
    loader = lambda data, h: ([row], [data], {'scope': 'diagnostic-only'})
    data = pair_inputs(b'old', 'a', b'new', 'b', loader)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        rows = json.loads(z.read('manifest.json'))
    assert len(rows) == 2
    assert rows[0]['comparison_id'] == rows[1]['comparison_id']
    preds = [{'id': rows[0]['id'], 'text': '', 'status': 'error'},
             {'id': rows[1]['id'], 'text': 'abc', 'status': 'ok'}]
    scores = paired_metrics(rows, preds, metrics)
    assert scores['original']['all_draft_lines']['cer'] == 1
    assert scores['automatic_with_fallback']['all_draft_lines']['cer'] == 0


def test_changed_reference_rejected():
    def loader(data, h):
        return [{**fixture(), 'text': data.decode()}], [b'image'], {}
    with pytest.raises(ValueError):
        pair_inputs(b'abc', 'a', b'changed', 'b', loader)


def test_fallback_must_retain_pixels():
    def loader(data, h):
        return [{**fixture(), 'geometry_status': 'fallback-original'}], [data], {}
    with pytest.raises(ValueError):
        pair_inputs(b'old', 'a', b'changed', 'b', loader)


def test_notebook_is_reproducible_and_small(tmp_path):
    from pathlib import Path
    from training.build_geometry_colab import build
    target = tmp_path / 'colab_auto_geometry.ipynb'
    build(target)
    root = Path(__file__).resolve().parents[1]
    assert target.read_bytes() == (root / 'training/colab_auto_geometry.ipynb').read_bytes()
    assert target.stat().st_size < 1_000_000
    nb = json.loads(target.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == []
            source = ''.join(cell['source'])
            if not source.startswith('%pip'):
                compile(source, '<cell>', 'exec')
