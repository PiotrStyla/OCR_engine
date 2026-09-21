import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('printed_dev', Path(__file__).resolve().parents[1] / 'training/kaggle_printed_dev_control.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(**overrides):
    return dict({'id': 'dev1', 'split': 'validation', 'collection': 'Choragiew_FT',
                 'image_sha256': 'a', 'text': 'A sufficiently long historical line'}, **overrides)


def test_selection_preserves_reference():
    selected, excluded = module.select([row(text='Historical text with \ufffd retained'), row(id='dev2', text='two\nlines')], [])
    assert '\ufffd' in selected[0]['text']
    assert excluded[0]['id'] == 'dev2'


@pytest.mark.parametrize('overrides', [{'split': 'test'}, {'collection': 'NA2_FT'}])
def test_frozen_guard(overrides):
    with pytest.raises(ValueError):
        module.select([row(**overrides)], [])


def test_hash_overlap():
    with pytest.raises(ValueError):
        module.select([row()], [{'image_sha256': 'a'}])


def test_duplicate_ids():
    with pytest.raises(ValueError):
        module.select([row(), row()], [])


def test_failed_prediction_stays_in_denominator():
    result = module.score([row()], [{'id': 'dev1', 'text': '', 'status': 'error'}])
    assert result['cer'] == 1
    assert result['errors'] == 1
    assert result['regions'] == 1


def test_id_alignment():
    with pytest.raises(ValueError):
        module.score([row()], [{'id': 'other', 'text': '', 'status': 'ok'}])


@pytest.mark.parametrize('path', ['../x', '/x', 'C:/x', 'images\\x'])
def test_unsafe_path(path):
    with pytest.raises(ValueError):
        module.safe_file(path)


def test_notebook_embeds_current_script():
    import json
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'training/kaggle_printed_dev_control.ipynb').read_text(encoding='utf-8-sig'))
    script = (root / 'training/kaggle_printed_dev_control.py').read_text(encoding='utf-8')
    assert ''.join(notebook['cells'][-1]['source']) == script
    compile(script, '<notebook>', 'exec')
