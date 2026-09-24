import ast
import json
from pathlib import Path

import pytest
from training.build_colab_notebooks import convert

ROOT = Path(__file__).resolve().parents[1] / 'training'


@pytest.mark.parametrize('name', ['body_dev_diagnostic', 'body_crop_ab'])
def test_colab_conversion_matches_committed_notebook(tmp_path, name):
    target = tmp_path / f'colab_{name}.ipynb'
    convert(ROOT / f'kaggle_{name}.ipynb', target, ROOT / 'kaggle_body_dev_diagnostic.py')
    nb = json.loads(target.read_text(encoding='utf-8'))
    assert nb == json.loads((ROOT / target.name).read_text(encoding='utf-8'))
    code = ''.join(nb['cells'][-2]['source'])
    tree = ast.parse(code)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    assert isinstance(run.body[-1], ast.Return)
    assert run.body[-1].value.id == 'archive'
    assert "Path('/content')" in code and '/kaggle/working' not in code
    assert 'result_archive = run(' in code
    assert 'files.download(str(result_archive))' in ''.join(nb['cells'][-1]['source'])
    assert 'max_new_tokens=256' in code
    assert "'85d0c91c26f8e088849096dded7c9ba10b4cd9c9'" in code
    assert target.stat().st_size < 1_000_000
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            source = ''.join(cell['source'])
            if not source.startswith('%pip'):
                compile(source, '<cell>', 'exec')
