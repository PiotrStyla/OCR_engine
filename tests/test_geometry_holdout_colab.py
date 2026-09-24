import json
from pathlib import Path

from training.build_geometry_holdout_colab import build, MANIFEST_COMMIT, MANIFEST_SHA256


def test_notebook_is_reproducible_small_and_compilable(tmp_path):
    target = tmp_path / 'colab_geometry_holdout.ipynb'
    build(target)
    saved = Path(__file__).resolve().parents[1] / 'training' / target.name
    assert target.read_bytes() == saved.read_bytes()
    assert target.stat().st_size < 1_000_000
    notebook = json.loads(target.read_text(encoding='utf-8'))
    assert notebook['metadata']['accelerator'] == 'GPU'
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == []
            source = ''.join(cell['source'])
            if not source.startswith('%pip'):
                compile(source, '<cell>', 'exec')
    run = ''.join(next(c for c in notebook['cells'] if c['id'] == 'run')['source'])
    assert MANIFEST_COMMIT in run and MANIFEST_SHA256 in run
    assert run.index('check_colab_environment') < run.index('MANIFEST_URL')
    assert 'INCOMPLETE RUN' in run and '12/12 regions retained' in run


def test_install_cell_pins_opencv_and_transformers(tmp_path):
    target = tmp_path / 'notebook.ipynb'
    build(target)
    notebook = json.loads(target.read_text(encoding='utf-8'))
    install = ''.join(next(c for c in notebook['cells'] if c['id'] == 'install')['source'])
    assert 'transformers==4.57.6' in install
    assert 'opencv-python-headless==4.12.0.88' in install
