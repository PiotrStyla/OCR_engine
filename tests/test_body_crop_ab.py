import ast
import base64
import io
import json
from pathlib import Path

from PIL import Image
from training.kaggle_body_dev_diagnostic import digest, load_input


def test_embedded_ab_is_small_valid_and_same_reference():
    notebook = Path(__file__).resolve().parents[1] / 'training/kaggle_body_crop_ab.ipynb'
    assert notebook.stat().st_size < 1_000_000
    source = ''.join(json.loads(notebook.read_text(encoding='utf-8'))['cells'][-1]['source'])
    tree = ast.parse(source)
    encoded = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'INPUT_BASE64')
    data = base64.b64decode(encoded)
    rows, images, provenance = load_input(data, digest(data))
    assert len(rows) == 2
    assert rows[0]['text'] == rows[1]['text']
    assert rows[0]['original_line_id'] == rows[1]['original_line_id']
    assert images[0] != images[1]
    assert provenance['reference_changed'] is False
    with Image.open(io.BytesIO(images[1])) as image:
        assert image.size == (985, 145)
    # Execute only the small metric override, never imports/inference/downloads.
    override = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'metrics'
                    and any(isinstance(x, ast.Return) and isinstance(x.value, ast.DictComp) for x in n.body))
    from training.kaggle_body_dev_diagnostic import metrics
    env = {'_base_metrics': metrics}
    exec(compile(ast.Module(body=[override], type_ignores=[]), '<metric-override>', 'exec'), env)
    scores = env['metrics'](rows, [{'id': r['id'], 'text': r['text'], 'status': 'ok'} for r in rows])
    assert set(scores) == {'A-original', 'B-manual-geometry'}
    assert all(s['cer'] == 0 for s in scores.values())
