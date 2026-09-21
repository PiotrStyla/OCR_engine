from training.kaggle_kraken_diagnostics import select_pages, write_gallery, validate_pairing, serializable
from types import SimpleNamespace
import pytest


def test_selection_ignores_input_order_and_scores():
    rows = [{'id': 'B__2'}, {'id': 'A__9'}, {'id': 'A__1'}, {'id': 'B__1'}]
    assert [r['id'] for r in select_pages(rows)] == ['A__1', 'B__1']


def test_gallery_escapes_ocr(tmp_path):
    write_gallery(tmp_path, [{'id': '<page>', 'directory': 'page-00',
        'lines': [{'order': 0, 'crop': 'line-0000.png', 'text': '<script>alert(1)</script>'}]}])
    page = (tmp_path / 'index.html').read_text()
    assert '<script>' not in page
    assert '&lt;script&gt;' in page
    assert 'page-00/line-0000.png' in page


def test_pairing_rejects_reordered_records():
    lines = [SimpleNamespace(id='a'), SimpleNamespace(id='b')]
    validate_pairing(lines, lines)
    with pytest.raises(ValueError):
        validate_pairing(lines, list(reversed(lines)))
    with pytest.raises(ValueError):
        validate_pairing([lines[0], lines[0]], [lines[0], lines[0]])


def test_configuration_serialization():
    value = serializable(SimpleNamespace(padding=16, device=[0], decoder=select_pages))
    assert value['attributes']['padding'] == 16
    assert value['attributes']['decoder']['callable'].endswith('.select_pages')
