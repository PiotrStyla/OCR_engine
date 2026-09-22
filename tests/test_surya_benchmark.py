import hashlib
import json

import pytest

from training.run_surya_benchmark import (
    align_tables,
    blocks_to_text,
    load_page_cases,
    run_pages,
)
from training.validate_submission import validate


class Block(dict):
    def __getattr__(self, name):
        return self[name]


def test_blocks_to_text_uses_table_row_convention():
    blocks = [{'label': 'SectionHeader', 'html': '<h1>Faktura VAT</h1>'},
              {'label': 'Table', 'html': '<table><tr><th>Lp.</th><th>Nazwa</th></tr>'
                                         '<tr><td>1</td><td>Transport</td></tr></table>'},
              {'label': 'Text', 'html': '<p>Suma netto: 100,00 zł</p>'},
              {'label': 'Picture', 'html': '', 'skipped': True},
              {'label': 'Text', 'html': 'tekst', 'error': True}]
    text = blocks_to_text(blocks)
    assert text.splitlines() == ['Faktura VAT', 'Lp. Nazwa', '1 Transport', 'Suma netto: 100,00 zł']


def test_align_tables_orders_by_reading_order_and_counts_surplus():
    tables = [{'html': '<table><tr><td>druga</td></tr></table>', 'image_bbox': [0, 200, 10, 210]},
              {'html': '<table><tr><td>pierwsza</td></tr></table>', 'image_bbox': [0, 10, 10, 20]}]
    htmls, surplus = align_tables(tables, 3)
    assert htmls[0] == '<table><tr><td>pierwsza</td></tr></table>'
    assert htmls[1] == '<table><tr><td>druga</td></tr></table>'
    assert htmls[2] == '' and surplus == 0
    assert align_tables(tables, 1) == (['<table><tr><td>pierwsza</td></tr></table>'], 1)


def fixture(tmp_path, rows):
    image = tmp_path / 'page.png'
    image.write_bytes(b'fixture')
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps({'id': id_, 'image': image.name,
                                              'sha256': sha, **extra})
                                  for id_, extra in rows), encoding='utf-8')
    return manifest, image


def test_b_run_fills_table_slots_per_page_with_per_page_cost(tmp_path):
    manifest, image = fixture(tmp_path, [
        ('t1', {'html': '<table><tr><td>a</td></tr></table>', 'table_index': 0}),
        ('t2', {'html': '<table><tr><td>b</td></tr></table>', 'table_index': 1}),
    ])
    pages = load_page_cases(manifest)
    assert len(pages) == 1 and len(pages[0]['rows']) == 2

    def predict(path):
        return [{'html': '<table><tr><td>b</td></tr></table>', 'image_bbox': [0, 50, 1, 51]},
                {'html': '<table><tr><td>a</td></tr></table>', 'image_bbox': [0, 5, 1, 6]}], 2.0

    run = run_pages('B', pages, predict, tmp_path / 'run', {'engine': 'fake'})
    assert run['state'] == 'completed' and run['surplus_tables'] == 0
    assert run['cost']['pages_timed'] == 1 and run['cost']['mean_elapsed_seconds'] == 2.0
    rows = [json.loads(line) for line in
            (tmp_path / 'run' / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['html'] for row in rows] == ['<table><tr><td>a</td></tr></table>',
                                             '<table><tr><td>b</td></tr></table>']
    assert rows[0]['elapsed_seconds'] == 2.0 and 'elapsed_seconds' not in rows[1]
    assert validate(manifest, tmp_path / 'run' / 'predictions.jsonl', subtask='B')['complete']
    with pytest.raises(FileExistsError):
        run_pages('B', pages, predict, tmp_path / 'run', {})


def test_failed_page_becomes_error_rows_for_all_slots(tmp_path):
    manifest, image = fixture(tmp_path, [
        ('t1', {'html': '<table><tr><td>a</td></tr></table>', 'table_index': 0}),
        ('t2', {'html': '<table><tr><td>b</td></tr></table>', 'table_index': 1}),
    ])

    def predict(path):
        raise RuntimeError('backend padl')

    run = run_pages('B', load_page_cases(manifest), predict, tmp_path / 'run', {'engine': 'fake'})
    assert run['error_pages'] == 1
    rows = [json.loads(line) for line in
            (tmp_path / 'run' / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows == [{'id': 't1', 'status': 'error', 'html': '', 'error_type': 'RuntimeError'},
                    {'id': 't2', 'status': 'error', 'html': '', 'error_type': 'RuntimeError'}]
    assert 'cost' not in run


def test_a_run_writes_one_text_row_per_page(tmp_path):
    manifest, image = fixture(tmp_path, [('p1', {'text': 'Ala'})])
    run = run_pages('A', load_page_cases(manifest),
                    lambda path: ('Ala\nma kota', 0.5), tmp_path / 'run', {'engine': 'fake'})
    assert run['pages_completed'] == 1
    rows = [json.loads(line) for line in
            (tmp_path / 'run' / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows == [{'id': 'p1', 'status': 'ok', 'text': 'Ala\nma kota',
                     'elapsed_seconds': 0.5}]
