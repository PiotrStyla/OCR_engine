import hashlib
import json
import random
import re

import pytest

from training.document_templates import DOC_TYPES, build_document, document_text
from training.generate_documents import DEGRADATIONS, generate, render_page
from training.kie_eval import SCHEMAS, evaluate as kie_evaluate, normalize_value
from training.table_eval import evaluate as table_evaluate, parse_table_html, teds
from training.transcription_eval import evaluate as transcription_evaluate
from training.validate_submission import validate


def test_ground_truth_consistency_and_normalization():
    for doc_type in DOC_TYPES:
        for index in range(6):
            doc = build_document(random.Random(f'{doc_type}:{index}'), doc_type)
            assert set(doc['fields']) <= set(SCHEMAS[doc_type])
            text = document_text(doc).casefold()
            for name, value in doc['fields'].items():
                field_type = SCHEMAS[doc_type][name]
                assert value.casefold() in text  # GT extractable from transcription
                normalized = normalize_value(field_type, value)
                if field_type == 'date':
                    assert re.match(r'^\d{4}-\d{2}-\d{2}$', normalized), (value, normalized)
                elif field_type == 'money':
                    assert re.match(r'^-?\d+(\.\d+)?$', normalized), (value, normalized)
                elif field_type == 'nip':
                    assert re.match(r'^\d{10}$', normalized), (value, normalized)
                elif field_type == 'pesel':
                    assert re.match(r'^\d{11}$', normalized), (value, normalized)


def test_tables_parse_and_self_score():
    for doc_type in DOC_TYPES:
        doc = build_document(random.Random(f'tables:{doc_type}'), doc_type)
        for table in doc['tables']:
            assert teds(table['html'], table['html']) == 1.0
            root = parse_table_html(table['html'])
            leaves = [node for node in _walk(root) if node.tag in ('td', 'th')]
            assert len(leaves) == sum(len(row) for row in table['rows'])
    invoice = build_document(random.Random('spans'), 'faktura')
    assert 'colspan="5"' in invoice['tables'][0]['html']


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def test_render_and_degradations_preserve_page():
    doc = build_document(random.Random('render'), 'umowa')
    image = render_page(doc, random.Random('r'), [])
    assert image.size == (1240, 1754)
    for name, degrade in DEGRADATIONS.items():
        degraded, recipe = degrade(image, random.Random(f'd:{name}'))
        assert degraded.size == image.size
        assert recipe['kind'] == name
        if name != 'clean':
            assert degraded.tobytes() != image.tobytes()


def test_generation_is_deterministic_and_refuses_overwrite(tmp_path):
    kwargs = dict(count=4, seed=11, types=DOC_TYPES, degradations=('clean', 'scan'))
    generate(tmp_path / 'a', **kwargs)
    generate(tmp_path / 'b', **kwargs)
    for name in ('manifest-A.jsonl', 'manifest-B.jsonl', 'manifest-C.jsonl', 'generation.json'):
        assert (tmp_path / 'a' / name).read_bytes() == (tmp_path / 'b' / name).read_bytes()
    generate(tmp_path / 'c', count=4, seed=12, types=DOC_TYPES, degradations=('clean', 'scan'))
    hashes_a = [row['sha256'] for row in _rows(tmp_path / 'a' / 'manifest-A.jsonl')]
    hashes_c = [row['sha256'] for row in _rows(tmp_path / 'c' / 'manifest-A.jsonl')]
    assert set(hashes_a) != set(hashes_c)
    with pytest.raises(FileExistsError):
        generate(tmp_path / 'a', count=1, seed=11)


def test_generation_report_records_recipes_and_hashes(tmp_path):
    generate(tmp_path / 'data', count=3, seed=9, types=('faktura',),
             degradations=('photo', 'compress', 'clean'))
    report = json.loads((tmp_path / 'data' / 'generation.json').read_text(encoding='utf-8'))
    assert {sample['degradation'] for sample in report['samples']} == \
        {'photo', 'compress', 'clean'}
    assert report['samples'][0]['recipe']['kind'] == 'photo'
    for name, sha in report['manifest_sha256'].items():
        assert hashlib.sha256((tmp_path / 'data' / name).read_bytes()).hexdigest() == sha


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def _predictions(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows),
                    encoding='utf-8', newline='\n')
    return path


def test_generated_set_scores_perfectly_through_evaluators(tmp_path):
    pytest.importorskip('jiwer')
    out = tmp_path / 'data'
    generate(out, count=4, seed=5, degradations=('clean',))
    rows_a, rows_b, rows_c = (_rows(out / f'manifest-{subtask}.jsonl') for subtask in 'ABC')
    pred_a = _predictions(tmp_path, 'pred-a.jsonl',
                          [{'id': row['id'], 'status': 'ok', 'text': row['text']} for row in rows_a])
    pred_b = _predictions(tmp_path, 'pred-b.jsonl',
                          [{'id': row['id'], 'status': 'ok', 'html': row['html']} for row in rows_b])
    pred_c = _predictions(tmp_path, 'pred-c.jsonl',
                          [{'id': row['id'], 'status': 'ok', 'fields': row['fields']} for row in rows_c])
    assert validate(out / 'manifest-A.jsonl', pred_a, subtask='A')['eligible_for_scoring']
    assert validate(out / 'manifest-B.jsonl', pred_b, subtask='B')['eligible_for_scoring']
    assert validate(out / 'manifest-C.jsonl', pred_c, subtask='C')['eligible_for_scoring']
    assert transcription_evaluate(out / 'manifest-A.jsonl', pred_a)['cer_micro'] == 0.0
    assert table_evaluate(out / 'manifest-B.jsonl', pred_b)['teds_mean'] == 1.0
    assert kie_evaluate(out / 'manifest-C.jsonl', pred_c)['f1_micro'] == 1.0
