import hashlib
import json

import pytest

from training.kie_eval import SCHEMAS, evaluate, normalize_value


@pytest.mark.parametrize('value,expected', [
    ('2026-10-13', '2026-10-13'),
    ('13.10.2026', '2026-10-13'),
    ('13/10/2026', '2026-10-13'),
    ('3.1.2026', '2026-01-03'),
    ('2026.10.13', '2026-10-13'),
    ('13 października 2026 r.', '2026-10-13'),
    ('13 pazdziernika 2026', '2026-10-13'),
    ('1 stycznia 2026', '2026-01-01'),
    ('31.02.2026', '31.02.2026'),
    ('13/10/26', '13/10/26'),
])
def test_date_normalization(value, expected):
    assert normalize_value('date', value) == expected


@pytest.mark.parametrize('value,expected', [
    ('1 234,56 zł', '1234.56'),
    ('1.234,56 PLN', '1234.56'),
    ('1234.56', '1234.56'),
    ('1234,50', '1234.5'),
    ('1234.00', '1234'),
    ('1 234,-', '1234'),
    ('-5,00 zł', '-5'),
    ('-0,00 zł', '0'),
    ('12,50', '12.5'),
    ('12,500', '12500'),
    ('1234,567', '1234.567'),
    ('nie podano', 'nie podano'),
])
def test_money_normalization(value, expected):
    assert normalize_value('money', value) == expected


def test_identifier_and_text_normalization():
    assert normalize_value('nip', '123-456-78 90') == '1234567890'
    assert normalize_value('nip', '123') == '123'
    assert normalize_value('pesel', '020708 03654') == '02070803654'
    assert normalize_value('text', '\u201eAla\u201d  MA\t kota') == '"ala" ma kota'
    assert normalize_value('name', 'ZAKŁAD "GÓRNIK" Sp. z o.o.') == \
        'zakład "górnik" sp. z o.o.'
    assert normalize_value('address', 'ul. Marszałkowska 1A, 00-001 Warszawa') == \
        'ul marszałkowska 1a 00-001 warszawa'
    assert normalize_value('text', '') == ''


def fixture(tmp_path, rows):
    image = tmp_path / 'page.png'
    image.write_bytes(b'fixture')
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps({
        'id': str(index), 'image': image.name, 'sha256': digest,
        'doc_type': doc_type, 'fields': fields,
    }, ensure_ascii=False) for index, (doc_type, fields) in enumerate(rows)),
        encoding='utf-8')
    return manifest


def predictions(tmp_path, rows):
    path = tmp_path / 'predictions.jsonl'
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows),
                    encoding='utf-8')
    return path


INVOICE = ('faktura', {'invoice_number': 'FV/1', 'issue_date': '13.10.2026',
                       'gross_total': '1 234,56 zł', 'seller_nip': '123-456-78-90'})


def test_normalization_counts_as_exact_match(tmp_path):
    manifest = fixture(tmp_path, [INVOICE])
    result = evaluate(manifest, predictions(tmp_path, [{'id': '0', 'status': 'ok', 'fields': {
        'invoice_number': 'fv/1', 'issue_date': '2026-10-13',
        'gross_total': '1234.56', 'seller_nip': '1234567890'}}]))
    assert result['f1_micro'] == 1.0
    assert result['counts_micro'] == {'tp': 4, 'fp': 0, 'fn': 0}
    assert result['per_type']['date']['f1'] == 1.0
    assert result['per_field']['faktura.gross_total']['tp'] == 1
    assert result['results'][0]['fields']['issue_date'] == 'match'


def test_mismatch_counts_fp_and_fn(tmp_path):
    manifest = fixture(tmp_path, [INVOICE])
    result = evaluate(manifest, predictions(tmp_path, [{'id': '0', 'status': 'ok', 'fields': {
        'invoice_number': 'FV/2', 'issue_date': '13.10.2026',
        'gross_total': '1 234,56 zł', 'seller_nip': '1234567890'}}]))
    assert result['counts_micro'] == {'tp': 3, 'fp': 1, 'fn': 1}
    assert result['f1_micro'] == pytest.approx(6 / 8)
    assert result['results'][0]['fields']['invoice_number'] == 'mismatch'


def test_hallucinated_and_missing_fields(tmp_path):
    manifest = fixture(tmp_path, [INVOICE])
    result = evaluate(manifest, predictions(tmp_path, [{'id': '0', 'status': 'ok', 'fields': {
        'buyer_name': 'FAKTYCZNY NABYWCA', 'issue_date': '13.10.2026'}}]))
    counts = result['counts_micro']
    assert counts == {'tp': 1, 'fp': 1, 'fn': 3}
    assert result['results'][0]['fields']['buyer_name'] == 'spurious'
    assert result['results'][0]['fields']['invoice_number'] == 'missing'


def test_error_and_missing_documents_keep_reference_fields_as_fn(tmp_path):
    manifest = fixture(tmp_path, [INVOICE, INVOICE])
    result = evaluate(manifest, predictions(tmp_path, [
        {'id': '0', 'status': 'error', 'fields': {}},
    ]))
    assert result['errors_or_missing'] == 2
    assert result['counts_micro'] == {'tp': 0, 'fp': 0, 'fn': 8}
    assert result['f1_micro'] == 0.0


def test_nothing_to_extract_scores_one(tmp_path):
    manifest = fixture(tmp_path, [('formularz', {})])
    result = evaluate(manifest, predictions(tmp_path, [
        {'id': '0', 'status': 'ok', 'fields': {'form_code': 'ZUS Z-3'}}]))
    assert result['counts_micro'] == {'tp': 0, 'fp': 1, 'fn': 0}
    assert result['f1_micro'] == 0.0
    empty = evaluate(manifest, predictions(tmp_path, [{'id': '0', 'status': 'ok', 'fields': {}}]))
    assert empty['f1_micro'] == 1.0
    assert empty['results'][0]['fields']['form_code'] == 'absent'


def test_elapsed_seconds_passthrough(tmp_path):
    manifest = fixture(tmp_path, [INVOICE])
    result = evaluate(manifest, predictions(tmp_path, [
        {'id': '0', 'status': 'ok', 'fields': {}, 'elapsed_seconds': 2.0}]))
    assert result['results'][0]['elapsed_seconds'] == 2.0


@pytest.mark.parametrize('row,error', [
    ({'id': '0', 'status': 'ok'}, 'match schema'),
    ({'id': '0', 'status': 'ok', 'fields': []}, 'match schema'),
    ({'id': '0', 'status': 'ok', 'fields': {'unknown': 'x'}}, 'match schema'),
    ({'id': '0', 'status': 'error', 'fields': {'invoice_number': 'x'}}, 'empty fields'),
    ({'id': '0', 'status': 'ok', 'fields': {'gross_total': {'amount': 1}}}, 'strings or numbers'),
])
def test_invalid_predictions_raise(tmp_path, row, error):
    manifest = fixture(tmp_path, [INVOICE])
    with pytest.raises(ValueError, match=error):
        evaluate(manifest, predictions(tmp_path, [row]))


def test_invalid_manifest_raises(tmp_path):
    manifest = fixture(tmp_path, [('niedokument', {}), INVOICE])
    with pytest.raises(ValueError, match='document type'):
        evaluate(manifest, predictions(tmp_path, []))
    manifest = fixture(tmp_path, [('faktura', {'nieznane_pole': 'x'}), INVOICE])
    with pytest.raises(ValueError, match='outside schema'):
        evaluate(manifest, predictions(tmp_path, []))


def test_schemas_cover_all_document_types():
    assert set(SCHEMAS) == {'faktura', 'umowa', 'pismo_urzedowe', 'formularz'}
    assert set(SCHEMAS['faktura'].values()) <= {'text', 'name', 'date', 'money',
                                                'nip', 'pesel', 'address'}
