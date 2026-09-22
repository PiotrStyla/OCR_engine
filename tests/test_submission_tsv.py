import json

import pytest

from training.submission_tsv import (
    decode_cell,
    encode_cell,
    pack,
    unpack,
    validate_tsv,
)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8', newline='\n')
    return path


def test_cell_escaping_round_trips_exotic_text():
    text = 'linia\tnext\nkolejna\r\nścieżka C:\\dane\\ź "cytat"'
    assert decode_cell(encode_cell(text)) == text
    assert '\n' not in encode_cell(text) and '\t' not in encode_cell(text)


def test_decode_rejects_dangling_and_unknown_escapes():
    with pytest.raises(ValueError, match='Dangling'):
        decode_cell('tekst\\')
    with pytest.raises(ValueError, match='Unknown'):
        decode_cell('tekst\\q')


def test_pack_requires_exact_ids(tmp_path):
    in_tsv = write(tmp_path, 'in.tsv', 'a\nb\n')
    predictions = write(tmp_path, 'p.jsonl', '\n'.join(json.dumps(row) for row in [
        {'id': 'a', 'status': 'ok', 'text': 'Ala'}, {'id': 'x', 'status': 'ok', 'text': ''}]))
    with pytest.raises(ValueError, match='unknown'):
        pack(in_tsv, predictions, tmp_path / 'out.tsv', 'A')
    predictions = write(tmp_path, 'p2.jsonl', json.dumps({'id': 'a', 'status': 'ok', 'text': 'Ala'}))
    with pytest.raises(ValueError, match='Missing 1'):
        pack(in_tsv, predictions, tmp_path / 'out.tsv', 'A')


def test_pack_unpack_round_trip_preserves_payloads_in_in_order(tmp_path):
    in_tsv = write(tmp_path, 'in.tsv', 'b\na\n')
    rows = [{'id': 'a', 'status': 'ok', 'text': 'Ala\nma\tkota'},
            {'id': 'b', 'status': 'error', 'text': ''}]
    predictions = write(tmp_path, 'p.jsonl', '\n'.join(json.dumps(row) for row in rows))
    report = pack(in_tsv, predictions, tmp_path / 'out.tsv', 'A')
    assert report['rows'] == 2
    cells = (tmp_path / 'out.tsv').read_text(encoding='utf-8').splitlines()
    assert cells == ['', 'Ala\\nma\\tkota']  # in.tsv order: b (error -> empty), a
    out = tmp_path / 'roundtrip.jsonl'
    unpack(in_tsv, tmp_path / 'out.tsv', out, 'A')
    restored = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
    assert restored == [{'id': 'b', 'status': 'ok', 'text': ''},
                        {'id': 'a', 'status': 'ok', 'text': 'Ala\nma\tkota'}]


def test_kie_payload_round_trips_as_json_object(tmp_path):
    in_tsv = write(tmp_path, 'in.tsv', 'doc\n')
    predictions = write(tmp_path, 'p.jsonl', json.dumps(
        {'id': 'doc', 'status': 'ok', 'fields': {'issue_date': '13.10.2026', 'gross_total': '1 234,56 zł'}},
        ensure_ascii=False))
    pack(in_tsv, predictions, tmp_path / 'out.tsv', 'C')
    out = tmp_path / 'roundtrip.jsonl'
    unpack(in_tsv, tmp_path / 'out.tsv', out, 'C')
    restored = json.loads(out.read_text(encoding='utf-8'))
    assert restored == {'id': 'doc', 'status': 'ok',
                        'fields': {'issue_date': '13.10.2026', 'gross_total': '1 234,56 zł'}}


def test_validate_tsv_checks_row_alignment(tmp_path):
    in_tsv = write(tmp_path, 'in.tsv', 'a\nb\n')
    out_tsv = write(tmp_path, 'out.tsv', 'x\ny\n')
    assert validate_tsv(in_tsv, out_tsv)['rows'] == 2
    short = write(tmp_path, 'short.tsv', 'x\n')
    with pytest.raises(ValueError, match='must match'):
        validate_tsv(in_tsv, short)


def test_unpack_rejects_non_object_kie_payload(tmp_path):
    in_tsv = write(tmp_path, 'in.tsv', 'doc\n')
    out_tsv = write(tmp_path, 'out.tsv', '[]\n')
    with pytest.raises(ValueError, match='JSON object'):
        unpack(in_tsv, out_tsv, tmp_path / 'p.jsonl', 'C')
