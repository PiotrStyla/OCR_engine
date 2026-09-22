"""PolOCRBench subtask C metric: key information extraction (field-level F1).

Deterministic, no LLM judges, stdlib only. Protocol ``polocrbench-kie-v1``.

Every document has a fixed type (``faktura``, ``umowa``, ``pismo_urzedowe``,
``formularz``) with the field schema in :data:`SCHEMAS` (single source of
truth; ``--dump-schemas`` exports it as JSON for participants).

Manifest JSONL: {id, image, sha256, doc_type, fields: {name: value}}.
Predictions JSONL: {id, status: ok|error, fields: {name: value}, elapsed_seconds?}.
Absent keys and empty values mean "not extracted"; error rows carry no fields.

A field is correct only on exact match after type-aware normalization:

- ``date`` -> ``YYYY-MM-DD`` (``13.10.2026``, ``2026-10-13``,
  ``13 października 2026 r.``; day-first for numeric dates);
- ``money`` -> plain decimal with ``.`` (``1 234,56 zł`` -> ``1234.56``,
  ``1 234,-`` -> ``1234``, trailing zeros stripped; currency is compared via
  its own field, not here);
- ``nip`` -> 10 digits, ``pesel`` -> 11 digits (separators stripped);
- ``address`` -> NFC + quotes + casefold + punctuation dropped (hyphens kept);
- ``text``/``name`` -> NFC + quotes + casefold + whitespace collapse.

A value that fails its typed parse falls back to text normalization, so it is
compared literally. Scoring is field-level micro: a mismatch or hallucinated
value counts FP+FN, a missed value counts FN. Missing/error documents keep
every non-empty reference field as FN. Main score: ``f1_micro`` = 2TP /
(2TP+FP+FN), 1.0 when there is nothing to extract at all. Auxiliary reports:
per field type (``per_type``) and per field name (``per_field``).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path

from training.transcription_eval import normalize
from training.validate_submission import load_jsonl

PROTOCOL_VERSION = 'polocrbench-kie-v1'

SCHEMAS = {
    'faktura': {
        'invoice_number': 'text',
        'issue_date': 'date',
        'due_date': 'date',
        'seller_name': 'name',
        'seller_nip': 'nip',
        'seller_address': 'address',
        'buyer_name': 'name',
        'buyer_nip': 'nip',
        'buyer_address': 'address',
        'net_total': 'money',
        'vat_amount': 'money',
        'gross_total': 'money',
        'currency': 'text',
    },
    'umowa': {
        'contract_number': 'text',
        'contract_date': 'date',
        'party_1': 'name',
        'party_1_address': 'address',
        'party_2': 'name',
        'party_2_address': 'address',
        'subject': 'text',
        'value': 'money',
        'currency': 'text',
    },
    'pismo_urzedowe': {
        'reference_number': 'text',
        'issue_date': 'date',
        'issuer': 'name',
        'addressee': 'name',
        'addressee_address': 'address',
        'subject': 'text',
    },
    'formularz': {
        'form_code': 'text',
        'form_name': 'name',
        'submission_date': 'date',
        'applicant_name': 'name',
        'applicant_pesel': 'pesel',
        'applicant_address': 'address',
    },
}

_MONTHS = {
    'styczen': 1, 'stycznia': 1, 'luty': 2, 'lutego': 2, 'marzec': 3, 'marca': 3,
    'kwiecien': 4, 'kwietnia': 4, 'maj': 5, 'maja': 5, 'czerwiec': 6, 'czerwca': 6,
    'lipiec': 7, 'lipca': 7, 'sierpien': 8, 'sierpnia': 8, 'wrzesien': 9, 'wrzesnia': 9,
    'pazdziernik': 10, 'pazdziernika': 10, 'listopad': 11, 'listopada': 11,
    'grudzien': 12, 'grudnia': 12,
}

_DATE_ISO = re.compile(r'^(\d{4})[-./](\d{1,2})[-./](\d{1,2})$')
_DATE_DMY = re.compile(r'^(\d{1,2})[-./](\d{1,2})[-./](\d{4})$')
_DATE_NAMED = re.compile(r'^(\d{1,2})\s+([a-z]+)\s+(\d{4})$')
_YEAR_SUFFIX = re.compile(r'\s*r\.?$')
_CURRENCY = re.compile(r'\s*(?:zł\.?|pln)\s*', re.IGNORECASE)
_GROUPING = re.compile(r'^\d{1,3}(?:,\d{3})+$|^\d{1,3}(?:\.\d{3})+$')
_SIGNED_DECIMAL = re.compile(r'^[+-]?\d+(?:[.,]\d+)?$')
_ADDRESS_DROP = re.compile(r'[^\w\s-]', re.UNICODE)


def _fold_ascii(text):
    decomposed = unicodedata.normalize('NFD', text)
    return ''.join(char for char in decomposed
                   if not unicodedata.combining(char))


def _iso(year, month, day):
    try:
        return datetime.date(year, month, day).isoformat()
    except ValueError:
        return None


def _parse_date(value):
    text = _YEAR_SUFFIX.sub('', normalize(value).casefold()).rstrip('.').strip()
    folded = _fold_ascii(text)
    match = _DATE_ISO.match(folded)
    if match:
        return _iso(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _DATE_DMY.match(folded)
    if match:
        return _iso(int(match.group(3)), int(match.group(2)), int(match.group(1)))
    match = _DATE_NAMED.match(folded)
    if match and match.group(2) in _MONTHS:
        return _iso(int(match.group(3)), _MONTHS[match.group(2)], int(match.group(1)))
    return None


def _parse_money(value):
    text = normalize(value).casefold()
    text = re.sub(r',\s*-\s*$', '.00', text)
    text = _CURRENCY.sub('', text)
    text = text.replace(' ', '').replace('\u2019', '').replace("'", '')
    if not text:
        return None
    sign = ''
    if text[0] in '+-':
        sign, text = ('-' if text[0] == '-' else ''), text[1:]
    if not text:
        return None
    if ',' in text and '.' in text:
        decimal_mark = ',' if text.rfind(',') > text.rfind('.') else '.'
        other = '.' if decimal_mark == ',' else ','
        text = text.replace(other, '').replace(decimal_mark, '.')
    elif ',' in text or '.' in text:
        mark = ',' if ',' in text else '.'
        if _GROUPING.match(text):
            text = text.replace(mark, '')
        elif _SIGNED_DECIMAL.match(text):
            text = text.replace(mark, '.')
        else:
            return None
    if not re.match(r'^\d+(?:\.\d+)?$', text):
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    rendered = format(number, 'f')
    if '.' in rendered:
        rendered = rendered.rstrip('0').rstrip('.')
    return ('' if number == 0 else sign) + rendered


def _parse_digits(value, length):
    digits = re.sub(r'\D', '', normalize(value))
    return digits if len(digits) == length else None


def normalize_value(field_type, value):
    """Canonical comparison form for one field value ('' means not extracted)."""
    text = normalize(value).casefold()
    if not text:
        return ''
    if field_type == 'date':
        return _parse_date(value) or text
    if field_type == 'money':
        return _parse_money(value) or text
    if field_type == 'nip':
        return _parse_digits(value, 10) or text
    if field_type == 'pesel':
        return _parse_digits(value, 11) or text
    if field_type == 'address':
        return normalize(_ADDRESS_DROP.sub(' ', text))
    return text


def coerce_value(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError('Field values must be strings or numbers')
    return str(value)


def _counts(records):
    return {'tp': sum(r[0] for r in records), 'fp': sum(r[1] for r in records),
            'fn': sum(r[2] for r in records)}


def _prf(counts):
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    if tp + fp + fn == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {'precision': precision, 'recall': recall, 'f1': f1}


def _report(records):
    counts = _counts(records)
    return {**counts, **_prf(counts)}


def _tally(reference, prediction):
    """(tp, fp, fn, verdict) for one field."""
    if reference and prediction:
        if reference == prediction:
            return 1, 0, 0, 'match'
        return 0, 1, 1, 'mismatch'
    if reference:
        return 0, 0, 1, 'missing'
    if prediction:
        return 0, 1, 0, 'spurious'
    return 0, 0, 0, 'absent'


def evaluate(manifest, predictions):
    manifest = Path(manifest)
    records = load_jsonl(manifest)
    if not records or len({row['id'] for row in records}) != len(records):
        raise ValueError('Manifest must contain unique nonempty records')
    rows = load_jsonl(Path(predictions))
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate predictions')
    supplied = {row['id']: row for row in rows}
    if set(supplied) - {row['id'] for row in records}:
        raise ValueError('Predictions contain unknown IDs')
    per_type, per_field, output = {}, {}, []
    for row in records:
        data = (manifest.parent / row['image']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Image checksum mismatch: {row["id"]}')
        doc_type = row.get('doc_type')
        if doc_type not in SCHEMAS:
            raise ValueError(f'Unknown document type: {row["id"]}')
        schema = SCHEMAS[doc_type]
        reference_fields = row.get('fields')
        if not isinstance(reference_fields, dict) or \
                set(reference_fields) - set(schema):
            raise ValueError(f'Manifest fields outside schema: {row["id"]}')
        prediction = supplied.get(row['id'])
        if prediction and prediction.get('status') not in ('ok', 'error'):
            raise ValueError('Prediction status must be ok or error')
        status = prediction['status'] if prediction else 'missing'
        predicted_fields = {}
        if status == 'ok':
            predicted_fields = prediction.get('fields')
        elif prediction and prediction.get('fields'):
            raise ValueError(f'Error predictions must have empty fields: {row["id"]}')
        if not isinstance(predicted_fields, dict) or set(predicted_fields) - set(schema):
            raise ValueError(f'Prediction fields must match schema: {row["id"]}')
        detail, entry_counts = {}, [0, 0, 0]
        for name, field_type in schema.items():
            reference = normalize_value(field_type, coerce_value(reference_fields[name])) \
                if name in reference_fields else ''
            predicted = normalize_value(field_type, coerce_value(predicted_fields[name])) \
                if name in predicted_fields else ''
            if status != 'ok':
                predicted = ''
            tp, fp, fn, verdict = _tally(reference, predicted)
            detail[name] = verdict
            entry_counts[0] += tp
            entry_counts[1] += fp
            entry_counts[2] += fn
            if verdict != 'absent':
                bucket = per_type.setdefault(field_type, [0, 0, 0])
                field = per_field.setdefault(f'{doc_type}.{name}', [0, 0, 0])
                for counter in (bucket, field):
                    counter[0] += tp
                    counter[1] += fp
                    counter[2] += fn
        entry = {'id': row['id'], 'status': status,
                 'tp': entry_counts[0], 'fp': entry_counts[1], 'fn': entry_counts[2],
                 'fields': detail}
        if prediction and 'elapsed_seconds' in prediction:
            entry['elapsed_seconds'] = prediction['elapsed_seconds']
        output.append(entry)
    totals = _report([(entry['tp'], entry['fp'], entry['fn']) for entry in output])
    return {'protocol_version': PROTOCOL_VERSION,
            'documents': len(output),
            'errors_or_missing': sum(entry['status'] != 'ok' for entry in output),
            'precision_micro': totals['precision'],
            'recall_micro': totals['recall'],
            'f1_micro': totals['f1'],
            'counts_micro': {key: totals[key] for key in ('tp', 'fp', 'fn')},
            'aggregation': 'field-level micro; mismatch counts FP+FN; error/missing '
                           'documents keep every non-empty reference field as FN',
            'normalization': 'per field type: date -> YYYY-MM-DD, money -> plain decimal, '
                             'nip -> 10 digits, pesel -> 11 digits, address/text/name -> '
                             'NFC + quotes + casefold + whitespace',
            'per_type': {kind: _report([tuple(counts)])
                         for kind, counts in sorted(per_type.items())},
            'per_field': {name: _report([tuple(counts)])
                          for name, counts in sorted(per_field.items())},
            'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'predictions_sha256': hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
            'results': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest')
    parser.add_argument('--predictions')
    parser.add_argument('--output')
    parser.add_argument('--dump-schemas', action='store_true',
                        help='print the field schemas as JSON and exit')
    args = parser.parse_args()
    if args.dump_schemas:
        print(json.dumps(SCHEMAS, ensure_ascii=False, indent=2))
        return
    if not (args.manifest and args.predictions and args.output):
        parser.error('--manifest, --predictions and --output are required')
    result = evaluate(args.manifest, args.predictions)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {key: result[key] for key in
               ('documents', 'errors_or_missing', 'precision_micro', 'recall_micro', 'f1_micro')}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
