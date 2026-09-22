"""AmuEval-compatible submission TSV for PolOCRBench (pack/unpack).

Deterministic, stdlib only. Protocol ``polocrbench-submission-tsv-v1``.

AmuEval aligns submissions by row order with ``test/in.tsv``. Here:

- ``in.tsv`` holds one instance id per line; that order is the submission
  order (ids are dropped in ``out.tsv`` and restored on unpack);
- ``out.tsv`` holds one payload per line, UTF-8, LF or CRLF: subtask A the
  Markdown transcription, B the table HTML, C a compact JSON object
  (``sort_keys`` order);
- TSV cells cannot hold raw control characters, so cell text is escaped:
  ``\\`` -> ``\\\\``, tab -> ``\\t``, LF -> ``\\n``, CR -> ``\\r``. Unpack is
  the exact inverse and rejects dangling or unknown escapes.

Empty cells mean empty output. A canonical JSONL run keeps ``status: error``
rows, but the TSV form cannot: pack maps them to empty cells and unpack
returns ``status: ok`` with empty payloads. Pack requires exactly the ids of
``in.tsv`` (no missing, no extra, no duplicates).

Usage:
  python -m training.submission_tsv --mode pack --subtask A --in-tsv in.tsv \
      --predictions run.jsonl --out-tsv out.tsv
  python -m training.submission_tsv --mode unpack --subtask A --in-tsv in.tsv \
      --out-tsv out.tsv --predictions run.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.validate_submission import load_jsonl

PROTOCOL_VERSION = 'polocrbench-submission-tsv-v1'
SUBTASKS = ('A', 'B', 'C')
_PAYLOAD_KEYS = {'A': 'text', 'B': 'html', 'C': 'fields'}
_ENCODE = str.maketrans({'\\': '\\\\', '\t': '\\t', '\n': '\\n', '\r': '\\r'})
_DECODE = {'\\': '\\', 't': '\t', 'n': '\n', 'r': '\r'}


def encode_cell(text):
    return text.translate(_ENCODE)


def decode_cell(text):
    out, index = [], 0
    while index < len(text):
        char = text[index]
        if char != '\\':
            out.append(char)
            index += 1
            continue
        index += 1
        if index >= len(text):
            raise ValueError('Dangling escape in TSV cell')
        if text[index] not in _DECODE:
            raise ValueError(f'Unknown escape in TSV cell: \\{text[index]}')
        out.append(_DECODE[text[index]])
        index += 1
    return ''.join(out)


def read_ids(path):
    ids = [line for line in Path(path).read_text(encoding='utf-8').splitlines()]
    if not ids or any(not instance.strip() for instance in ids):
        raise ValueError('in.tsv must hold one nonempty id per line')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate ids in in.tsv')
    return ids


def _payload(row, subtask):
    if row.get('status') == 'error':
        return ''
    value = row.get(_PAYLOAD_KEYS[subtask])
    if subtask == 'C':
        return json.dumps(value or {}, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'))
    return value if isinstance(value, str) else ''


def pack(in_tsv, predictions, out_tsv, subtask):
    """Write out.tsv rows in in.tsv order from canonical prediction JSONL."""
    if subtask not in SUBTASKS:
        raise ValueError('Subtask must be A, B or C')
    ids = read_ids(in_tsv)
    rows = load_jsonl(Path(predictions))
    supplied = {row.get('id'): row for row in rows}
    if len(supplied) != len(rows):
        raise ValueError('Duplicate predictions')
    if set(supplied) - set(ids):
        raise ValueError('Predictions contain unknown IDs')
    missing = sorted(set(ids) - set(supplied))
    if missing:
        raise ValueError(f'Missing {len(missing)} page predictions')
    cells = [encode_cell(_payload(supplied[instance], subtask)) for instance in ids]
    Path(out_tsv).write_text('\n'.join(cells) + '\n', encoding='utf-8', newline='\n')
    return {'protocol_version': PROTOCOL_VERSION, 'mode': 'pack', 'subtask': subtask,
            'rows': len(ids), 'out_tsv': str(out_tsv)}


def unpack(in_tsv, out_tsv, predictions, subtask):
    """Rebuild canonical prediction JSONL from out.tsv rows in in.tsv order."""
    if subtask not in SUBTASKS:
        raise ValueError('Subtask must be A, B or C')
    ids = read_ids(in_tsv)
    cells = Path(out_tsv).read_text(encoding='utf-8').splitlines()
    if len(cells) != len(ids):
        raise ValueError(f'out.tsv rows ({len(cells)}) must match in.tsv ids ({len(ids)})')
    rows = []
    for instance, cell in zip(ids, cells):
        payload = decode_cell(cell)
        if subtask == 'C':
            fields = json.loads(payload) if payload.strip() else {}
            if not isinstance(fields, dict):
                raise ValueError(f'KIE payload must be a JSON object: {instance}')
            rows.append({'id': instance, 'status': 'ok', 'fields': fields})
        else:
            rows.append({'id': instance, 'status': 'ok', _PAYLOAD_KEYS[subtask]: payload})
    Path(predictions).write_text('\n'.join(json.dumps(row, ensure_ascii=False)
                                           for row in rows), encoding='utf-8', newline='\n')
    return {'protocol_version': PROTOCOL_VERSION, 'mode': 'unpack', 'subtask': subtask,
            'rows': len(ids), 'predictions': str(predictions)}


def validate_tsv(in_tsv, out_tsv):
    """Row-count/order check exactly as AmuEval aligns submissions."""
    ids = read_ids(in_tsv)
    cells = Path(out_tsv).read_text(encoding='utf-8').splitlines()
    if len(cells) != len(ids):
        raise ValueError(f'out.tsv rows ({len(cells)}) must match in.tsv ids ({len(ids)})')
    return {'protocol_version': PROTOCOL_VERSION, 'rows': len(ids),
            'order': 'out.tsv rows align with in.tsv ids'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('pack', 'unpack'), required=True)
    parser.add_argument('--subtask', choices=SUBTASKS, required=True)
    parser.add_argument('--in-tsv', required=True)
    parser.add_argument('--out-tsv', required=True)
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()
    if args.mode == 'pack':
        report = pack(args.in_tsv, args.predictions, args.out_tsv, args.subtask)
    else:
        report = unpack(args.in_tsv, args.out_tsv, args.predictions, args.subtask)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
