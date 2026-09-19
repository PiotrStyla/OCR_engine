"""Validate subtask A predictions independently of OCR/ML dependencies."""
import argparse
import json
import math
from pathlib import Path

from training.stage_impact_benchmark import digest


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f'Non-finite JSON constant: {value}')


def load_jsonl(path):
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding='utf-8').split('\n'), 1):
        if line.strip():
            try:
                row = json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
            except ValueError as error:
                raise ValueError(f'Invalid JSONL record at line {index}: {error}') from error
            if not isinstance(row, dict):
                raise ValueError(f'Expected object at line {index}')
            rows.append(row)
    return rows


def unique_ids(rows, label):
    values = [row.get('id') for row in rows]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f'{label}: invalid ID')
    if len(set(values)) != len(values):
        raise ValueError(f'{label}: duplicate IDs')
    return set(values)


def validate(manifest, predictions, allow_missing=False):
    references, rows = load_jsonl(manifest), load_jsonl(predictions)
    if not references:
        raise ValueError('Empty manifest')
    expected, supplied = unique_ids(references, 'Manifest'), unique_ids(rows, 'Predictions')
    reference_by_id = {row['id']: row for row in references}
    if supplied - expected:
        raise ValueError('Predictions contain unknown IDs')
    for row in rows:
        for field in ('source_sha256', 'sha256'):
            if field in row and row[field] != reference_by_id[row['id']].get('sha256'):
                raise ValueError(f"Image hash mismatch for {row['id']}")
        if row.get('status') not in ('ok', 'error') or not isinstance(row.get('text'), str):
            raise ValueError(f"Invalid status/text for {row['id']}")
        if row['status'] == 'error' and row['text']:
            raise ValueError('Error predictions must have empty text')
        if 'elapsed_seconds' in row:
            elapsed = row['elapsed_seconds']
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Invalid elapsed_seconds')
    missing = sorted(expected - supplied)
    if missing and not allow_missing:
        raise ValueError(f'Missing {len(missing)} page predictions')
    return {'schema': 'polocrbench-submission-validation-v1', 'subtask': 'A',
            'manifest_sha256': digest(manifest), 'predictions_sha256': digest(predictions),
            'pages_expected': len(expected), 'pages_supplied': len(rows), 'missing_ids': missing,
            'error_pages': sum(row['status'] == 'error' for row in rows),
            'complete': not missing, 'eligible_for_scoring': not missing or allow_missing,
            'scope': 'Format/completeness only; not model-track eligibility or AmuEval certification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--allow-missing', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = validate(args.manifest, args.predictions, args.allow_missing)
    if args.output:
        with Path(args.output).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
