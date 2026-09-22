"""Validate PolOCRBench predictions independently of OCR/ML dependencies.

Subtasks: A transcription (Markdown text), B tables (HTML), C KIE (fields JSON).
Checks prediction JSONL format/completeness against the manifest and, when
given, the submission metadata (``submission_meta.json``) with the track rules:
``constrained`` (organizer data only, open-weight models), ``open`` (anything),
``zero-shot`` (no fine-tuning, the frozen organizer prompt hash). Training-data
and model claims are self-declared; this validator certifies their format, not
their truth.
"""
import argparse
import json
import math
from pathlib import Path

from training.stage_impact_benchmark import digest

SUBTASKS = ('A', 'B', 'C')
_PAYLOAD_KEYS = {'A': 'text', 'B': 'html', 'C': 'fields'}
TRACKS = ('constrained', 'open', 'zero-shot')
ORGANIZER_DATA_PREFIX = 'organizer:'
ZERO_SHOT_PROMPTS = {
    'polocrbench-zero-shot-prompt-v1':
        '69f80f1c0bbc2a0f1531babee5fa4a2e8b3f8db50ef8ea3426879469169c909e',
}


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


def validate_meta(meta):
    """Validate the submission declaration; claims stay self-declared."""
    if not isinstance(meta, dict):
        raise ValueError('Meta must be an object')
    if meta.get('schema') != 'polocrbench-submission-meta-v1':
        raise ValueError('Unknown meta schema')
    if not isinstance(meta.get('team'), str) or not meta['team'].strip():
        raise ValueError('Meta team must be a nonempty string')
    track = meta.get('track')
    if track not in TRACKS:
        raise ValueError('Meta track must be constrained, open or zero-shot')
    subtasks = meta.get('subtasks')
    if not isinstance(subtasks, list) or not subtasks or \
            any(task not in SUBTASKS for task in subtasks) or len(set(subtasks)) != len(subtasks):
        raise ValueError('Meta subtasks must be a nonempty unique subset of A, B, C')
    models = meta.get('models')
    if not isinstance(models, list) or not models:
        raise ValueError('Meta models must be a nonempty list')
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get('name'), str) \
                or not model['name'].strip():
            raise ValueError('Meta models need nonempty names')
        if not isinstance(model.get('open_weight'), bool):
            raise ValueError('Meta models need an open_weight flag')
    if not isinstance(meta.get('finetuned'), bool):
        raise ValueError('Meta finetuned must be true or false')
    data = meta.get('training_data')
    if not isinstance(data, list) or \
            any(not isinstance(item, str) or not item.strip() for item in data):
        raise ValueError('Meta training_data must be a list of dataset ids')
    if track == 'constrained':
        if any(not item.startswith(ORGANIZER_DATA_PREFIX) for item in data):
            raise ValueError('Constrained track allows only organizer: training data')
        if any(not model['open_weight'] for model in models):
            raise ValueError('Constrained track requires open-weight models')
    if track == 'zero-shot':
        if meta['finetuned'] or data:
            raise ValueError('Zero-shot track forbids fine-tuning and extra training data')
        version = meta.get('prompt_version')
        if version not in ZERO_SHOT_PROMPTS or meta.get('prompt_sha256') != ZERO_SHOT_PROMPTS[version]:
            raise ValueError('Zero-shot track requires the organizer prompt version and hash')
    return {'team': meta['team'].strip(), 'track': track, 'subtasks': subtasks,
            'models': [model['name'] for model in models],
            'open_weight': all(model['open_weight'] for model in models),
            'finetuned': meta['finetuned'], 'training_data': data}


def _check_manifest_payload(references, subtask):
    if subtask == 'A':
        return
    from training.kie_eval import SCHEMAS  # deferred: kie_eval imports this module
    for row in references:
        if subtask == 'B':
            if not isinstance(row.get('html'), str):
                raise ValueError(f'Manifest table HTML missing: {row.get("id")}')
        else:
            schema = SCHEMAS.get(row.get('doc_type'))
            fields = row.get('fields')
            if schema is None or not isinstance(fields, dict) or set(fields) - set(schema):
                raise ValueError(f'Manifest fields outside schema: {row.get("id")}')


def validate(manifest, predictions, allow_missing=False, subtask='A', meta=None):
    if subtask not in SUBTASKS:
        raise ValueError('Subtask must be A, B or C')
    references, rows = load_jsonl(manifest), load_jsonl(predictions)
    if not references:
        raise ValueError('Empty manifest')
    _check_manifest_payload(references, subtask)
    expected, supplied = unique_ids(references, 'Manifest'), unique_ids(rows, 'Predictions')
    reference_by_id = {row['id']: row for row in references}
    if supplied - expected:
        raise ValueError('Predictions contain unknown IDs')
    payload = _PAYLOAD_KEYS[subtask]
    for row in rows:
        for field in ('source_sha256', 'sha256'):
            if field in row and row[field] != reference_by_id[row['id']].get('sha256'):
                raise ValueError(f"Image hash mismatch for {row['id']}")
        value = row.get(payload)
        valid = isinstance(value, dict) if subtask == 'C' else isinstance(value, str)
        if row.get('status') not in ('ok', 'error') or not valid:
            raise ValueError(f'Invalid status/{payload} for {row["id"]}')
        if row['status'] == 'error' and value:
            raise ValueError(f'Error predictions must have empty {payload}')
        if 'elapsed_seconds' in row:
            elapsed = row['elapsed_seconds']
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Invalid elapsed_seconds')
    if subtask == 'C':
        from training.kie_eval import SCHEMAS  # deferred: kie_eval imports this module
        for row in rows:
            schema = SCHEMAS.get(reference_by_id[row['id']].get('doc_type'))
            if schema is None or set(row['fields']) - set(schema):
                raise ValueError(f'Fields outside the document schema for {row["id"]}')
            for value in row['fields'].values():
                if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                    raise ValueError(f'Invalid field value for {row["id"]}')
    missing = sorted(expected - supplied)
    if missing and not allow_missing:
        raise ValueError(f'Missing {len(missing)} page predictions')
    return {'schema': 'polocrbench-submission-validation-v1', 'subtask': subtask,
            'manifest_sha256': digest(manifest), 'predictions_sha256': digest(predictions),
            'pages_expected': len(expected), 'pages_supplied': len(rows), 'missing_ids': missing,
            'error_pages': sum(row['status'] == 'error' for row in rows),
            'complete': not missing, 'eligible_for_scoring': not missing or allow_missing,
            'scope': 'Format/completeness and declared track metadata only; '
                     'training-data and model claims are self-declared.',
            **({'meta': validate_meta(meta)} if meta is not None else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--allow-missing', action='store_true')
    parser.add_argument('--subtask', choices=SUBTASKS, default='A')
    parser.add_argument('--meta')
    parser.add_argument('--output')
    args = parser.parse_args()
    meta = json.loads(Path(args.meta).read_text(encoding='utf-8')) if args.meta else None
    report = validate(args.manifest, args.predictions, args.allow_missing,
                      subtask=args.subtask, meta=meta)
    if args.output:
        with Path(args.output).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
