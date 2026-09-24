"""Create an aggregate recognition-error profile without publishing OCR text."""
from collections import Counter
import argparse
import json
from pathlib import Path
import unicodedata
import zipfile

from training.audit_geometry_holdout import audit
from training.geometry_holdout_runner import normalize

MODEL = 'PiotrSty/trocr-pl-mixed-v3'
VARIANT = 'rectangle'


def edit_alignment(reference, hypothesis):
    n, m = len(reference), len(hypothesis)
    costs = [list(range(m + 1))]
    moves = [[''] + ['I'] * m]
    for i in range(1, n + 1):
        costs.append([i] + [0] * m)
        moves.append(['D'] + [''] * m)
        for j in range(1, m + 1):
            options = [(costs[i - 1][j] + 1, 'D'), (costs[i][j - 1] + 1, 'I'),
                       (costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                        'M' if reference[i - 1] == hypothesis[j - 1] else 'S')]
            costs[i][j], moves[i][j] = min(options, key=lambda x: (x[0], 'MSDI'.index(x[1])))
    result, i, j = [], n, m
    while i or j:
        move = moves[i][j]
        if move in {'M', 'S'}:
            result.append((move, reference[i - 1], hypothesis[j - 1])); i -= 1; j -= 1
        elif move == 'D':
            result.append((move, reference[i - 1], None)); i -= 1
        else:
            result.append((move, None, hypothesis[j - 1])); j -= 1
    return list(reversed(result))


def base_character(character):
    if character is None:
        return None
    return ''.join(c for c in unicodedata.normalize('NFD', character)
                   if unicodedata.category(c) != 'Mn')


def classify(operation, reference, hypothesis):
    if reference is not None and unicodedata.category(reference) == 'Co':
        return 'private_use_reference'
    if reference in {'ſ', 's'} or hypothesis in {'ſ', 's'}:
        return 'long_s_or_s'
    if operation == 'S' and reference.lower() == hypothesis.lower():
        return 'case_only'
    if operation == 'S' and base_character(reference).lower() == base_character(hypothesis).lower():
        return 'diacritic'
    if any(c is not None and unicodedata.category(c).startswith('P') for c in [reference, hypothesis]):
        return 'punctuation'
    if reference == ' ' or hypothesis == ' ':
        return 'whitespace'
    return 'other'


def analyze(evidence_path, manifest_path):
    verified = audit(evidence_path, manifest_path)
    with zipfile.ZipFile(evidence_path) as archive:
        manifest = json.loads(archive.read('input-manifest.json'))
        predictions = json.loads(archive.read(MODEL.replace('/', '--') + '.json'))[VARIANT]
    rows = manifest['regions']
    categories, operations = Counter(), Counter()
    reference_errors, inserted = Counter(), Counter()
    per_region = []
    for row, prediction in zip(rows, predictions):
        alignment = edit_alignment(normalize(row['text']), normalize(prediction['text']))
        edits = 0
        for operation, reference, hypothesis in alignment:
            operations[operation] += 1
            if operation == 'M':
                continue
            edits += 1
            categories[classify(operation, reference, hypothesis)] += 1
            if reference is not None:
                reference_errors[f'U+{ord(reference):04X} {reference}'] += 1
            if operation == 'I':
                inserted[f'U+{ord(hypothesis):04X} {hypothesis}'] += 1
        per_region.append({'id': row['id'], 'collection': row['collection'],
                           'reference_characters': len(normalize(row['text'])), 'character_edits': edits,
                           'cer': edits / len(normalize(row['text'])),
                           'reference_lines': len(row['text'].splitlines()),
                           'detected_lines': prediction['detected_lines'],
                           'reference_private_use_count': row['reference_private_use_count']})
    per_region.sort(key=lambda r: (-r['cer'], r['id']))
    return {'scope': 'aggregate error profile; raw OCR text excluded', 'model': MODEL,
            'model_revision': '85d0c91c26f8e088849096dded7c9ba10b4cd9c9',
            'variant': VARIANT, 'evidence_zip_sha256': verified['evidence_zip_sha256'],
            'manifest_sha256': verified['manifest_sha256'], 'normalization': manifest['normalization'],
            'regions': len(rows), 'reference_characters': sum(r['reference_characters'] for r in per_region),
            'total_character_edits': sum(r['character_edits'] for r in per_region),
            'operation_counts': dict(operations), 'error_categories': dict(categories),
            'top_reference_characters_in_errors': reference_errors.most_common(20),
            'top_inserted_characters': inserted.most_common(20), 'per_region': per_region,
            'data_quality': {'regions_with_private_use_references': sum(r['reference_private_use_count'] > 0 for r in per_region),
                             'private_use_reference_characters': sum(r['reference_private_use_count'] for r in per_region),
                             'regions_with_line_count_mismatch': sum(r['reference_lines'] != r['detected_lines'] for r in per_region),
                             'upstream_references_manually_reviewed': False},
            'policy': 'All categories remain errors in primary CER. Diacritic and long-s diagnostics never modernize the reference.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.evidence, args.manifest)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'per_region'}}, ensure_ascii=True, indent=2))
