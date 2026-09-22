"""Cross-split integrity checks for PolOCRBench datasets.

Deterministic gate over two or more split manifests (standard library plus PIL
for perceptual hashes):

- exact duplicates: page image SHA-256, reference-text SHA-256 and record ids
  shared between splits (violations);
- near-duplicates: word 3-gram similarity on references (violations) and 64-bit
  difference hash (dHash) Hamming distance on page images (warnings — same
  template pages legitimately share a layout; a duplicated document always
  shares its reference text) — ``--near-image-distance`` (default 6),
  ``--near-text-similarity`` (default 0.9) over Jaccard and containment, so
  re-scans, formatting-only copies and a train region copied from a test page
  are all caught (the open question in ``benchmarks/polocrbench/README.md``:
  shared hashes alone rule out none of these);
- holdout claims: ``--holdout-field`` with ``--holdout-split`` verifies that
  the Test B values of that row/metadata field are disjoint from every other
  split (Test B must contain document types and degradations absent from
  train/Test A). Values come from manifest rows or, with ``--metadata``
  (``generation.json``), from the generator's per-sample records.

Image files missing locally are skipped for image checks and counted in the
report (fresh checkouts may lack the frozen bytes); text and hash checks still
run. Exits 1 when any violation is found, 0 otherwise.

Usage:
  python -m training.check_split_integrity --split train=.../history_train_pool.jsonl \\
      --split testA=.../history_testA_manifest.jsonl --output report.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import sys
from pathlib import Path

from training.stage_impact_benchmark import digest
from training.transcription_eval import normalize
from training.validate_submission import load_jsonl, unique_ids

PROTOCOL_VERSION = 'polocrbench-split-integrity-v1'
_SHINGLE = 3


def dhash(image_bytes):
    """64-bit difference hash; robust to re-encoding and light degradation."""
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes)).convert('L').resize((9, 8))
    pixels = list(image.tobytes())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | (pixels[row * 9 + column] < pixels[row * 9 + column + 1])
    return value


def hamming(a, b):
    return bin(a ^ b).count('1')


def shingles(text):
    words = normalize(text).casefold().split()
    return {tuple(words[i:i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)}


def text_similarity(a, b):
    """(jaccard, containment) over word 3-grams; containment catches subpages."""
    if not a or not b:
        return 0.0, 0.0
    overlap = len(a & b)
    union = len(a | b)
    return (overlap / union if union else 0.0,
            overlap / min(len(a), len(b)))


def load_split(name, manifest, metadata=None):
    manifest = Path(manifest)
    rows = load_jsonl(manifest)
    if not rows:
        raise ValueError(f'Empty manifest: {name}')
    unique_ids(rows, name)
    metadata_path = Path(metadata) if metadata else manifest.parent / 'generation.json'
    samples = {}
    if metadata_path.exists():
        report = json.loads(metadata_path.read_text(encoding='utf-8'))
        samples = {sample['id']: sample for sample in report.get('samples', [])}
    entries = []
    for row in rows:
        text = row.get('text') if isinstance(row.get('text'), str) else \
            row.get('html') if isinstance(row.get('html'), str) else ''
        image_path = (manifest.parent / row['image']) if row.get('image') else None
        entries.append({'id': row['id'], 'row': row, 'sample': samples.get(row['id']),
                        'text_sha': hashlib.sha256(text.encode('utf-8')).hexdigest() if text else None,
                        'shingles': shingles(text),
                        'image_sha': row.get('sha256'),
                        'image_path': image_path if image_path and image_path.exists() else None,
                        'dhash': None})
    return {'name': name, 'manifest': str(manifest), 'manifest_sha256': digest(manifest),
            'records': len(rows), 'samples': entries}


def holdout_value(entry, field):
    value = entry['row'].get(field)
    if value is None and entry['sample'] is not None:
        value = entry['sample'].get(field)
        if value is None and field == 'degradation':
            value = (entry['sample'].get('recipe') or {}).get('kind')
    return value


def compare(entry_a, entry_b, near_image, near_text):
    found = {'exact_image': [], 'exact_text': [], 'near_image': [], 'near_text': []}
    for a in entry_a:
        for b in entry_b:
            if a['image_sha'] and a['image_sha'] == b['image_sha']:
                found['exact_image'].append([a['id'], b['id']])
            elif a['dhash'] is not None and b['dhash'] is not None:
                distance = hamming(a['dhash'], b['dhash'])
                if distance <= near_image:
                    found['near_image'].append({'a': a['id'], 'b': b['id'],
                                                'distance': distance})
            if a['text_sha'] and a['text_sha'] == b['text_sha']:
                found['exact_text'].append([a['id'], b['id']])
                continue
            jaccard, containment = text_similarity(a['shingles'], b['shingles'])
            if max(jaccard, containment) >= near_text:
                found['near_text'].append({'a': a['id'], 'b': b['id'],
                                           'jaccard': round(jaccard, 4),
                                           'containment': round(containment, 4)})
    return found


def check(splits, holdout_field=None, holdout_split=None,
          near_image_distance=6, near_text_similarity=0.9):
    """Compare every split pair; returns the full report including violations."""
    for split in splits:
        missing = [entry['id'] for entry in split['samples'] if entry['image_path'] is None]
        split['images_missing'] = len(missing)
        for entry in split['samples']:
            if entry['image_path'] is not None:
                entry['dhash'] = dhash(entry['image_path'].read_bytes())
    pairs = []
    violations = 0
    warnings = 0
    for first, second in itertools.combinations(splits, 2):
        found = compare(first['samples'], second['samples'],
                        near_image_distance, near_text_similarity)
        shared_ids = sorted({entry['id'] for entry in first['samples']} &
                            {entry['id'] for entry in second['samples']})
        # Image look-alikes are warnings: same-template pages legitimately share a
        # layout. A duplicated document always shares its reference text, so the
        # text signals and id collisions are hard violations.
        hard = (len(found['exact_image']) + len(found['exact_text'])
                + len(found['near_text']) + len(shared_ids))
        violations += hard
        warnings += len(found['near_image'])
        pairs.append({'a': first['name'], 'b': second['name'],
                      **{key: value for key, value in found.items()},
                      'shared_ids': shared_ids,
                      'violations': hard, 'warnings': len(found['near_image'])})
    holdout = None
    if holdout_field:
        if holdout_split not in {split['name'] for split in splits}:
            raise ValueError(f'Unknown holdout split: {holdout_split}')
        others = [split for split in splits if split['name'] != holdout_split]
        claimed = {}
        for split in others:
            for entry in split['samples']:
                value = holdout_value(entry, holdout_field)
                if value is not None:
                    claimed.setdefault(str(value), []).append(entry['id'])
        overlaps, unknown = [], 0
        for entry in next(split for split in splits if split['name'] == holdout_split)['samples']:
            value = holdout_value(entry, holdout_field)
            if value is None:
                unknown += 1
            elif str(value) in claimed:
                overlaps.append({'value': str(value), 'id': entry['id'],
                                 'other_ids': claimed[str(value)][:5]})
        violations += len(overlaps)
        holdout = {'field': holdout_field, 'split': holdout_split,
                   'overlaps': overlaps, 'rows_without_value': unknown}
    return {'schema': PROTOCOL_VERSION,
            'splits': {split['name']: {'manifest_sha256': split['manifest_sha256'],
                                       'records': split['records'],
                                       'images_missing': split['images_missing']}
                       for split in splits},
            'thresholds': {'near_image_distance': near_image_distance,
                           'near_text_similarity': near_text_similarity},
            'pairs': pairs, 'holdout': holdout,
            'violations': violations, 'warnings': warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', action='append', required=True,
                        help='name=manifest.jsonl (repeatable, at least two)')
    parser.add_argument('--metadata',
                        help='per-sample fields override for every split; by default '
                             'each split reads generation.json next to its manifest')
    parser.add_argument('--holdout-field')
    parser.add_argument('--holdout-split')
    parser.add_argument('--near-image-distance', type=int, default=6)
    parser.add_argument('--near-text-similarity', type=float, default=0.9)
    parser.add_argument('--output')
    args = parser.parse_args()
    splits = []
    for spec in args.split:
        name, _, manifest = spec.partition('=')
        if not (name and manifest):
            parser.error('--split expects name=manifest.jsonl')
        splits.append(load_split(name, manifest, args.metadata))
    if len(splits) < 2:
        parser.error('at least two --split manifests are required')
    report = check(splits, holdout_field=args.holdout_field, holdout_split=args.holdout_split,
                   near_image_distance=args.near_image_distance,
                   near_text_similarity=args.near_text_similarity)
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding='utf-8', newline='\n')
    print(json.dumps({'splits': {name: data['records'] for name, data in report['splits'].items()},
                      'violations': report['violations'],
                      'warnings': report['warnings']}, indent=2))
    sys.exit(1 if report['violations'] else 0)


if __name__ == '__main__':
    main()
