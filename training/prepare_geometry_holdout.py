"""Freeze a collection-disjoint geometry holdout before running OCR."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import unicodedata
from urllib.request import urlopen

from training.kaggle_printed_dev_control import DATASET, REVISION, FROZEN, safe_file

SAMPLE_COLLECTIONS = 12
RANK_SALT = 'polocrbench-geometry-holdout-v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rank(value):
    return digest((RANK_SALT + '\0' + value).encode('utf-8'))


def has_broken_reference(text):
    return '\ufffd' in text


def private_use_count(text):
    return sum(unicodedata.category(c) == 'Co' for c in text)


def select(train_rows, test_rows, used_pages, sample_collections=SAMPLE_COLLECTIONS):
    if not 1 <= sample_collections <= 22:
        raise ValueError('Expected 1-22 collections')
    test_hashes = {r['image_sha256'] for r in test_rows}
    seen, eligible, exclusions = set(), defaultdict(list), []
    for row in train_rows:
        if row['id'] in seen:
            raise ValueError('Duplicate region ID')
        seen.add(row['id'])
        if row['split'] != 'train' or row['collection'] in FROZEN:
            raise ValueError('Unexpected split or frozen collection')
        if row['image_sha256'] in test_hashes:
            raise ValueError('Exact image overlap with frozen test')
        reason = None
        lines = row['text'].splitlines()
        if row['page_id'] in used_pages:
            reason = 'page used during geometry development'
        elif row['region_type'] != 'paragraph':
            reason = 'not a paragraph'
        elif not 3 <= len(lines) <= 12:
            reason = 'outside 3-12 reference lines'
        elif any(not line.strip() for line in lines):
            reason = 'blank reference line'
        elif has_broken_reference(row['text']):
            reason = 'replacement reference character'
        elif row['license'] != 'CC-BY-3.0':
            reason = 'unexpected license'
        if reason:
            exclusions.append({'id': row['id'], 'reason': reason})
        else:
            eligible[row['collection']].append(row)
    collections = sorted(eligible, key=rank)[:sample_collections]
    if len(collections) != sample_collections:
        raise ValueError('Insufficient eligible collections')
    selected = [min(eligible[collection], key=lambda r: rank(r['id'])) for collection in collections]
    for collection, rows in eligible.items():
        for row in rows:
            if row not in selected:
                exclusions.append({'id': row['id'], 'reason': 'deterministic one-region collection quota'})
    return sorted(selected, key=lambda r: rank(r['id'])), exclusions


def source_path(row):
    name = row.get('file_name') or 'images/' + row['id'] + '.jpg'
    path = PurePosixPath('regions/train') / safe_file(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe source path')
    return path.as_posix()


def freeze(output, train_rows, test_rows, used_pages, fetch):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    selected, exclusions = select(train_rows, test_rows, used_pages)
    rows = []
    for row in selected:
        path = source_path(row)
        image = fetch(path)
        if digest(image) != row['image_sha256']:
            raise ValueError('Source image checksum mismatch: ' + row['id'])
        rows.append({k: row[k] for k in ['id', 'split', 'collection', 'page_id', 'text',
                                         'region_type', 'image_sha256', 'license']} | {
            'source_path': path, 'reference_status': 'upstream-unreviewed',
            'reference_private_use_count': private_use_count(row['text']),
            'eligible_for_benchmark': False})
    manifest = {'scope': 'geometry holdout diagnostic; not benchmark or model holdout',
                'dataset': DATASET, 'revision': REVISION, 'rank_salt': RANK_SALT,
                'selection_frozen_before_ocr': True, 'sample_collections': SAMPLE_COLLECTIONS,
                'used_geometry_development_pages': sorted(used_pages),
                'normalization': 'NFC and whitespace only; historical spelling retained',
                'selection_rules': ['train split', 'collection-disjoint from prior development and frozen test',
                                    'one paragraph region per collection', '3-12 nonblank lines',
                                    'exclude replacement characters; retain and flag private-use historical glyphs',
                                    'CC-BY-3.0 source images'],
                'limitations': 'Upstream references are not manually reviewed. Model training separation is based on its current card, not an independent leakage audit.',
                'regions': rows}
    output.mkdir(parents=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'exclusions.json').write_text(json.dumps(exclusions, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'checksums.json').write_text(json.dumps({
        'manifest.json': digest((output / 'manifest.json').read_bytes()),
        'exclusions.json': digest((output / 'exclusions.json').read_bytes())}, indent=2), encoding='utf-8')
    return manifest


def download_jsonl(path):
    url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{safe_file(path)}'
    with urlopen(url, timeout=120) as response:
        return [json.loads(line) for line in response.read().decode('utf-8').splitlines() if line.strip()]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--used-selection', default='data/body-dev-review-v2/selection.json')
    args = parser.parse_args()
    train = download_jsonl('regions/train/metadata.jsonl')
    test = download_jsonl('regions/test/metadata.jsonl')
    used = {r['page_id'] for r in json.loads(Path(args.used_selection).read_text(encoding='utf-8'))}
    def fetch(path):
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{safe_file(path)}'
        with urlopen(url, timeout=120) as response:
            return response.read()
    manifest = freeze(args.output, train, test, used, fetch)
    print(json.dumps({'regions': len(manifest['regions']),
                      'collections': [r['collection'] for r in manifest['regions']],
                      'reference_lines': sum(len(r['text'].splitlines()) for r in manifest['regions'])}, indent=2))


if __name__ == '__main__':
    main()
