"""Build a training-source-only teacher pilot, without granting label eligibility."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

from PIL import Image

from training.full_page_pilot import digest, read_rows, safe_relative, write_json, write_rows
from training.prepare_historical_line_corpus import DATASET, REVISION, METADATA_SHA256

TARGETS = '\u017f\u00e1\u0247'
SALT = 'slayer-recognizer-data-v3-pilot-v1'
SOURCE_MANIFEST = 'dbed37d35d77fe85af634dcdd353f628f0c318b37b02cff08520eb746f090314'


def choose(rows, count=64, max_per_collection=8, max_per_page=4):
    if not 1 <= count <= 256 or max_per_collection < 1 or max_per_page < 1:
        raise ValueError('Invalid bounded pilot quotas')
    groups = defaultdict(list)
    for row in rows:
        rank = hashlib.sha256((SALT+':'+row['id']).encode()).hexdigest()
        groups[row['collection']].append((row, rank))
    for group in groups.values():
        group.sort(key=lambda item: (-sum(char in item[0]['text'] for char in TARGETS),
                                    '\u0247' not in item[0]['text'], item[1]))
    selected, pages, collections = [], Counter(), Counter()
    while len(selected) < count:
        progressed = False
        for collection in sorted(groups):
            if collections[collection] >= max_per_collection:
                continue
            while groups[collection]:
                row, rank = groups[collection].pop(0)
                if pages[row['page_id']] >= max_per_page:
                    continue
                selected.append({**row, 'selection_rank_sha256': rank})
                pages[row['page_id']] += 1
                collections[collection] += 1
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError('Insufficient distinct training sources under frozen quotas')
    return selected


def prepare(corpus, train_regions, test_regions, validation, holdout, output, *, count=64):
    corpus, train_regions, test_regions, validation, holdout, output = map(
        Path, (corpus, train_regions, test_regions, validation, holdout, output))
    if output.exists():
        raise FileExistsError('Use a new pilot directory')
    manifest = corpus/'manifest.jsonl'
    if digest(manifest) != SOURCE_MANIFEST:
        raise ValueError('Historical corpus manifest revision mismatch')
    for path, split in ((train_regions, 'train'), (test_regions, 'test')):
        if digest(path) != METADATA_SHA256[split]:
            raise ValueError('Source metadata revision mismatch')
    frozen = json.loads(holdout.read_text(encoding='utf-8'))
    if frozen.get('dataset') != DATASET or frozen.get('revision') != REVISION:
        raise ValueError('Geometry holdout provenance mismatch')
    rows = read_rows(manifest)
    validation_rows = read_rows(validation)
    tests = read_rows(test_regions)
    sources = {row['id']: row for row in read_rows(train_regions)}
    if len({row['id'] for row in rows}) != len(rows) or not validation_rows or not tests:
        raise ValueError('Unique corpus IDs and nonempty forbidden sets required')
    forbidden_rows = validation_rows + tests + frozen['regions'] + [r for r in rows if r['split'] != 'train']
    forbidden_collections = {row['collection'] for row in forbidden_rows}
    forbidden_pages = {row.get('page_id', row['id']) for row in forbidden_rows}
    forbidden_hashes = {row[key] for row in forbidden_rows for key in
                        ('sha256', 'image_sha256', 'source_sha256', 'source_image_sha256') if row.get(key)}
    candidates, excluded = [], []
    seen_hashes = set()
    for row in rows:
        if row['split'] != 'train':
            excluded.append({'id': row['id'], 'reason': 'not source train split'})
            continue
        if not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']):
            raise ValueError('Unsafe line ID')
        source = sources.get(row['source_region_id'])
        if (not source or source['split'] != 'train' or source['page_id'] != row['page_id']
                or source['collection'] != row['collection']
                or source['image_sha256'] != row['source_image_sha256']):
            raise ValueError('Line/source region identity mismatch')
        if (row['collection'] in forbidden_collections or row['page_id'] in forbidden_pages
                or row['image_sha256'] in forbidden_hashes
                or row['source_image_sha256'] in forbidden_hashes):
            raise ValueError('Training-source overlap with validation/test/geometry holdout')
        image, label = corpus/'train'/(row['id']+'.png'), corpus/'train'/(row['id']+'.txt')
        if digest(image) != row['image_sha256'] or digest(label) != row['text_sha256']:
            raise ValueError('Line image or label checksum mismatch')
        if row['image_sha256'] in seen_hashes:
            raise ValueError('Duplicate training crop hash')
        seen_hashes.add(row['image_sha256'])
        text = label.read_text(encoding='utf-8').rstrip('\n')
        if not any(char in text for char in TARGETS):
            excluded.append({'id': row['id'], 'reason': 'no targeted historical glyph'})
            continue
        if row['license'] != 'CC-BY-3.0':
            raise ValueError('Unexpected source license')
        with Image.open(image) as scan:
            width, height = scan.size
            if scan.format != 'PNG':
                raise ValueError('Expected PNG line crops')
        candidates.append({**row, 'text': text, 'width': width, 'height': height,
            'source_region_bbox': source['bbox'], 'source_region_text': source['text']})
    selected = choose(candidates, count)
    output.mkdir(parents=True)
    (output/'images').mkdir()
    records, inputs = [], []
    for row in selected:
        image = 'images/'+row['id']+'.png'
        shutil.copyfile(corpus/'train'/(row['id']+'.png'), output/image)
        record = {**row, 'image': image, 'sha256': row['image_sha256'],
            'source_split': 'train', 'split': 'training-review',
            'reference_status': 'automatic-count-matched-unreviewed',
            'line_geometry_verified': False, 'eligible_for_training': False,
            'eligible_for_evaluation': False, 'final_test': False,
            'dataset': DATASET, 'revision': REVISION}
        records.append(record)
        inputs.append({key: record[key] for key in ('id', 'image', 'sha256', 'width', 'height')}
                      | {'source_regions': []})
    write_rows(output/'manifest.jsonl', records)
    write_rows(output/'inference-inputs.jsonl', inputs)
    write_rows(output/'excluded.jsonl', excluded)
    report = {'schema': 'slayer-recognizer-data-pilot-v1', 'source_dataset': DATASET,
        'source_revision': REVISION, 'source_manifest_sha256': digest(manifest),
        'train_regions_sha256': digest(train_regions), 'test_regions_sha256': digest(test_regions),
        'validation_manifest_sha256': digest(validation), 'geometry_holdout_sha256': digest(holdout),
        'lines': len(records), 'pages': len({row['page_id'] for row in records}),
        'collections': dict(Counter(row['collection'] for row in records)),
        'target_line_counts': {char: sum(char in row['text'] for row in records) for char in TARGETS},
        'source_train_lines': sum(row['split'] == 'train' for row in rows),
        'target_candidate_lines': len(candidates), 'excluded_records': len(excluded),
        'forbidden_collections': sorted(forbidden_collections),
        'collection_overlap': [], 'page_overlap': [], 'exact_hash_overlap': [],
        'selection_salt': SALT, 'selection_by_model_predictions': False,
        'max_per_collection': 8, 'max_per_page': 4, 'reference_text_sent_to_model': False,
        'training_examples_created': 0, 'gold_labels_created': 0,
        'normalization': 'none; retain source label text, drop final file newline only',
        'limitations': 'Automatic crop/reference alignment needs visual review. Collection/page/hash firewall only; work/edition and near-duplicate audits are not complete. Training-source teacher proposals, not a benchmark or clean training labels.'}
    write_json(output/'report.json', report)
    files = [path for path in output.rglob('*') if path.is_file()]
    write_json(output/'checksums.json', {path.relative_to(output).as_posix(): digest(path) for path in files})
    archive = output/'recognizer-data-v3-pilot-input.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for path in files + [output/'checksums.json']:
            stream.write(path, path.relative_to(output).as_posix())
    return {**report, 'archive': str(archive), 'archive_sha256': digest(archive),
            'manifest_sha256': digest(output/'manifest.jsonl')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'train-regions', 'test-regions', 'validation', 'holdout', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--count', type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(prepare(args.corpus, args.train_regions, args.test_regions,
        args.validation, args.holdout, args.output, count=args.count), ensure_ascii=False, indent=2))
