"""Prepare a distinct exploratory corpus without changing frozen review gates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import zipfile

from training.adjudicate_reviews import text_hash
from training.build_annotation_review import issues
from training.full_page_pilot import digest, read_rows, safe_relative, write_json, write_rows
from training.merge_recognizer_reviewed_pool import verified_package


def prepare(sources, config, output):
    config, output = Path(config), Path(output)
    cfg = json.loads(config.read_text(encoding='utf-8'))
    if (cfg['schema'] != 'slayer-recognizer-reviewed-exploratory-training-v1'
            or cfg['independent_benchmark'] is not False or cfg['sota_claim'] is not False
            or cfg['bibliographic_independence_certified'] is not False
            or cfg['automatic_production_promotion'] is not False):
        raise ValueError('Only explicitly exploratory training is permitted')
    if output.exists() or output.with_suffix('.zip').exists():
        raise FileExistsError('Use a fresh corpus directory')
    if len(sources) != len(cfg['sources']):
        raise ValueError('Source count mismatch')
    rows, seen_ids, seen_hashes, ledger = [], set(), set(), []
    for root, expected in zip(map(Path, sources), cfg['sources']):
        verified_package(root)
        manifest = root/'manifest.jsonl'
        records = read_rows(manifest)
        if (digest(manifest) != expected['manifest_sha256']
                or digest(root/'checksums.json') != expected['checksums_sha256']
                or len(records) != expected['count']):
            raise ValueError('Frozen approved source mismatch')
        for row in records:
            approved = (row.get('annotation_verified') is True if expected['kind'] == 'confirmed-gated-expansion'
                else row.get('review_status') == 'verified' and row.get('line_geometry_verified') is True)
            if (not approved or row['geometry_decision'] != 'complete-line' or row['gold'] is not False
                    or row['source_split'] != 'train' or row['final_test'] is not False
                    or not row['text'].strip() or '\n' in row['text'] or '\r' in row['text']
                    or issues(row['text']) or row['reviewed_text_sha256'] != text_hash(row['text'])):
                raise ValueError('Unapproved or invalid source annotation')
            image = root/safe_relative(row['image'])
            if not image.resolve().is_relative_to(root.resolve()) or digest(image) != row['sha256']:
                raise ValueError('Crop identity mismatch')
            identifier = row.get('root_line_id', row['id'])
            if identifier in seen_ids or row['sha256'] in seen_hashes:
                raise ValueError('Duplicate approved source pair')
            seen_ids.add(identifier)
            seen_hashes.add(row['sha256'])
            excluded = row['collection'] in cfg['excluded_collections']
            split = ('excluded-work-family' if excluded else 'development'
                if row['collection'] == cfg['development_collection'] else 'train')
            entry = {**row, 'source_package_manifest_sha256': digest(manifest),
                'original_eligible_for_training': row['eligible_for_training'],
                'eligible_for_training': False, 'eligible_for_evaluation': False,
                'experimental_training_eligible': split == 'train',
                'split': split, 'independent_benchmark': False,
                'bibliographic_independence_certified': False, 'gold': False}
            ledger.append(entry)
            if not excluded:
                rows.append((entry, image))
    counts = Counter(r['split'] for r in ledger)
    actual = {'approved_source': len(ledger), 'excluded_work_family': counts['excluded-work-family'],
              'historical_train': counts['train'], 'historical_development': counts['development']}
    if actual != cfg['expected_counts']:
        raise ValueError(f'Unexpected selection counts: {actual}')
    train_pages = {r['page_id'] for r in ledger if r['split'] == 'train'}
    dev_pages = {r['page_id'] for r in ledger if r['split'] == 'development'}
    if train_pages & dev_pages:
        raise ValueError('Train/development source page overlap')
    output.mkdir(parents=True)
    for name in ('train', 'development'):
        (output/name).mkdir()
    manifest = []
    for row, image in rows:
        relative = row['split']+'/'+row['id']+'.png'
        target = output/safe_relative(relative)
        shutil.copyfile(image, target)
        target.with_suffix('.txt').write_text(row['text'], encoding='utf-8')
        manifest.append({**row, 'image': relative, 'text_sha256': text_hash(row['text'])})
    write_rows(output/'manifest.jsonl', manifest)
    write_rows(output/'source-ledger.jsonl', ledger)
    shutil.copyfile(config, output/'config.json')
    report = {'schema': cfg['schema'], 'config_sha256': digest(config), 'counts': actual,
        'excluded_collections': cfg['excluded_collections'],
        'development_collection': cfg['development_collection'],
        'train_development_page_overlap': [], 'exact_image_overlap': [],
        'source_review_packages_modified': False, 'experimental_training_requested_by_user': True,
        'certified_training_freeze_ready': False, 'independent_benchmark': False,
        'bibliographic_independence_certified': False, 'sota_claim': False,
        'limitations': 'New exploratory-use policy only; original strict eligibility gates unchanged. '
                      'Collection-held development is not certified work-independent gold.'}
    write_json(output/'report.json', report)
    write_json(output/'checksums.json', {p.relative_to(output).as_posix(): digest(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    with zipfile.ZipFile(output.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                archive.write(p, p.relative_to(output).as_posix())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))
