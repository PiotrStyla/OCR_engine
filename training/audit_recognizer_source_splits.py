"""Bounded CPU split audit; digital page editions are not verified work IDs."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

from PIL import Image

from training.benchmark_pages import normalize
from training.check_split_integrity import dhash, hamming, text_similarity
from training.full_page_pilot import digest, fetch, read_rows, safe_relative, write_json, write_rows
from training.prepare_historical_line_corpus import DATASET, REVISION

PAGE_METADATA = {
    'train': ('991ef94a2e2fb3ac519ec3582fa588b1a75a92a87b1b002468cf7c31d3c10427', 65),
    'validation': ('536fc014963c2c55626e89f9bd07500826cb930f05ca3f8848352ca80c8856b1', 15),
    'test': ('b5aac0c527e4bd1b5e603cd26069f040f8b390666d43ffa306062f90892a4024', 9),
}


def diplomatic_shingles(text, size=7):
    words = normalize(text).split()
    return {tuple(words[i:i+size]) for i in range(len(words)-size+1)}


def compare_pages(rows, *, near_image_distance=4, near_text_similarity=.95):
    if not 0 <= near_image_distance <= 64 or not 0 < near_text_similarity <= 1:
        raise ValueError('Invalid similarity thresholds')
    flags, comparisons = [], 0
    for a, b in itertools.combinations(rows, 2):
        if a['audit_split'] == b['audit_split']:
            continue
        comparisons += 1
        reasons = []
        if a['image_sha256'] == b['image_sha256']:
            reasons.append('exact-image')
        if a['document_id'] == b['document_id']:
            reasons.append('shared-source-document-id')
        if a['edition_id'] == b['edition_id']:
            reasons.append('shared-digital-edition-id')
        distance = hamming(int(a['dhash'], 16), int(b['dhash'], 16))
        ratio_delta = abs((a['width']/a['height'])/(b['width']/b['height'])-1)
        if distance <= near_image_distance and ratio_delta <= .1:
            reasons.append('perceptual-image-proposal')
        sa, sb = diplomatic_shingles(a['text']), diplomatic_shingles(b['text'])
        jaccard, containment = text_similarity(sa, sb)
        if min(len(sa), len(sb)) >= 12 and max(jaccard, containment) >= near_text_similarity:
            reasons.append('long-text-overlap-proposal')
        if reasons:
            flags.append({'a': a['id'], 'b': b['id'], 'split_a': a['audit_split'], 'split_b': b['audit_split'],
                'reasons': reasons, 'dhash_distance': distance,
                'text_jaccard': jaccard, 'text_containment': containment,
                'status': 'needs-human-overlap-review', 'proven_duplicate': False})
    return flags, comparisons


def audit(metadata_directory, holdout, pilot_config, output, *, opener=urlopen, image_cache=None):
    metadata_directory, holdout, pilot_config, output = map(Path, (metadata_directory, holdout, pilot_config, output))
    if output.exists():
        raise FileExistsError('Use a fresh split-audit directory')
    frozen = json.loads(holdout.read_text(encoding='utf-8'))
    pilot = json.loads(pilot_config.read_text(encoding='utf-8'))
    if frozen.get('dataset') != DATASET or frozen.get('revision') != REVISION:
        raise ValueError('Holdout source mismatch')
    holdout_collections = {r['collection'] for r in frozen['regions']}
    forbidden = set(pilot['dataset']['forbidden_collections'])
    rows, hashes, ids = [], {}, set()
    for split, (sha, count) in PAGE_METADATA.items():
        path = metadata_directory/'pages'/split/'metadata.jsonl'
        if digest(path) != sha:
            raise ValueError('Frozen page metadata mismatch')
        records = read_rows(path)
        if len(records) != count:
            raise ValueError('Frozen page count mismatch')
        hashes[split] = sha
        for r in records:
            if (r['split'] != split or r['id'] in ids or not r['id']
                    or type(r.get('edition_id')) is not int or type(r.get('document_id')) is not int
                    or r['width'] <= 0 or r['height'] <= 0 or not isinstance(r['text'], str)):
                raise ValueError('Invalid page metadata')
            ids.add(r['id'])
            kind = 'geometry-holdout' if split == 'train' and r['collection'] in holdout_collections else split
            if kind == 'train' and r['collection'] in forbidden:
                raise ValueError('Unexpected forbidden training collection')
            rows.append({**r, 'audit_split': kind, 'work_id': None, 'work_identity_verified': False})
    output.mkdir(parents=True)
    (output/'images').mkdir()
    for split in PAGE_METADATA:
        shutil.copyfile(metadata_directory/'pages'/split/'metadata.jsonl', output/(split+'-source-metadata.jsonl'))
    settings = {'near_image_distance': 4, 'near_text_similarity': .95}
    write_json(output/'frozen-settings.json', {'dataset': DATASET, 'revision': REVISION,
        'metadata_sha256': hashes, 'holdout_sha256': digest(holdout), 'pilot_config_sha256': digest(pilot_config),
        'thresholds': settings, 'dhash_bits': 64, 'text_shingle_words': 7,
        'minimum_distinct_text_shingles': 12, 'max_aspect_ratio_delta': .1,
        'normalization': 'NFC and whitespace only; no casefold, modernization or character substitution'})
    config = {'dataset': {'repo': DATASET, 'revision': REVISION}}
    for index, row in enumerate(rows, 1):
        source = 'pages/'+row['split']+'/'+safe_relative(row['file_name'])
        target = output/'images'/(row['id']+'.jpg')
        safe_relative(target.relative_to(output).as_posix())
        if image_cache is not None:
            cached = Path(image_cache)/(row['id']+'.jpg')
            if cached.exists():
                if digest(cached) != row['image_sha256']:
                    raise ValueError('Cached image checksum mismatch')
                shutil.copyfile(cached, target)
        fetch(config, source, target, row['image_sha256'], opener=opener)
        with Image.open(target) as image:
            if image.format != 'JPEG' or image.size != (row['width'], row['height']):
                raise ValueError('Page format/dimensions mismatch')
        row['image'] = target.relative_to(output).as_posix()
        row['dhash'] = f'{dhash(target.read_bytes()):016x}'
        row['sha256'] = row['image_sha256']
        row['eligible_for_training'] = row['eligible_for_evaluation'] = False
        print(f'Page signatures {index}/{len(rows)}: {row["id"]}', flush=True)
    flags, comparisons = compare_pages(rows, **settings)
    write_rows(output/'page-signatures.jsonl', rows)
    write_rows(output/'overlap-review-queue.jsonl', flags)
    collections = []
    for collection in sorted({r['collection'] for r in rows}):
        group = [r for r in rows if r['collection'] == collection]
        collections.append({'collection': collection, 'pages': [r['id'] for r in group],
            'audit_splits': sorted({r['audit_split'] for r in group}),
            'digital_edition_ids': sorted({r['edition_id'] for r in group}),
            'source_xml_paths': [r['source_xml_path'] for r in group],
            'work_id': None, 'work_title': None, 'bibliographic_edition_id': None,
            'identity_status': 'requires-bibliographic-source-review'})
    write_rows(output/'work-identity-catalog.jsonl', collections)
    report = {'schema': 'slayer-recognizer-source-split-audit-v1', 'dataset': DATASET, 'revision': REVISION,
        'metadata_sha256': hashes, 'pages': len(rows), 'page_image_hashes_verified': len(rows),
        'splits': dict(Counter(r['audit_split'] for r in rows)), 'cross_split_comparisons': comparisons,
        'overlap_proposals': len(flags), 'proposal_reasons': dict(Counter(x for f in flags for x in f['reasons'])),
        'digital_edition_ids': len({r['edition_id'] for r in rows}),
        'source_document_ids': len({r['document_id'] for r in rows}),
        'work_identities_verified': 0, 'work_catalog_collections': len(collections),
        'training_freeze_ready': False, 'sota_claim': False,
        'external_final_test_corpora_audited': False,
        'claim_boundary': f'Bounded cross-split screening of {len(rows)} source pages. dHash/text matches are proposals, not proven duplicates. Digital edition IDs are not verified bibliographic work IDs.',
        'limitations': 'Source text is not adjudicated gold. dHash can miss crops/rotations and produce same-layout false positives. No external corpus or upstream pretraining audit. Work/edition grouping remains unresolved.'}
    write_json(output/'audit-report.json', report)
    write_json(output/'checksums.json', {p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata-directory', 'holdout', 'pilot-config', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--image-cache')
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))
