"""Materialize crop/region context for human review, never approve training labels."""
import argparse
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

from PIL import Image

from training.build_annotation_review import build
from training.full_page_pilot import digest, fetch, read_rows, safe_relative, write_json, write_rows
from training.prepare_historical_line_corpus import METADATA_SHA256


def prepare(audit_directory, train_metadata, output, *, opener=urlopen):
    audit_directory, train_metadata, output = map(Path, (audit_directory, train_metadata, output))
    if output.exists():
        raise FileExistsError('Use a new line review directory')
    if digest(train_metadata) != METADATA_SHA256['train']:
        raise ValueError('Frozen training-region metadata mismatch')
    audit = json.loads((audit_directory/'audit.json').read_text(encoding='utf-8'))
    config = json.loads((audit_directory/'evidence/config.json').read_text(encoding='utf-8'))
    manifest = audit_directory/'dataset/manifest.jsonl'
    if (digest(manifest) != config['dataset']['manifest_sha256'] or audit.get('proposals_and_report_recomputed') is not True
            or digest(audit_directory/'recomputed/proposals.jsonl') != audit['recomputed_proposals_sha256']
            or digest(audit_directory/'recomputed/report.json') != audit['recomputed_report_sha256']):
        raise ValueError('Verified audit required')
    proposals = {r['id']: r for r in read_rows(audit_directory/'recomputed/proposals.jsonl')}
    rows = read_rows(manifest)
    sources = {r['id']: r for r in read_rows(train_metadata)}
    if set(proposals) != {r['id'] for r in rows}:
        raise ValueError('Teacher/crop coverage mismatch')
    for row in rows:
        region = sources[row['source_region_id']]
        if (row['source_split'] != 'train' or row['eligible_for_training'] is not False
                or row['collection'] in config['dataset']['forbidden_collections']
                or region['split'] != 'train' or region['page_id'] != row['page_id']
                or region['collection'] != row['collection']
                or region['image_sha256'] != row['source_image_sha256']
                or digest(audit_directory/'dataset'/row['image']) != row['sha256']):
            raise ValueError('Training-source crop/context identity mismatch')
    output.mkdir(parents=True)
    inputs = output/'input'
    (inputs/'lines').mkdir(parents=True)
    (inputs/'regions').mkdir()
    review_rows, contexts, diagnostics = [], {}, {}
    # All 64 remain in review; ordering does not exclude inconvenient outputs.
    rows.sort(key=lambda r: (r['id'] != 'NA1_FT__433925__r011__line000',
        proposals[r['id']]['status'] != 'teacher-agreement-proposal',
        proposals[r['id']]['status'] != 'teacher-abstention', r['collection'], r['page_id'], r['id']))
    for index, row in enumerate(rows, 1):
        region = sources[row['source_region_id']]
        original_path = safe_relative(region['file_name'])
        if not original_path.startswith('images/'):
            raise ValueError('Unexpected region path')
        relative = 'regions/'+region['id']+'.jpg'
        fetch({'dataset': {'repo': row['dataset'], 'revision': row['revision']}},
              'regions/train/'+original_path, inputs/relative, region['image_sha256'], opener=opener)
        with Image.open(inputs/relative) as image:
            width, height = image.size
            if image.format != 'JPEG':
                raise ValueError('Expected unchanged source JPEG region')
        contexts[row['id']] = {'image': relative, 'sha256': region['image_sha256'],
            'width': width, 'height': height, 'text': region['text'], 'source_region_id': region['id']}
        line_path = 'lines/'+row['id']+'.png'
        shutil.copyfile(audit_directory/'dataset'/row['image'], inputs/line_path)
        review_rows.append({**row, 'image': line_path})
        vote = proposals[row['id']]
        diagnostics[row['id']] = {'items': [], 'candidate_label': 'Qwen3-VL / surowy odczyt',
            'candidate_text': vote['teachers']['qwen3-vl-4b']['text'],
            'baseline_label': 'TrOCR mixed-v3 / surowy odczyt',
            'baseline_text': vote['teachers']['trocr-mixed-v3']['text'],
            'review_status_label': {'teacher-agreement-proposal': 'Zgodność modeli, nie gold',
                'teacher-disagreement': 'Rozbieżność modeli', 'teacher-abstention': 'Abstencja / limit'}[vote['status']]}
        print(f'Context {index}/{len(rows)}: {row["id"]}', flush=True)
    write_rows(inputs/'manifest.jsonl', review_rows)
    write_json(output/'contexts.json', contexts)
    write_json(output/'diagnostics.json', diagnostics)
    summary = build(inputs/'manifest.jsonl', output/'review', diagnostics=diagnostics,
                    contexts=contexts, geometry_review=True)
    report = {'schema': 'slayer-recognizer-line-review-preparation-v1', 'lines': len(rows),
        'source_regions': len({r['source_region_id'] for r in rows}),
        'source_archive_sha256': audit['source_archive_sha256'], 'original_manifest_sha256': digest(manifest),
        'review_manifest_sha256': digest(inputs/'manifest.jsonl'), 'review_summary': summary,
        'review_patch_schema': 'slayer-recognizer-line-review-v1',
        'geometry_decisions': ['unreviewed', 'complete-line', 'reject-crop'],
        'original_line_labels_changed': False, 'training_examples_created': 0, 'gold_labels_created': 0,
        'claim_boundary': 'Human review required for both geometry and diplomatic text. Complete-line is separate from text verification. No automatic training promotion.'}
    write_json(output/'report.json', report)
    files = [p for p in output.rglob('*') if p.is_file()]
    write_json(output/'checksums.json', {p.relative_to(output).as_posix(): digest(p) for p in files})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-directory', 'train-metadata', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.audit_directory, args.train_metadata, args.output), ensure_ascii=False, indent=2))
