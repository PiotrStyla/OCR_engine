"""Prepare only unresolved line cases; recrops are proposals, never approvals."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import zipfile

from PIL import Image

from training.adjudicate_reviews import text_hash
from training.build_annotation_review import build
from training.full_page_pilot import digest, read_rows, safe_relative, write_json, write_rows


def prepare(source_directory, recipes, output):
    source_directory, recipes, output = map(Path, (source_directory, recipes, output))
    if output.exists() or output.with_suffix('.zip').exists():
        raise FileExistsError('Use a fresh remediation directory')
    checksums = json.loads((source_directory/'checksums.json').read_text(encoding='utf-8'))
    required = {'reviewed-lines.jsonl', 'import-report.json', 'original-review.json', 'source-report.json', 'pilot-config.json'}
    if not required <= set(checksums):
        raise ValueError('Parent checksum coverage mismatch')
    for name, sha in checksums.items():
        path = source_directory/safe_relative(name)
        if not path.resolve().is_relative_to(source_directory.resolve()) or digest(path) != sha:
            raise ValueError('Parent import checksums mismatch')
    parent_report = json.loads((source_directory/'import-report.json').read_text(encoding='utf-8'))
    rows = read_rows(source_directory/'reviewed-lines.jsonl')
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Unique parent line IDs required')
    plan = json.loads(recipes.read_text(encoding='utf-8'))
    if (plan.get('schema') != 'slayer-recognizer-recrop-proposals-v1'
            or plan.get('parent_review_sha256') != parent_report['review_export_sha256']
            or plan.get('actor_kind') != 'AI-assistant' or plan.get('human_review_required') is not True):
        raise ValueError('Recrop proposal provenance mismatch')
    rejected = {r['id']: r for r in rows if r['import_status'] == 'rejected-crop'}
    proposals = plan['proposals']
    if len({p['id'] for p in proposals}) != len(proposals) or {p['id'] for p in proposals} != set(rejected):
        raise ValueError('Every rejected crop needs exactly one bound proposal')
    for p in proposals:
        row = rejected[p['id']]
        box = p.get('bbox')
        context = row['context']
        if (p.get('crop_sha256') != row['sha256'] or p.get('context_sha256') != context['sha256']
                or not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box)
                or not 0 <= box[0] < box[2] <= context['width']
                or not 0 <= box[1] < box[3] <= context['height']):
            raise ValueError('Invalid or unbound recrop box')
    selected = [r for r in rows if r['import_status'] in ('pending', 'rejected-crop')]
    if not selected:
        raise ValueError('No unresolved cases')
    for row in selected:
        if (not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']) or row['source_split'] != 'train'
                or row['eligible_for_training'] is not False or row['eligible_for_evaluation'] is not False):
            raise ValueError('Expected excluded training-source cases')
    output.mkdir(parents=True)
    (output/'input/lines').mkdir(parents=True)
    (output/'input/regions').mkdir()
    proposal_map = {p['id']: p for p in proposals}
    records, contexts, diagnostics = [], {}, {}
    for row in selected:
        context = row['context']
        context_relative = 'regions/'+row['source_region_id']+'.jpg'
        shutil.copyfile(source_directory/context['image'], output/'input'/context_relative)
        kind = 'recrop' if row['id'] in proposal_map else 'text-confirmation'
        identifier = row['id']+('__recrop-v1' if kind == 'recrop' else '__text-review-v1')
        image_relative = 'lines/'+identifier+'.png'
        if kind == 'recrop':
            box = proposal_map[row['id']]['bbox']
            with Image.open(source_directory/context['image']) as image:
                image.crop(tuple(box)).save(output/'input'/image_relative, format='PNG')
            transformation = {'kind': 'rectangle-proposal-from-frozen-context', 'bbox_in_region': box,
                'actor_kind': 'AI-assistant', 'human_geometry_review_required': True,
                'resize': False, 'pixel_cleanup': False}
        else:
            shutil.copyfile(source_directory/row['image'], output/'input'/image_relative)
            transformation = {'kind': 'unchanged-image-text-confirmation', 'human_text_review_required': True}
        with Image.open(output/'input'/image_relative) as image:
            width, height = image.size
        sha = digest(output/'input'/image_relative)
        record = {**row, 'id': identifier, 'image': image_relative, 'sha256': sha, 'image_sha256': sha,
            'width': width, 'height': height, 'text_sha256': text_hash(row['text']),
            'source_reference_text': row['source_reference_text'], 'parent_line_id': row['id'],
            'parent_weak_reference_text': row['source_reference_text'],
            'parent_crop_sha256': row['sha256'], 'parent_review_event_id': row['review_event_id'],
            'parent_review_status': row['review_status'], 'parent_geometry_decision': row['geometry_decision'],
            'parent_annotation_note': row['annotation_note'], 'parent_import_status': row['import_status'],
            'reference_status': 'unverified-followup-proposal', 'review_status': 'unreviewed',
            'geometry_decision': 'unreviewed', 'review_event_id': None, 'reviewer': None,
            'line_geometry_verified': False, 'eligible_for_training': False, 'eligible_for_evaluation': False,
            'gold': False, 'import_status': 'followup-review-required', 'annotation_note': '',
            'split': 'training-review', 'transformation': transformation,
            'parent_review_export_sha256': parent_report['review_export_sha256']}
        record.pop('context', None)
        records.append(record)
        contexts[identifier] = {**context, 'image': context_relative}
        old = row['teacher_diagnostics']
        diagnostics[identifier] = {**old, 'items': [],
            'candidate_label': 'Qwen / odczyt poprzedniego wycinka',
            'baseline_label': 'TrOCR / odczyt poprzedniego wycinka',
            'review_status_label': 'Nowy wycinek: do sprawdzenia' if kind == 'recrop' else 'Propozycja tekstu: do zatwierdzenia'}
    write_rows(output/'input/manifest.jsonl', records)
    write_json(output/'contexts.json', contexts)
    write_json(output/'diagnostics.json', diagnostics)
    shutil.copyfile(source_directory/'original-review.json', output/'prior-original-review.json')
    shutil.copyfile(source_directory/'import-report.json', output/'prior-import-report.json')
    shutil.copyfile(recipes, output/'recrop-proposals.json')
    shutil.copyfile(source_directory/'pilot-config.json', output/'pilot-config.json')
    write_json(output/'agent-observations.json', {'schema': 'slayer-recognizer-data-pilot-agent-observations-v1',
        'actor_kind': 'AI-assistant', 'observations': [], 'human_review_events_created': 0})
    prep = json.loads((source_directory/'source-report.json').read_text(encoding='utf-8'))
    summary = build(output/'input/manifest.jsonl', output/'review', contexts=contexts,
                    diagnostics=diagnostics, geometry_review=True)
    report = {'schema': 'slayer-recognizer-line-review-preparation-v1', 'lines': len(records),
        'case_types': dict(Counter(r['transformation']['kind'] for r in records)),
        'source_regions': len({r['source_region_id'] for r in records}),
        'review_manifest_sha256': digest(output/'input/manifest.jsonl'),
        'original_manifest_sha256': prep['original_manifest_sha256'],
        'source_archive_sha256': prep['source_archive_sha256'], 'parent_import_report_sha256': digest(source_directory/'import-report.json'),
        'parent_review_export_sha256': parent_report['review_export_sha256'], 'recrop_proposals_sha256': digest(recipes),
        'human_review_required': True, 'training_examples_created': 0, 'gold_labels_created': 0,
        'prior_accepted_lines_repeated': 0, 'review_summary': summary,
        'claim_boundary': f'{len(records)} unresolved cases only. New crop hashes require fresh geometry/text review; old verified status is not transferred. Teacher outputs belong to prior crops.'}
    write_json(output/'report.json', report)
    files = [p for p in output.rglob('*') if p.is_file() and 'review' not in p.relative_to(output).parts]
    write_json(output/'checksums.json', {p.relative_to(output).as_posix(): digest(p) for p in files})
    with zipfile.ZipFile(output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as stream:
        for path in files + [output/'checksums.json']:
            stream.write(path, path.relative_to(output).as_posix())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-directory', 'recipes', 'output'):
        parser.add_argument('--'+name, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))
