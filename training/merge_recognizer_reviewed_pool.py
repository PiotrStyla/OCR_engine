"""Merge reviewed follow-ups without overwriting prior human evidence."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from PIL import Image

from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.import_recognizer_line_review import bound_file, import_review, unique_rows


def verified_package(root):
    root = Path(root)
    sums = json.loads((root/'checksums.json').read_text(encoding='utf-8'))
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if set(sums) != actual - {'checksums.json'}:
        raise ValueError('Package checksum coverage mismatch')
    for name, expected in sums.items():
        if digest(bound_file(root, name)) != expected:
            raise ValueError('Package checksum mismatch: '+name)
    return sums


def replay_package(root, scratch, config, metadata_directory):
    """Rebuild imports from raw decisions, including train/heldout checks."""
    verified_package(root)
    source = scratch/'source'
    source.mkdir(parents=True)
    shutil.copytree(root/'input', source/'input')
    shutil.copyfile(root/'original-manifest.jsonl', source/'input/manifest.jsonl')
    for name in ('contexts.json', 'diagnostics.json', 'report.json'):
        shutil.copyfile(root/('source-'+name), source/name)
    write_json(source/'checksums.json', {p.relative_to(source).as_posix(): digest(p)
        for p in source.rglob('*') if p.is_file()})
    if digest(root/'pilot-config.json') != digest(config):
        raise ValueError('Frozen pilot config mismatch')
    confirmation = root/'context-scope-confirmation.json'
    report = import_review(source, root/'original-review.json', config, metadata_directory,
        root/'agent-observations.json', scratch/'replayed',
        context_confirmations=confirmation if confirmation.exists() else None)
    original = json.loads((root/'import-report.json').read_text(encoding='utf-8'))
    if report != original:
        raise ValueError('Import report does not reproduce')
    for name in ('reviewed-lines.jsonl', 'training-candidates.jsonl'):
        if read_rows(root/name) != read_rows(scratch/'replayed'/name):
            raise ValueError('Imported rows do not reproduce: '+name)
    return report


def root_id(row):
    return row.get('parent_line_id') or row.get('supersedes_crop_id') or row['id']


def validate_followups(base, followup, ledger, review_sha):
    unique_rows(base)
    unique_rows(followup)
    parents = unique_rows(ledger)
    occupied = {root_id(r) for r in base}
    if len(occupied) != len(base):
        raise ValueError('Duplicate active root')
    for row in followup:
        key = row.get('parent_line_id')
        parent = parents.get(key)
        if parent is None or key in occupied or parent['import_status'] not in ('pending', 'rejected-crop'):
            raise ValueError('Follow-up must resolve an inactive pending/rejected parent')
        occupied.add(key)
        bindings = {'parent_crop_sha256': parent['sha256'],
            'parent_review_event_id': parent['review_event_id'],
            'parent_review_export_sha256': review_sha,
            'parent_import_status': parent['import_status'],
            'parent_weak_reference_text': parent['source_reference_text']}
        if any(row.get(k) != v for k, v in bindings.items()):
            raise ValueError('Follow-up parent evidence mismatch')
        for field in ('page_id', 'collection', 'source_region_id', 'source_image_sha256', 'dataset', 'revision'):
            if row[field] != parent[field]:
                raise ValueError('Follow-up source identity mismatch')
        if row['context'] != parent['context']:
            raise ValueError('Follow-up context identity mismatch')
        kind = row['transformation']['kind']
        if kind == 'unchanged-image-text-confirmation':
            if row['sha256'] != parent['sha256'] or parent['import_status'] != 'pending':
                raise ValueError('Text confirmation must retain the original image')
        elif kind == 'rectangle-proposal-from-frozen-context':
            if row['sha256'] == parent['sha256'] or parent['import_status'] != 'rejected-crop':
                raise ValueError('Recrop must create a new image from a rejected parent')
        else:
            raise ValueError('Unknown follow-up transformation')
    combined = base + followup
    if len({r['id'] for r in combined}) != len(combined) or len({r['sha256'] for r in combined}) != len(combined):
        raise ValueError('Duplicate active ID or image')


def verify_recrop(root, row):
    if row['transformation']['kind'] != 'rectangle-proposal-from-frozen-context':
        return
    bbox = row['transformation'].get('bbox_in_region')
    with Image.open(bound_file(root, row['context']['image'])) as context:
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(type(v) is not int for v in bbox)
                or not (0 <= bbox[0] < bbox[2] <= context.width and 0 <= bbox[1] < bbox[3] <= context.height)):
            raise ValueError('Invalid recrop rectangle')
        expected = context.crop(tuple(bbox))
        with Image.open(bound_file(root, row['image'])) as actual:
            if actual.mode != expected.mode or actual.size != expected.size or actual.tobytes() != expected.tobytes():
                raise ValueError('Recrop pixels differ from the frozen context')


def merge(base_directory, followup_directory, config, metadata_directory, output, *, audit_directory=None):
    base_directory, followup_directory, config, metadata_directory, output = map(
        Path, (base_directory, followup_directory, config, metadata_directory, output))
    archive = output.with_suffix('.zip')
    if output.exists() or archive.exists():
        raise FileExistsError('Use a new pool directory and archive')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='review-pool-', dir=output.parent) as temporary:
        scratch = Path(temporary)
        reports = [replay_package(root, scratch/name, config, metadata_directory)
            for name, root in (('base', base_directory), ('followup', followup_directory))]
        if reports[0]['metadata_sha256'] != reports[1]['metadata_sha256']:
            raise ValueError('Frozen metadata mismatch')
        base = read_rows(base_directory/'training-candidates.jsonl')
        followup = read_rows(followup_directory/'training-candidates.jsonl')
        ledger = read_rows(base_directory/'reviewed-lines.jsonl')
        validate_followups(base, followup, ledger, reports[0]['review_export_sha256'])
        for row in followup:
            verify_recrop(followup_directory, row)
        if audit_directory is not None:
            audit_directory = Path(audit_directory)
            verified_package(audit_directory)
        stage = scratch/'pool'
        (stage/'images').mkdir(parents=True)
        (stage/'contexts').mkdir()

        def materialize(row, root, label):
            result = deepcopy(row)
            for item, folder, suffix in ((result, 'images', '.png'), (result['context'], 'contexts', '.jpg')):
                source = bound_file(root, item['image'])
                if digest(source) != item['sha256']:
                    raise ValueError('Asset checksum mismatch')
                relative = folder+'/'+item['sha256']+suffix
                target = stage/relative
                if not target.exists():
                    shutil.copyfile(source, target)
                item['image'] = relative
            result['root_line_id'] = root_id(row)
            result['review_source'] = label
            return result

        active, historical = [], []
        for label, root, rows in (('base', base_directory, base), ('followup', followup_directory, followup)):
            history = stage/'history'/label
            history.mkdir(parents=True)
            for p in root.iterdir():
                if p.is_file():
                    shutil.copyfile(p, history/p.name)
            historical.extend(materialize(r, root, label) for r in read_rows(root/'reviewed-lines.jsonl'))
            active.extend(materialize(r, root, label) for r in rows)
        # Raw per-import manifests remain byte-for-byte in history; this ledger uses portable asset paths.
        write_rows(stage/'review-history.jsonl', historical)
        active.sort(key=lambda r: r['root_line_id'])
        write_rows(stage/'manifest.jsonl', active)
        write_rows(stage/'training-candidates.jsonl', active)
        active_pairs = {(r['root_line_id'], r['sha256']) for r in active}
        inactive = [r for r in historical if (r['root_line_id'], r['sha256']) not in active_pairs
            or r['eligible_for_training'] is not True]
        write_rows(stage/'inactive-history.jsonl', inactive)
        if audit_directory is not None:
            (stage/'source-audit').mkdir()
            for name in ('audit-report.json', 'frozen-settings.json', 'work-identity-catalog.jsonl',
                         'overlap-review-queue.jsonl', 'checksums.json'):
                shutil.copyfile(audit_directory/name, stage/'source-audit'/name)
        report = {'schema': 'slayer-recognizer-reviewed-pool-v1',
            'training_candidates': len(active), 'unique_root_lines': len({r['root_line_id'] for r in active}),
            'base_candidates': len(base), 'followup_candidates': len(followup),
            'historical_ledger_rows': len(historical), 'inactive_historical_rows': len(inactive),
            'crop_images': len(list((stage/'images').glob('*.png'))),
            'context_images': len(list((stage/'contexts').glob('*.jpg'))),
            'pages': len({r['page_id'] for r in active}), 'collections': len({r['collection'] for r in active}),
            'manifest_sha256': digest(stage/'manifest.jsonl'),
            'pilot_config_sha256': digest(config), 'metadata_sha256': reports[0]['metadata_sha256'],
            'source_import_report_sha256': {label: digest(root/'import-report.json')
                for label, root in (('base', base_directory), ('followup', followup_directory))},
            'review_export_sha256': {label: digest(root/'original-review.json')
                for label, root in (('base', base_directory), ('followup', followup_directory))},
            'source_audit_report_sha256': digest(audit_directory/'audit-report.json') if audit_directory else None,
            'imports_replayed': True, 'gold_labels_created': 0, 'model_training_performed': False,
            'training_freeze_ready': False, 'eligible_for_evaluation': False, 'sota_claim': False,
            'normalization_applied': 'none; spelling and historical glyphs retained verbatim',
            'limitations': 'Single-human-reviewed training pilot, not independent gold or measured OCR quality. '
                'Work/edition grouping, larger clean training pool and independent dev freeze remain required. '
                'Recrops reproduce rectangles, not a claim of perfect margin geometry. '
                'Source-audit checksums describe the separate published audit; audit scans are not duplicated here.'}
        write_json(stage/'pool-report.json', report)
        write_json(stage/'checksums.json', {p.relative_to(stage).as_posix(): digest(p)
            for p in sorted(stage.rglob('*')) if p.is_file()})
        stage.rename(output)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                stream.write(p, p.relative_to(output).as_posix())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-directory', 'followup-directory', 'config', 'metadata-directory', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--audit-directory')
    print(json.dumps(merge(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
