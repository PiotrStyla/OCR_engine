"""Import geometry/text reviews into traceable single-review training candidates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import zipfile

from PIL import Image

from training.adjudicate_reviews import DECISIONS, event_time, reviewer_key, text_hash
from training.build_annotation_review import issues
from training.full_page_pilot import digest, read_rows, safe_relative, write_json, write_rows
from training.prepare_historical_line_corpus import DATASET, REVISION, METADATA_SHA256

GEOMETRY = {'unreviewed', 'complete-line', 'reject-crop'}


def unique_rows(rows):
    if (not rows or any(not isinstance(r.get('id'), str) or not r['id'] for r in rows)
            or len({r['id'] for r in rows}) != len(rows)):
        raise ValueError('Expected unique nonempty source IDs')
    return {r['id']: r for r in rows}


def validate_events(packet, rows, contexts, manifest_hash):
    if (not isinstance(packet, dict) or packet.get('schema') != 'slayer-recognizer-line-review-v1'
            or packet.get('manifest_sha256') != manifest_hash
            or not isinstance(packet.get('events'), list)):
        raise ValueError('Review schema or manifest mismatch')
    sources = unique_rows(rows)
    texts = {r['id']: r['text'] for r in rows}
    latest, seen, times = {}, set(), {}
    for e in packet['events']:
        if not isinstance(e, dict):
            raise ValueError('Invalid event')
        line = sources.get(e.get('page_id'))
        if (line is None or not isinstance(e.get('id'), str) or not e['id'] or e['id'] in seen):
            raise ValueError('Unknown line or duplicate/invalid event ID')
        key = line['id']
        if (e.get('original_text_sha256') != text_hash(line['text'])
                or e.get('image_sha256') != line['sha256']
                or e.get('context_image_sha256') != contexts[key]['sha256']
                or e.get('before') != texts[key]):
            raise ValueError('Review source identity or text history mismatch')
        if (e.get('decision') not in DECISIONS or e.get('geometry_decision') not in GEOMETRY
                or not isinstance(e.get('after'), str) or len(e['after']) > 200000
                or not isinstance(e.get('reviewer'), str) or not reviewer_key(e['reviewer'])
                or len(e['reviewer']) > 100 or not isinstance(e.get('note'), str)
                or len(e['note']) > 2000):
            raise ValueError('Invalid event fields')
        if (e['decision'] == 'verified' and e['geometry_decision'] == 'unreviewed'
                or e['geometry_decision'] == 'reject-crop' and not e['note'].strip()):
            raise ValueError('Verified text needs geometry; rejection needs a reason')
        timestamp = event_time(e.get('timestamp'))
        if key in times and timestamp <= times[key]:
            raise ValueError('Line chronology must be strictly increasing; ties are ambiguous')
        times[key] = timestamp
        seen.add(e['id'])
        texts[key] = e['after']
        latest[key] = e
    if len({reviewer_key(e['reviewer']) for e in packet['events']}) > 1:
        raise ValueError('Single-review import requires one reviewer; use adjudication for multiple reviewers')
    return latest


def bound_file(root, name):
    path = root / safe_relative(name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Source path escapes its root')
    return path


def import_review(review_directory, review_json, config, metadata_directory,
                  observations, output, *, context_confirmations=None):
    review_directory, review_json, config, metadata_directory, observations, output = map(
        Path, (review_directory, review_json, config, metadata_directory, observations, output))
    if output.exists():
        raise FileExistsError('Use a new import directory')
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError('Archive already exists')
    manifest = review_directory / 'input/manifest.jsonl'
    sources = read_rows(manifest)
    by_id = unique_rows(sources)
    contexts = json.loads((review_directory/'contexts.json').read_text(encoding='utf-8'))
    diagnostics = json.loads((review_directory/'diagnostics.json').read_text(encoding='utf-8'))
    prep = json.loads((review_directory/'report.json').read_text(encoding='utf-8'))
    cfg = json.loads(config.read_text(encoding='utf-8'))
    sums = json.loads((review_directory/'checksums.json').read_text(encoding='utf-8'))
    required = {'input/manifest.jsonl', 'contexts.json', 'diagnostics.json', 'report.json'}
    required.update('input/'+safe_relative(r['image']) for r in sources)
    required.update('input/'+safe_relative(c['image']) for c in contexts.values())
    if not required <= set(sums) or any(digest(bound_file(review_directory, n)) != sums[n] for n in required):
        raise ValueError('Review source checksums mismatch')
    if (set(contexts) != set(by_id) or set(diagnostics) != set(by_id)
            or prep['review_manifest_sha256'] != digest(manifest)
            or prep['original_manifest_sha256'] != cfg['dataset']['manifest_sha256']):
        raise ValueError('Frozen review/config coverage mismatch')

    metadata = {}
    for split, expected in METADATA_SHA256.items():
        path = metadata_directory / split / 'metadata.jsonl'
        if digest(path) != expected:
            raise ValueError('Frozen region metadata mismatch')
        metadata[split] = unique_rows(read_rows(path))
    heldout = [r for split in ('validation', 'test') for r in metadata[split].values()]
    heldout_pages = {r['page_id'] for r in heldout}
    heldout_hashes = {r['image_sha256'] for r in heldout}
    forbidden = set(cfg['dataset']['forbidden_collections'])
    for row in sources:
        region = metadata['train'].get(row['source_region_id'])
        context = contexts[row['id']]
        if (region is None or row['source_split'] != 'train' or row['final_test'] is not False
                or row['eligible_for_training'] is not False or row['eligible_for_evaluation'] is not False
                or row['dataset'] != DATASET or row['revision'] != REVISION
                or region['split'] != 'train' or row['page_id'] != region['page_id']
                or row['collection'] != region['collection'] or row['collection'] in forbidden
                or row['page_id'] in heldout_pages or row['sha256'] in heldout_hashes
                or row['source_image_sha256'] in heldout_hashes
                or row['source_image_sha256'] != region['image_sha256']
                or context['source_region_id'] != region['id'] or context['sha256'] != region['image_sha256']
                or context['text'] != region['text'] or row['source_region_text'] != region['text']):
            raise ValueError('Train/heldout firewall or context identity mismatch')
        for item, expected_format in ((row, 'PNG'), (context, 'JPEG')):
            path = bound_file(manifest.parent, item['image'])
            if digest(path) != item['sha256']:
                raise ValueError('Image checksum is not bound to the manifest')
            with Image.open(path) as image:
                if image.format != expected_format or image.size != (item['width'], item['height']):
                    raise ValueError('Unexpected image format/dimensions')
    packet = json.loads(review_json.read_text(encoding='utf-8'))
    latest = validate_events(packet, sources, contexts, digest(manifest))
    observed = json.loads(observations.read_text(encoding='utf-8'))
    if observed.get('schema') != 'slayer-recognizer-data-pilot-agent-observations-v1':
        raise ValueError('Unexpected agent observation schema')
    conflicts = {}
    for item in observed['observations']:
        if item['kind'] == 'crop-reference-alignment-failure':
            if item['id'] not in by_id or item['image_sha256'] != by_id[item['id']]['sha256']:
                raise ValueError('Unbound crop conflict')
            conflicts[item['id']] = item
    confirmations = {}
    if context_confirmations is not None:
        confirmation_packet = json.loads(Path(context_confirmations).read_text(encoding='utf-8'))
        if confirmation_packet.get('schema') != 'slayer-line-context-scope-confirmation-v1':
            raise ValueError('Unexpected scope confirmation schema')
        confirmations = unique_rows(confirmation_packet['confirmations'])
        for key, c in confirmations.items():
            e = latest.get(key)
            context = contexts.get(key)
            if (key not in conflicts or e is None or e['decision'] != 'verified'
                    or e['geometry_decision'] != 'complete-line' or c.get('scope') != 'full-context-single-line'
                    or c.get('review_event_id') != e['id'] or c.get('crop_sha256') != by_id[key]['sha256']
                    or c.get('context_sha256') != context['sha256'] or c.get('text_sha256') != text_hash(e['after'])
                    or c.get('source_kind') != 'direct-user-message' or not c.get('question')
                    or not c.get('answer') or not c.get('resolved_answer')
                    or e['after'] != context['text'] or not e['after'].strip()
                    or '\n' in e['after'] or '\r' in e['after'] or issues(e['after'])):
                raise ValueError('Scope confirmation is not bound to a verified single-line context')

    results, candidates = [], []
    for row in sources:
        e = latest.get(row['id'])
        text = e['after'] if e else row['text']
        if e is None:
            status = 'unreviewed'
        elif e['geometry_decision'] == 'reject-crop':
            status = 'rejected-crop'
        elif e['decision'] != 'verified' or e['geometry_decision'] != 'complete-line':
            status = 'pending'
        elif row['id'] in conflicts:
            status = 'geometry-conflict'
        elif not text.strip() or issues(text) or '\n' in text or '\r' in text:
            status = 'text-quality-quarantine'
        else:
            status = 'single-review-candidate'
        results.append({**row, 'source_reference_text': row['text'], 'source_reference_text_sha256': text_hash(row['text']),
            'text': text, 'reviewed_text_sha256': text_hash(text), 'changed_from_source': text != row['text'],
            'import_status': status, 'review_status': e['decision'] if e else 'unreviewed',
            'geometry_decision': e['geometry_decision'] if e else 'unreviewed',
            'review_event_id': e['id'] if e else None, 'reviewer': e['reviewer'] if e else None,
            'annotation_note': e['note'] if e else '', 'unicode_issues': issues(text),
            'teacher_diagnostics': diagnostics[row['id']], 'eligible_for_training': status == 'single-review-candidate',
            'eligible_for_evaluation': False, 'gold': False, 'context': contexts[row['id']]})

    output.mkdir(parents=True)
    (output/'input/lines').mkdir(parents=True)
    (output/'input/regions').mkdir()
    for name in required:
        if name.startswith('input/') and name != 'input/manifest.jsonl':
            shutil.copyfile(review_directory/name, output/name)
    shutil.copyfile(manifest, output/'original-manifest.jsonl')
    shutil.copyfile(review_json, output/'original-review.json')
    shutil.copyfile(observations, output/'agent-observations.json')
    shutil.copyfile(config, output/'pilot-config.json')
    for name in ('contexts.json', 'diagnostics.json', 'report.json'):
        shutil.copyfile(review_directory/name, output/('source-'+name))
    if context_confirmations is not None:
        shutil.copyfile(context_confirmations, output/'context-scope-confirmation.json')
    for row in results:
        row['image'] = 'input/'+row['image']
        row['context'] = {**row['context'], 'image': 'input/'+row['context']['image']}
        if row['eligible_for_training']:
            candidates.append(dict(row))
    for key, c in confirmations.items():
        source = next(r for r in results if r['id'] == key)
        new_id = key+'__context-v1'
        relative = 'input/lines/'+new_id+'.png'
        safe_relative(relative)
        with Image.open(output/source['context']['image']) as image:
            image.save(output/relative, format='PNG')
            with Image.open(output/relative) as converted:
                if converted.mode != image.mode or converted.size != image.size or converted.tobytes() != image.tobytes():
                    raise ValueError('Context PNG conversion changed decoded pixels')
        candidates.append({**source, 'id': new_id, 'image': relative, 'sha256': digest(output/relative),
            'image_sha256': digest(output/relative), 'width': source['context']['width'], 'height': source['context']['height'],
            'import_status': 'single-review-context-replacement', 'eligible_for_training': True,
            'reference_status': 'single-human-reviewed-context', 'line_geometry_verified': True,
            'supersedes_crop_id': key, 'supersedes_crop_sha256': source['sha256'],
            'scope_confirmation': c, 'conversion': 'JPEG decoded pixels to PNG; no resize/crop/deskew'})
    for r in candidates:
        r['split'] = 'train-candidate'
        r['reference_status'] = 'single-human-reviewed-training-candidate'
        r['line_geometry_verified'] = True
        r['text_sha256'] = text_hash(r['text'])
    write_rows(output/'reviewed-lines.jsonl', results)
    write_rows(output/'training-candidates.jsonl', candidates)
    for status in ('pending', 'rejected-crop', 'geometry-conflict', 'text-quality-quarantine', 'unreviewed'):
        write_rows(output/(status+'.jsonl'), [r for r in results if r['import_status'] == status])
    report = {'schema': 'slayer-recognizer-line-review-import-v1', 'lines': len(sources),
        'events': len(packet['events']), 'reviewed_lines': len(latest),
        'review_manifest_sha256': digest(manifest), 'review_export_sha256': digest(review_json),
        'pilot_config_sha256': digest(config), 'observations_sha256': digest(observations),
        'context_confirmation_sha256': digest(context_confirmations) if context_confirmations else None,
        'metadata_sha256': {s: digest(metadata_directory/s/'metadata.jsonl') for s in METADATA_SHA256},
        'source_statuses': dict(Counter(r['import_status'] for r in results)),
        'training_candidates': len(candidates), 'context_replacements': len(confirmations),
        'changed_source_transcriptions': sum(r['changed_from_source'] for r in results),
        'candidate_pages': len({r['page_id'] for r in candidates}),
        'candidate_collections': len({r['collection'] for r in candidates}),
        'gold_labels_created': 0, 'model_training_performed': False, 'sota_claim': False,
        'policy': 'Latest verified text + complete-line; reject/pending/conflicts excluded. Explicit context scope confirmation creates a NEW pair, never overwrites a crop.',
        'normalization_applied': 'none; historical glyphs and spelling preserved verbatim',
        'firewall': 'Pinned train/validation/test metadata; forbidden collections, page IDs and exact image hashes checked.',
        'limitations': 'Single self-reported reviewer; not independent gold. Work/edition/near-duplicate audit remains outstanding. Small training-data pilot, not an OCR quality measurement.'}
    write_json(output/'import-report.json', report)
    write_json(output/'checksums.json', {p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()})
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                stream.write(p, p.relative_to(output).as_posix())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('review-directory', 'review-json', 'config', 'metadata-directory', 'observations', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--context-confirmations')
    args = parser.parse_args()
    print(json.dumps(import_review(**vars(args)), ensure_ascii=False, indent=2))
