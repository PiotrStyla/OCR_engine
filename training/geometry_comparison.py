"""Pair frozen original inputs and experimental geometry without dropping IDs."""
import io
import json
import zipfile


def pair_inputs(original_data, original_hash, auto_data, auto_hash, loader):
    original, images_a, _ = loader(original_data, original_hash)
    modified, images_b, provenance = loader(auto_data, auto_hash)
    if [r['id'] for r in original] != [r['id'] for r in modified]:
        raise ValueError('Original/automatic IDs differ')
    for a, b, old_image, new_image in zip(original, modified, images_a, images_b):
        for field in ['text', 'original_text', 'source_review_decision', 'collection', 'page_id']:
            if a[field] != b[field]:
                raise ValueError('Reference or sample metadata changed: ' + field)
        if b['geometry_status'] == 'fallback-original' and old_image != new_image:
            raise ValueError('Fallback image differs from original')
    rows = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for variant, records, images in [('original', original, images_a), ('automatic_with_fallback', modified, images_b)]:
            for i, (r, image) in enumerate(zip(records, images)):
                path = f'{variant}/{i:04d}.png'
                info = zipfile.ZipInfo(path)
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, image)
                rows.append({**r, 'id': r['id'] + '__' + variant, 'comparison_id': r['id'],
                             'geometry_variant': variant, 'image': path,
                             'comparison_geometry_status': modified[i]['geometry_status']})
        z.writestr(zipfile.ZipInfo('manifest.json'), json.dumps(rows, ensure_ascii=False))
        z.writestr(zipfile.ZipInfo('provenance.json'), json.dumps({
            **provenance, 'experiment': 'paired image-only geometry development control',
            'source_archive_hashes': [original_hash, auto_hash],
            'reference_changed': False, 'metrics_scope': 'Report each 63-line arm separately; no pooled score.'}))
    return buffer.getvalue()


def paired_metrics(rows, predictions, base_metrics):
    if [r['id'] for r in rows] != [p['id'] for p in predictions]:
        raise ValueError('Prediction/reference ID mismatch')
    result = {}
    for variant in ['original', 'automatic_with_fallback']:
        indices = [i for i, r in enumerate(rows) if r['geometry_variant'] == variant]
        result[variant] = base_metrics([rows[i] for i in indices], [predictions[i] for i in indices])
        changed = [i for i in indices if rows[i]['comparison_geometry_status'] == 'auto-proposal']
        result[variant]['changed_geometry_only'] = base_metrics(
            [rows[i] for i in changed], [predictions[i] for i in changed])['all_draft_lines']
    return result
