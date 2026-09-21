"""Run after the reproducible baseline in the SAME Kaggle session.

Exports three fixed test-page diagnostics, not a tuning or training run.
The notebook version embeds this file and does not require a GitHub update.
"""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import html
import importlib.metadata
import json
import os
import zipfile


def serializable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if callable(value):
        return {'callable': value.__module__ + '.' + value.__qualname__}
    if hasattr(value, '__dict__'):
        return {'class': type(value).__module__ + '.' + type(value).__name__,
                'attributes': serializable(vars(value))}
    return {'type': type(value).__name__, 'value': str(value)}


def validate_pairing(lines, predictions):
    ids = [str(line.id) for line in lines]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate segmentation line IDs')
    if ids != [str(record.id) for record in predictions]:
        raise ValueError('Prediction/segmentation identity or order mismatch')


def select_pages(rows):
    selected = {}
    for row in sorted(rows, key=lambda r: r['id']):
        collection = row['id'].split('__')[0]
        selected.setdefault(collection, row)
    return list(selected.values())


def write_gallery(output, entries):
    body = ['<!doctype html><meta charset="utf-8"><title>Kraken diagnostics</title>',
            '<style>body{font:16px sans-serif;margin:24px}img{max-width:100%}',
            'pre{white-space:pre-wrap}section{border-bottom:1px solid #bbb;padding:16px 0}</style>',
            '<h1>Kraken line diagnostics</h1>',
            '<p>Version 2: actual recognizer crops, checked line IDs. Normalized previews are reconstructed from the returned crops using the active input configuration, not captured with a network hook. No ground truth is sent to the model.</p>']
    for entry in entries:
        body += [f'<h2>{html.escape(entry["id"])}</h2>',
                 f'<img src="{entry["directory"]}/overlay.jpg" alt="Detected lines in native order">']
        for line in entry['lines']:
            body += [f'<section><h3>Line {line["order"]}</h3>',
                     (f'<img src="{entry["directory"]}/{line["crop"]}" alt="Line crop">'
                      if line['crop'] else '<p>No image returned by recognizer for this line.</p>'),
                     f'<pre>{html.escape(line["text"])}</pre></section>']
            if line.get('normalized_preview'):
                body.append(f'<img src="{entry["directory"]}/{line["normalized_preview"]}" alt="Normalized input preview">')
    (output / 'index.html').write_text('\n'.join(body), encoding='utf-8')


def main():
    root = Path('/kaggle/working')
    candidates = sorted(root.glob('polocrbench-kraken-*/full/run.json'))
    candidates = [p for p in candidates if json.loads(p.read_text())['state'] == 'completed']
    if not candidates:
        raise RuntimeError('Open the original completed Kaggle session. No saved baseline found.')
    work = candidates[-1].parents[1]
    run = json.loads(candidates[-1].read_text())
    manifest = work / 'png/manifest.jsonl'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != run['manifest_sha256']:
        raise ValueError('Baseline manifest mismatch')
    rows = [json.loads(line) for line in manifest.read_text().split('\n') if line.strip()]
    pages = select_pages(rows)
    if len(pages) != 3:
        raise ValueError('Expected exactly three benchmark collections')
    assert importlib.metadata.version('kraken') == '7.1.1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    from huggingface_hub import hf_hub_download
    from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel
    from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
    from kraken.lib.dataset import ImageInputTransforms
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU')
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    models = {}
    for label in ('recognizer', 'segmenter'):
        path = hf_hub_download('PiotrSty/ehri-dataset', 'models/' + run[label]['filename'],
            repo_type='dataset', revision='1947d4c107936ed59ded16446a77b6587fa3874a')
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != run[label]['sha256']:
            raise ValueError('Model hash mismatch')
        models[label] = path
    seg_model = SegmentationTaskModel.load_model(models['segmenter'])
    rec_model = RecognitionTaskModel.load_model(models['recognizer'])
    sc = SegmentationInferenceConfig(accelerator='cuda', device=[0])
    rc = RecognitionInferenceConfig(accelerator='cuda', device=[0], return_line_image=True)
    output = root / ('kraken-input-check-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(exist_ok=False)
    entries = []
    for page_number, row in enumerate(pages):
        path = manifest.parent / row['image']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Page hash mismatch')
        with Image.open(path) as source:
            image = source.convert('L')
        folder = output / f'page-{page_number:02d}'
        folder.mkdir()
        with torch.inference_mode():
            segmentation = seg_model.predict(image, sc)
            predictions = list(rec_model.predict(image, segmentation, rc))
        validate_pairing(segmentation.lines, predictions)
        net = rec_model.net
        batch, channels, height, width = net.input
        transforms = ImageInputTransforms(batch, height, width, channels,
            (net._inf_config.padding, 0), segmentation.type != 'baselines', dtype=net._m_dtype)
        overlay = image.convert('RGB')
        draw = ImageDraw.Draw(overlay)
        lines = []
        for index, (line, prediction) in enumerate(zip(segmentation.lines, predictions)):
            name = f'line-{index:04d}.png'
            crop = prediction.image
            if crop is None:
                lines.append({'order': index, 'line_id': str(line.id), 'crop': '',
                              'text': prediction.prediction, 'status': 'no_image_returned'})
                continue
            crop.save(folder / name)
            tensor = transforms(crop).detach().float().cpu().numpy()
            if not np.isfinite(tensor).all():
                raise ValueError('Non-finite normalized input')
            np.save(folder / f'line-{index:04d}-input.npy', tensor, allow_pickle=False)
            plane = tensor[0]
            low, high = float(plane.min()), float(plane.max())
            normalized = ((plane-low)/(high-low)*255).astype('uint8') if high > low else np.zeros_like(plane, dtype='uint8')
            preview_name = f'line-{index:04d}-input.png'
            Image.fromarray(normalized).save(folder / preview_name)
            boundary = [tuple(point) for point in line.boundary]
            baseline = [tuple(point) for point in line.baseline]
            draw.line(boundary + boundary[:1], fill=(220, 30, 30), width=2)
            draw.line(baseline, fill=(0, 100, 230), width=3)
            draw.text(baseline[0], str(index), fill=(0, 0, 0), stroke_width=2, stroke_fill='white')
            lines.append({'order': index, 'line_id': str(line.id), 'boundary': boundary,
                          'baseline': baseline, 'crop': name, 'text': prediction.prediction,
                          'status': 'ok', 'normalized_preview': preview_name,
                          'tensor_shape': list(tensor.shape), 'tensor_min': float(tensor.min()),
                          'tensor_max': float(tensor.max()), 'tensor_std': float(tensor.std())})
        overlay.thumbnail((1600, 2200))
        overlay.save(folder / 'overlay.jpg', quality=92)
        entry = {'id': row['id'], 'image_sha256': row['sha256'], 'directory': folder.name, 'lines': lines}
        entries.append(entry)
        (folder / 'lines.json').write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding='utf-8')
        print(row['id'], 'lines:', len(lines), flush=True)
    write_gallery(output, entries)
    metadata = {'source_run': run, 'selection': 'First ID per collection, independent of scores',
                'purpose': 'Test error inspection only; no parameter tuning or training',
                'diagnostic_version': 2,
                'crop_scope': 'Actual prediction.image from recognizer; normalized tensors reconstructed before batch padding',
                'segmentation_config': serializable(sc), 'recognition_config': serializable(rc),
                'active_recognition_config': serializable(rec_model.net._inf_config),
                'model_input': serializable(rec_model.net.input),
                'codec': serializable(rec_model.net.codec),
                'use_legacy_polygons': bool(rec_model.net.use_legacy_polygons),
                'one_channel_mode': serializable(rec_model.one_channel_mode),
                'script_identifier': 'embedded-notebook-v2'}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    hashes = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'checksums.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    archive_path = output.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                archive.write(p, p.relative_to(output))
    print('Output:', archive_path)
    from IPython.display import HTML, display
    if archive_path.stat().st_size < 25_000_000:
        encoded = base64.b64encode(archive_path.read_bytes()).decode('ascii')
        display(HTML(f'<a download="{archive_path.name}" href="data:application/zip;base64,{encoded}">Download diagnostics ZIP</a>'))
    else:
        print('Download ZIP through the Kaggle file panel (too large for inline embedding).')


if __name__ == '__main__':
    main()
