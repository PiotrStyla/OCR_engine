"""Collection-disjoint region OCR diagnostic for two image-only geometries."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import base64
import gc
import hashlib
import importlib.metadata
import io
import json
import unicodedata
from urllib.request import urlopen
import zipfile

from training.body_line_geometry import crop_line_band, detect_lines
from training.kaggle_printed_dev_control import DATASET, REVISION, MODELS, FROZEN

VARIANTS = ('rectangle', 'line_band')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def load_manifest(data, expected_sha256):
    if digest(data) != expected_sha256:
        raise ValueError('Manifest checksum mismatch')
    manifest = json.loads(data)
    if manifest['scope'] != 'geometry holdout diagnostic; not benchmark or model holdout':
        raise ValueError('Unexpected scope')
    if manifest['dataset'] != DATASET or manifest['revision'] != REVISION:
        raise ValueError('Dataset provenance mismatch')
    rows = manifest['regions']
    if len(rows) != 12 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Expected 12 unique regions')
    if len({r['collection'] for r in rows}) != len(rows):
        raise ValueError('Expected one region per collection')
    for row in rows:
        path = PurePosixPath(row['source_path'])
        if path.is_absolute() or '..' in path.parts or '\\' in row['source_path'] or ':' in row['source_path']:
            raise ValueError('Unsafe source path')
        if row['split'] != 'train' or row['collection'] in FROZEN:
            raise ValueError('Invalid holdout split')
        if row['eligible_for_benchmark'] is not False or '\ufffd' in row['text']:
            raise ValueError('Invalid reference status')
    return manifest


def fetch_sources(rows, opener=urlopen):
    images = []
    root = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/'
    for row in rows:
        with opener(root + row['source_path'], timeout=120) as response:
            content = response.read()
        if digest(content) != row['image_sha256']:
            raise ValueError('Source checksum mismatch: ' + row['id'])
        images.append(content)
    return images


def segment(content, variant):
    from PIL import Image
    if variant not in VARIANTS:
        raise ValueError('Unknown geometry variant')
    with Image.open(io.BytesIO(content)) as source:
        source = source.convert('RGB')
        result = detect_lines(source, follow_lines=variant == 'line_band')
        crops = []
        for index, box in enumerate(result['boxes']):
            crop = (crop_line_band(source, box, result['line_bands'][index])
                    if variant == 'line_band' else source.crop(box))
            crops.append(crop)
    return result, crops


def score(rows, predictions):
    from jiwer import cer, wer, process_characters
    if [r['id'] for r in rows] != [p['id'] for p in predictions]:
        raise ValueError('Prediction/reference IDs differ')
    result = {}
    subsets = {'all_regions': list(range(len(rows))),
               'without_private_use_references': [i for i, r in enumerate(rows)
                                                   if r['reference_private_use_count'] == 0]}
    for label, indices in subsets.items():
        refs = [normalize(rows[i]['text']) for i in indices]
        hyps = [normalize(predictions[i]['text']) for i in indices]
        result[label] = {'regions': len(indices), 'reference_characters': sum(len(x) for x in refs),
                         'cer': cer(refs, hyps), 'wer': wer(refs, hyps),
                         'lowercase_cer_diagnostic': cer([x.lower() for x in refs], [x.lower() for x in hyps]),
                         'errors': sum(predictions[i]['status'] != 'ok' for i in indices),
                         'empty': sum(not x for x in hyps),
                         'detected_lines': sum(predictions[i]['detected_lines'] for i in indices)}
    edits = []
    for row, pred in zip(rows, predictions):
        value = process_characters(normalize(row['text']), normalize(pred['text']))
        edits.append(value.substitutions + value.deletions + value.insertions)
    result['per_region_character_edits'] = dict(zip([r['id'] for r in rows], edits))
    return result


def compare_variants(rows, rectangle, line_band):
    a = score(rows, rectangle)['per_region_character_edits']
    b = score(rows, line_band)['per_region_character_edits']
    deltas = {row['id']: b[row['id']] - a[row['id']] for row in rows}
    return {'improved': sum(x < 0 for x in deltas.values()),
            'regressed': sum(x > 0 for x in deltas.values()),
            'tied': sum(x == 0 for x in deltas.values()),
            'net_character_edit_change': sum(deltas.values()), 'deltas': deltas}


def run(manifest_data, expected_sha256, runner_sha256=None):
    manifest = load_manifest(manifest_data, expected_sha256)
    rows = manifest['regions']
    image_bytes = fetch_sources(rows)
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    if not torch.cuda.is_available():
        raise RuntimeError('Select GPU runtime in Colab, then Run all')
    torch.manual_seed(0)
    output = Path('/content') / ('geometry-holdout-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    save('input-manifest.json', manifest)
    provenance = {'source_revision': REVISION, 'manifest_sha256': expected_sha256,
                  'runner_sha256': runner_sha256, 'reference_text_sent_to_model': False,
                  'selection_frozen_before_ocr': True, 'all_regions_retained': True,
                  'segmentation_uses_reference_line_count': False,
                  'scope': manifest['scope'], 'limitations': manifest['limitations']}
    save('provenance.json', provenance)
    report = {'scope': manifest['scope'], 'models': MODELS, 'gpu': torch.cuda.get_device_name(0),
              'environment': {p: importlib.metadata.version(p) for p in
                              ['torch', 'transformers', 'Pillow', 'jiwer', 'huggingface_hub',
                               'opencv-python-headless']},
              'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 256, 'dtype': 'float32'},
              'normalization': manifest['normalization'], 'results': {}}
    for model_id, revision in MODELS.items():
        processor = model = None
        by_variant = {variant: [] for variant in VARIANTS}
        try:
            processor = TrOCRProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=False)
            model = VisionEncoderDecoderModel.from_pretrained(
                model_id, revision=revision, trust_remote_code=False).float().cuda().eval()
            eos = model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            for variant in VARIANTS:
                for row, content in zip(rows, image_bytes):
                    prediction = {'id': row['id'], 'variant': variant, 'text': '', 'status': 'ok',
                                  'detected_lines': 0, 'lines': []}
                    try:
                        geometry, crops = segment(content, variant)
                        prediction['detected_lines'] = len(crops)
                        prediction['geometry'] = {'boxes': geometry['boxes'],
                                                  'foreign_ink_fraction': geometry['foreign_ink_fraction']}
                        if not crops:
                            raise ValueError('No lines detected')
                        texts = []
                        for index, crop in enumerate(crops):
                            pixels = processor(images=crop, return_tensors='pt').pixel_values.cuda()
                            with torch.inference_mode():
                                ids = model.generate(pixels, do_sample=False, num_beams=1,
                                                     max_new_tokens=256)[0].tolist()
                            text = processor.batch_decode([ids], skip_special_tokens=True)[0]
                            texts.append(text)
                            prediction['lines'].append({'index': index, 'text': text, 'token_ids': ids,
                                                        'ended_with_eos': ids[-1] in eos,
                                                        'possibly_truncated': len(ids) >= 257 and ids[-1] not in eos})
                        prediction['text'] = '\n'.join(texts)
                    except Exception as exc:
                        prediction['status'] = 'error'
                        prediction['error'] = type(exc).__name__ + ': ' + str(exc)
                    by_variant[variant].append(prediction)
        except Exception as exc:
            for variant in VARIANTS:
                by_variant[variant] = [{'id': r['id'], 'variant': variant, 'text': '', 'status': 'error',
                                        'detected_lines': 0, 'lines': [],
                                        'error': type(exc).__name__ + ': ' + str(exc)} for r in rows]
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        save(model_id.replace('/', '--') + '.json', by_variant)
        report['results'][model_id] = {variant: score(rows, by_variant[variant]) for variant in VARIANTS}
        report['results'][model_id]['comparison'] = compare_variants(
            rows, by_variant['rectangle'], by_variant['line_band'])
        save('report.json', report)
        print(model_id, json.dumps(report['results'][model_id], indent=2))
    save('checksums.json', {p.name: digest(p.read_bytes()) for p in output.iterdir() if p.is_file()})
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for path in output.iterdir():
            bundle.write(path, path.name)
    from IPython.display import HTML, display
    encoded = base64.b64encode(archive.read_bytes()).decode('ascii')
    display(HTML('<a download="' + archive.name + '" href="data:application/zip;base64,' + encoded +
                 '">Pobierz ZIP wynikow</a>'))
    print('Output:', archive)
    return archive
