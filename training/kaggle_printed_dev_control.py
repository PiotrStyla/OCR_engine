"""Small historical-region diagnostic, not a held-out or full-page benchmark."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import base64
import gc
import hashlib
import html
import importlib.metadata
import json
import shutil
import time
import unicodedata
import zipfile

DATASET = 'PiotrSty/impact-psnc-polish-ocr'
REVISION = 'c7cb156fb95d2880699c33725bbaf1fbc1008fea'
MODELS = {
    'microsoft/trocr-base-printed': '93450be3f1ed40a930690d951ef3932687cc1892',
    'PiotrSty/trocr-pl-mixed-v3': '85d0c91c26f8e088849096dded7c9ba10b4cd9c9',
}
FROZEN = {'NA2_FT', 'Nowiny_z_Rakuz_FT', 'Powodzenia_FT'}


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def select(rows, test_rows):
    test_hashes = {r['image_sha256'] for r in test_rows}
    selected, excluded = [], []
    seen = set()
    for row in sorted(rows, key=lambda r: r['id']):
        if row['id'] in seen:
            raise ValueError('Duplicate region ID')
        seen.add(row['id'])
        if row['split'] != 'validation' or row['collection'] in FROZEN:
            raise ValueError('Unexpected development split/collection')
        if row['image_sha256'] in test_hashes:
            raise ValueError('Exact image overlap with test')
        text = row['text']
        if '\n' in text or '\r' in text or len(text.strip()) < 20:
            excluded.append({'id': row['id'], 'reason': 'multiline reference or fewer than 20 characters'})
        else:
            selected.append(row)
    if not selected or len(selected) > 24:
        raise ValueError('Unexpected sample size; inspect metadata before inference')
    return selected, excluded


def safe_file(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe input path')
    return str(path)


def score(rows, predictions):
    from jiwer import cer, wer
    if [r['id'] for r in rows] != [r['id'] for r in predictions]:
        raise ValueError('Prediction/reference IDs differ')
    refs = [normalize(r['text']) for r in rows]
    hyps = [normalize(r['text']) for r in predictions]
    return {'cer': cer(refs, hyps), 'wer': wer(refs, hyps), 'regions': len(rows),
            'errors': sum(r['status'] != 'ok' for r in predictions),
            'empty': sum(not h for h in hyps)}


def main():
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    if not torch.cuda.is_available():
        raise RuntimeError('Enable Kaggle GPU and Internet, then Run All')
    torch.manual_seed(0)
    output = Path('/kaggle/working') / ('printed-dev-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    (output / 'crops').mkdir()

    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    def metadata(split):
        name = 'regions/' + split + '/metadata.jsonl'
        path = Path(hf_hub_download(DATASET, name, repo_type='dataset', revision=REVISION))
        shutil.copyfile(path, output / (split + '-metadata.jsonl'))
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

    rows, excluded = select(metadata('validation'), metadata('test'))
    save('selection.json', rows)
    save('excluded.json', excluded)
    report = {'scope': 'Development diagnostic of candidate single-line regions, mostly headings. Not manually audited ground truth, not full pages, not SOTA.',
              'limitations': 'No near-duplicate audit. Upstream model training overlap not ruled out. No newline does not prove a single visual line. Inspect review.html.',
              'dataset': DATASET, 'revision': REVISION, 'models': MODELS,
              'normalization': 'NFC and whitespace only; historical spelling preserved',
              'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 256, 'dtype': 'float32'},
              'environment': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'Pillow', 'jiwer', 'huggingface_hub']},
              'gpu': torch.cuda.get_device_name(0), 'results': {}, 'reference_flags': {}}
    images = []
    for row in rows:
        source = Path(hf_hub_download(DATASET, 'regions/validation/' + safe_file(row['file_name']), repo_type='dataset', revision=REVISION))
        if hashlib.sha256(source.read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError('Crop checksum mismatch: ' + row['id'])
        target = output / 'crops' / (row['id'] + '.jpg')
        shutil.copyfile(source, target)
        with Image.open(source) as im:
            images.append(im.convert('RGB'))
        report['reference_flags'][row['id']] = {'replacement_characters': row['text'].count('\ufffd'),
            'private_use_characters': sum(unicodedata.category(c) == 'Co' for c in row['text'])}
    results = {}
    for model_id, revision in MODELS.items():
        predictions = []
        model = None
        try:
            processor = TrOCRProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=False)
            model = VisionEncoderDecoderModel.from_pretrained(model_id, revision=revision, trust_remote_code=False).float().cuda().eval()
            eos = model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            for row, im in zip(rows, images):
                start = time.perf_counter()
                pred = {'id': row['id'], 'text': '', 'status': 'error'}
                try:
                    # Reference text is only used later for scoring, never for generation.
                    pixels = processor(images=im, return_tensors='pt').pixel_values.cuda()
                    with torch.inference_mode():
                        ids = model.generate(pixels, do_sample=False, num_beams=1, max_new_tokens=256)[0].tolist()
                    pred.update(text=processor.batch_decode([ids], skip_special_tokens=True)[0], status='ok',
                                token_ids=ids, ended_with_eos=ids[-1] in eos,
                                possibly_truncated=len(ids) >= 257 and ids[-1] not in eos)
                except Exception as exc:
                    pred['error'] = type(exc).__name__ + ': ' + str(exc)
                pred['seconds'] = time.perf_counter() - start
                predictions.append(pred)
        except Exception as exc:
            predictions = [{'id': r['id'], 'text': '', 'status': 'error', 'error': type(exc).__name__ + ': ' + str(exc)} for r in rows]
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        results[model_id] = predictions
        report['results'][model_id] = score(rows, predictions)
        save(model_id.replace('/', '--') + '.json', predictions)
        save('report.json', report)
        print(model_id, report['results'][model_id])
    body = ['<!doctype html><meta charset="utf-8"><title>Printed development diagnostic</title>',
            '<style>body{max-width:1100px;margin:30px auto;font-family:system-ui}img{max-width:100%}pre{white-space:pre-wrap}article{border-top:1px solid #bbb;padding:20px 0}</style>',
            '<h1>Development diagnostic, not a benchmark</h1><p>Inspect whether each crop contains one visual line. References may contain corrupted Unicode. Failed outputs remain in metrics.</p>']
    for i, row in enumerate(rows):
        body.append('<article><h2>' + html.escape(row['id']) + '</h2><img src="crops/' + row['id'] + '.jpg"><pre>Reference: ' + html.escape(row['text']) + '</pre>')
        for name, predictions in results.items():
            body.append('<pre>' + html.escape(name + ': ' + predictions[i]['text'] + '\nStatus: ' + predictions[i]['status']) + '</pre>')
        body.append('</article>')
    (output / 'review.html').write_text('\n'.join(body), encoding='utf-8')
    checksums = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
    save('checksums.json', checksums)
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file():
                z.write(p, p.relative_to(output))
    print('Download:', archive)
    if archive.stat().st_size < 25_000_000:
        from IPython.display import HTML, display
        encoded = base64.b64encode(archive.read_bytes()).decode('ascii')
        display(HTML('<a download="' + archive.name + '" href="data:application/zip;base64,' + encoded + '">Pobierz ZIP wynikow</a>'))


if __name__ == '__main__':
    main()
