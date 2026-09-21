"""Diagnostic only: user-corrected draft references, not approved benchmark labels."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import base64
import gc
import hashlib
import importlib.metadata
import io
import json
import unicodedata
import zipfile

MODELS = {
    'microsoft/trocr-base-printed': '93450be3f1ed40a930690d951ef3932687cc1892',
    'PiotrSty/trocr-pl-mixed-v3': '85d0c91c26f8e088849096dded7c9ba10b4cd9c9',
}
FROZEN = {'NA2_FT', 'Nowiny_z_Rakuz_FT', 'Powodzenia_FT'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pack(manifest, target):
    manifest, target = Path(manifest), Path(target)
    if target.exists():
        raise FileExistsError(target)
    source = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
    rows, images = [], {}
    for i, row in enumerate(source):
        path = (manifest.parent / row['image']).resolve()
        if not path.is_relative_to(manifest.parent.resolve()):
            raise ValueError('Image outside draft directory')
        data = path.read_bytes()
        if digest(data) != row['sha256'] or row['eligible_for_evaluation'] is not False:
            raise ValueError('Expected checksum-verified draft')
        name = f'images/{i:04d}.png'
        images[name] = data
        rows.append({k: row[k] for k in ['id', 'text', 'original_text', 'collection', 'page_id',
                                        'sha256', 'source_review_decision', 'review_status', 'eligible_for_evaluation']})
        rows[-1]['image'] = name
    provenance = {'scope': 'diagnostic-only', 'source_manifest_sha256': digest(manifest.read_bytes()),
                  'reference_status': 'User-corrected draft; repeat review explicitly skipped.',
                  'sampling': '63 count-matched line proposals from 11 of 19 selected regions. Eight regions failed line-count matching. Not full-page evaluation.',
                  'limitations': 'Geometry and Unicode issues remain; no independent double review, near-duplicate audit or proof of upstream training separation.'}
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.json', json.dumps(rows, ensure_ascii=False))
        archive.writestr('provenance.json', json.dumps(provenance))
        for name, data in images.items():
            archive.writestr(name, data)
    return digest(target.read_bytes())


def load_input(data, expected_sha256):
    if digest(data) != expected_sha256:
        raise ValueError('Input ZIP checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        if sum(i.file_size for i in archive.infolist()) > 50_000_000:
            raise ValueError('Input too large')
        for name in names:
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe ZIP member')
        rows = json.loads(archive.read('manifest.json'))
        provenance = json.loads(archive.read('provenance.json'))
        if provenance['scope'] != 'diagnostic-only' or not rows or len(rows) > 200:
            raise ValueError('Invalid diagnostic input')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate line ID')
        images = []
        for r in rows:
            if r['collection'] in FROZEN or r['eligible_for_evaluation'] is not False:
                raise ValueError('Expected non-test draft records')
            if r['source_review_decision'] not in {'verified', 'proposed', 'needs-review'}:
                raise ValueError('Missing source review decision')
            if not r['text'].strip():
                raise ValueError('Empty reference requires explicit policy')
            image = archive.read(r['image'])
            if digest(image) != r['sha256']:
                raise ValueError('Image checksum mismatch')
            images.append(image)
    return rows, images, provenance


def metrics(rows, predictions):
    from jiwer import cer, wer
    if [r['id'] for r in rows] != [p['id'] for p in predictions]:
        raise ValueError('Reference/prediction ID mismatch')
    normalize = lambda t: ' '.join(unicodedata.normalize('NFC', t).split())
    result = {}
    subsets = {'all_draft_lines': list(range(len(rows))),
               'without_needs_review': [i for i, r in enumerate(rows) if r['source_review_decision'] != 'needs-review']}
    for label, indices in subsets.items():
        if not indices:
            result[label] = {'lines': 0, 'cer': None, 'wer': None}
            continue
        refs = [normalize(rows[i]['text']) for i in indices]
        hyps = [normalize(predictions[i]['text']) for i in indices]
        result[label] = {'lines': len(indices), 'cer': cer(refs, hyps), 'wer': wer(refs, hyps),
                         'lowercase_cer_diagnostic': cer([s.lower() for s in refs], [s.lower() for s in hyps]),
                         'errors': sum(predictions[i]['status'] != 'ok' for i in indices),
                         'empty': sum(not s for s in hyps)}
    return result


def run(data, expected_sha256, runner_sha256=None):
    rows, image_bytes, provenance = load_input(data, expected_sha256)
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU and Internet in Kaggle')
    torch.manual_seed(0)
    output = Path('/kaggle/working') / ('body-dev-diagnostic-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    save('input-manifest.json', rows)
    save('provenance.json', provenance)
    report = {'scope': 'DRAFT diagnostic, not benchmark or SOTA evidence', 'input_zip_sha256': expected_sha256,
              'runner_sha256': runner_sha256,
              'models': MODELS, 'gpu': torch.cuda.get_device_name(0),
              'environment': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'Pillow', 'jiwer', 'huggingface_hub']},
              'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 256, 'dtype': 'float32'},
              'normalization': 'NFC and whitespace; additional lowercase CER reported separately',
              'uncertain_ids': [r['id'] for r in rows if r['source_review_decision'] == 'needs-review'],
              'reference_flags': {r['id']: sum(c == '\ufffd' or unicodedata.category(c) == 'Co' for c in r['text']) for r in rows},
              'results': {}}
    for model_id, revision in MODELS.items():
        model = None
        predictions = []
        try:
            processor = TrOCRProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=False)
            model = VisionEncoderDecoderModel.from_pretrained(model_id, revision=revision, trust_remote_code=False).float().cuda().eval()
            eos = model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            for row, content in zip(rows, image_bytes):
                pred = {'id': row['id'], 'text': '', 'status': 'error'}
                try:
                    with Image.open(io.BytesIO(content)) as im:
                        pixels = processor(images=im.convert('RGB'), return_tensors='pt').pixel_values.cuda()
                    with torch.inference_mode():
                        ids = model.generate(pixels, do_sample=False, num_beams=1, max_new_tokens=256)[0].tolist()
                    pred.update(text=processor.batch_decode([ids], skip_special_tokens=True)[0], status='ok',
                                token_ids=ids, ended_with_eos=ids[-1] in eos,
                                possibly_truncated=len(ids) >= 257 and ids[-1] not in eos)
                except Exception as exc:
                    pred['error'] = type(exc).__name__ + ': ' + str(exc)
                predictions.append(pred)
        except Exception as exc:
            predictions = [{'id': r['id'], 'text': '', 'status': 'error', 'error': type(exc).__name__ + ': ' + str(exc)} for r in rows]
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        save(model_id.replace('/', '--') + '.json', predictions)
        report['results'][model_id] = metrics(rows, predictions)
        save('report.json', report)
        print(model_id, json.dumps(report['results'][model_id], indent=2))
    save('checksums.json', {p.name: digest(p.read_bytes()) for p in output.iterdir() if p.is_file()})
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in output.iterdir():
            z.write(path, path.name)
    from IPython.display import HTML, display
    encoded = base64.b64encode(archive.read_bytes()).decode('ascii')
    display(HTML('<a download="' + archive.name + '" href="data:application/zip;base64,' + encoded + '">Pobierz ZIP wynikow</a>'))
    print('Output:', archive)
