"""One-page in-domain sanity check, NOT an independent benchmark."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import importlib.metadata
import json
import os
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

REVISION = '1947d4c107936ed59ded16446a77b6587fa3874a'
PAGE = 'EHRI-ET-ZIH3010106_01'
WEIGHTS = {
    'polish_nfd_finetuned.safetensors': '9b731f0694af3c0c4c6f31a689fc4f2788292bab31eedaad85184698ee8f3ff6',
    'polish_seg_best.safetensors': 'a817f2a1d3a0eee51b456008ba9952afe60b2ade712f2e7d75ec1608d008ea86',
}


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def parse_alto(path):
    root = ET.parse(path).getroot()
    page = root.find('.//{*}Page')
    if page is None:
        raise ValueError('Missing ALTO Page')
    size = (int(page.attrib['WIDTH']), int(page.attrib['HEIGHT']))
    unit = root.find('.//{*}MeasurementUnit')
    if unit is None or unit.text.strip() != 'pixel':
        raise ValueError('ALTO must use pixel coordinates')
    rows = []
    for line in page.findall('.//{*}TextLine'):
        polygon = line.find('.//{*}Polygon')
        def points(value, minimum):
            numbers = [float(v) for v in value.replace(',', ' ').split()]
            if len(numbers) % 2 or len(numbers) < minimum * 2:
                raise ValueError('Incomplete line geometry')
            pairs = list(zip(numbers[::2], numbers[1::2]))
            if any(not (0 <= x < size[0] and 0 <= y < size[1]) for x, y in pairs):
                raise ValueError('Geometry outside ALTO page')
            return pairs
        if polygon is None:
            raise ValueError('Missing line polygon; no silent exclusions')
        strings = line.findall('{*}String')
        text = ' '.join(s.attrib['CONTENT'] for s in strings)
        rows.append({'id': line.attrib['ID'], 'text': text,
                     'baseline': points(line.attrib.get('BASELINE', ''), 2),
                     'boundary': points(polygon.attrib.get('POINTS', ''), 3)})
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate lines')
    return size, rows


def main():
    if importlib.metadata.version('kraken') != '7.1.1':
        raise RuntimeError('Install kraken==7.1.1 and jiwer==4.0.0 first')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from jiwer import cer, wer
    from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel
    from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
    from kraken.containers import Segmentation, BaselineLine
    if not torch.cuda.is_available():
        raise RuntimeError('Enable Kaggle GPU')
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    output = Path('/kaggle/working') / ('ehri-control-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(exist_ok=False)
    files = {}
    provenance = {}
    for name in ['data/polish/' + PAGE + ext for ext in ('.tif', '.xml')] + ['models/' + name for name in WEIGHTS]:
        path = Path(hf_hub_download('PiotrSty/ehri-dataset', name, repo_type='dataset', revision=REVISION))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.name in WEIGHTS and digest != WEIGHTS[path.name]:
            raise ValueError('Model checksum mismatch')
        files[path.name] = path
        provenance[name] = digest
    xml_path = files[PAGE + '.xml']
    size, rows = parse_alto(xml_path)
    with Image.open(files[PAGE + '.tif']) as source:
        source.seek(0)
        image = source.convert('L')
    if image.size != size:
        raise ValueError('Image/ALTO size mismatch; do not rescale silently')
    (output / 'reference.xml').write_bytes(xml_path.read_bytes())
    rec = RecognitionTaskModel.load_model(str(files['polish_nfd_finetuned.safetensors']))
    seg = SegmentationTaskModel.load_model(str(files['polish_seg_best.safetensors']))
    rc = RecognitionInferenceConfig(accelerator='cuda', device=[0], return_line_image=True)
    sc = SegmentationInferenceConfig(accelerator='cuda', device=[0])
    # Only geometry is passed into oracle inference. Reference text is not.
    oracle = Segmentation(type='baselines', imagename=str(files[PAGE + '.tif']),
        text_direction='horizontal-lr', script_detection=False,
        lines=[BaselineLine(id=r['id'], baseline=r['baseline'], boundary=r['boundary']) for r in rows])
    reference = normalize('\n'.join(r['text'] for r in rows))
    report = {'scope': 'In-domain sanity check; train/validation overlap possible, NOT held-out quality',
              'selection': 'Fixed first EHRI page, not selected by score', 'page': PAGE,
              'revision': REVISION, 'input_hashes': provenance, 'tracks': {},
              'normalization': 'NFC + whitespace; XML line order; String tokens separated by spaces',
              'gpu': torch.cuda.get_device_name(0),
              'packages': {n: importlib.metadata.version(n) for n in ('kraken', 'torch', 'Pillow', 'jiwer')}}
    for mode in ('oracle', 'end_to_end'):
        folder = output / mode
        folder.mkdir()
        try:
            with torch.inference_mode():
                regions = oracle if mode == 'oracle' else seg.predict(image, sc)
                records = list(rec.predict(image, regions, rc))
            if [str(r.id) for r in records] != [str(line.id) for line in regions.lines]:
                raise ValueError('Line identity mismatch')
            predictions = []
            for i, record in enumerate(records):
                crop_name = None
                if record.image is not None:
                    crop_name = f'line-{i:03d}.png'
                    record.image.save(folder / crop_name)
                predictions.append({'id': str(record.id), 'text': record.prediction, 'crop': crop_name})
            hypothesis = normalize('\n'.join(r['text'] for r in predictions))
            (folder / 'predictions.json').write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding='utf-8')
            result = {'status': 'ok', 'lines': len(records), 'empty_lines': sum(not r['text'].strip() for r in predictions),
                      'reference': reference, 'hypothesis': hypothesis,
                      'page_cer': cer(reference, hypothesis), 'page_wer': wer(reference, hypothesis)}
            if mode == 'oracle':
                refs = [normalize(r['text']) for r in rows]
                hyps = [normalize(r['text']) for r in predictions]
                result.update(line_cer_micro=cer(refs, hyps), line_wer_micro=wer(refs, hyps))
        except Exception as exc:
            result = {'status': 'error', 'error_type': type(exc).__name__}
        report['tracks'][mode] = result
        print(mode, {k: v for k, v in result.items() if k not in ('reference', 'hypothesis')}, flush=True)
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    hashes = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
    (output / 'checksums.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    archive_path = output.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in output.rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    print('Download:', archive_path)
    from IPython.display import HTML, display
    if archive_path.stat().st_size < 25_000_000:
        encoded = base64.b64encode(archive_path.read_bytes()).decode('ascii')
        display(HTML(f'<a download="{archive_path.name}" href="data:application/zip;base64,{encoded}">Pobierz ZIP kontroli EHRI</a>'))
    else:
        print('ZIP is available in the Kaggle Output file panel.')


if __name__ == '__main__':
    main()
