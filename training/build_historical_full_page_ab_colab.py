"""Generate the frozen Colab full-page A/B evaluation for historical OCR."""
from __future__ import annotations

import json
from pathlib import Path

from training.build_historical_recognizer_colab import code_cell


CODE_REVISION = "d49704129fdee6f11530ddc25c574f795681678b"
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-09-27/historical-full-page-ab-v1/config.json"
)
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def build(target: str | Path) -> None:
    setup = f'''import gc
import getpass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

CODE_REVISION = {CODE_REVISION!r}
EXPECTED_CONFIG = {CONFIG!r}

repo = Path('/content/OCR_engine')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/PiotrStyla/OCR_engine.git', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', CODE_REVISION], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', CODE_REVISION], check=True)
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == CODE_REVISION
CONFIG = json.loads((repo / 'experiments/2026-09-27/historical-full-page-ab-v1/config.json').read_text())
assert CONFIG == EXPECTED_CONFIG
os.chdir(repo)
sys.path.insert(0, str(repo))

for name in list(sys.modules):
    if (name == 'huggingface_hub' or name.startswith('huggingface_hub.')
            or name == 'transformers' or name.startswith('transformers.')
            or name == 'tokenizers' or name.startswith('tokenizers.')):
        del sys.modules[name]

import torch
assert torch.cuda.is_available(), 'Select a GPU runtime, then Run all.'
assert importlib.metadata.version('transformers') == '4.57.6'
assert importlib.metadata.version('tokenizers') == '0.22.2'
assert importlib.metadata.version('huggingface_hub') == '0.36.2'
assert importlib.metadata.version('jiwer') == '4.0.0'
assert importlib.metadata.version('Pillow') == '11.3.0'
assert importlib.metadata.version('opencv-python-headless') == '4.12.0.88'
from transformers.generation import GenerationMixin
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
assert GenerationMixin is not None and TrOCRProcessor is not None and VisionEncoderDecoderModel is not None
print('TRANSFORMERS_IMPORT_PREFLIGHT_OK')

try:
    from google.colab import userdata
    HF_TOKEN = (userdata.get('HF_TOKEN') or '').strip()
except Exception:
    HF_TOKEN = ''
if not HF_TOKEN:
    HF_TOKEN = getpass.getpass('Hugging Face token with access to the private v2 model: ').strip()
assert HF_TOKEN, 'HF_TOKEN is required for the private candidate model.'
from huggingface_hub import whoami
assert whoami(token=HF_TOKEN).get('name') == 'PiotrSty'
print('HF_PRIVATE_ACCESS_OK')

run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
work = Path('/content') / ('historical-full-page-ab-v1-' + run_id)
work.mkdir()
print('GPU:', torch.cuda.get_device_name(0))
print('Work:', work)
'''

    inputs = '''from huggingface_hub import hf_hub_download, snapshot_download
from training.slayer_vision_onnx_smoke import find_image, safe_extract_tar

def sha256_file(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()

models = {}
for label in ('baseline', 'candidate'):
    spec = CONFIG['models'][label]
    local = work / (label + '-model')
    models[label] = Path(snapshot_download(
        repo_id=spec['repo'],
        revision=spec['revision'],
        token=HF_TOKEN,
        local_dir=local,
    ))
    observed = sha256_file(models[label] / 'model.safetensors')
    assert observed == spec['model_sha256'], (label, observed)

dataset = CONFIG['dataset']
archive_path = Path(hf_hub_download(
    dataset['repo'],
    dataset['file'],
    repo_type='dataset',
    revision=dataset['revision'],
))
assert sha256_file(archive_path) == dataset['archive_sha256']
safe_extract_tar(archive_path, work / 'dataset')
bench_dir = work / 'dataset' / 'impact-print-v2'
manifest_path = bench_dir / 'test_manifest.jsonl'
records = [
    json.loads(line)
    for line in manifest_path.read_text(encoding='utf-8').splitlines()
    if line.strip()
]
assert len(records) == dataset['pages'] == 36
assert len({row['id'] for row in records}) == 36
assert len({row['id'].split('__')[0] for row in records}) == dataset['collections'] == 3
page_paths = {}
for row in records:
    path = find_image(work / 'dataset', bench_dir, row)
    assert sha256_file(path) == row['sha256']
    page_paths[row['id']] = path

dataset_root = (work / 'dataset').resolve()
portable_records = []
for row in records:
    resolved = page_paths[row['id']].resolve()
    relative = resolved.relative_to(dataset_root).as_posix()
    assert '\\\\' not in relative and not relative.startswith('../')
    portable_records.append({**row, 'image': relative})
manifest_path = work / 'dataset' / 'evaluation-manifest.jsonl'
manifest_path.write_text(
    ''.join(json.dumps(row, ensure_ascii=False) + '\\n' for row in portable_records),
    encoding='utf-8',
)
records = portable_records
print('PINNED_INPUTS_OK pages=36 collections=3')
'''

    segmentation = '''import numpy as np
from PIL import Image
from ocr.config import OcrConfig
from ocr.opencv_detector import OpenCVDetector
from ocr.preprocess import deskew, load_image

page_dir = work / 'segmented-pages'
page_dir.mkdir()
detector_config = OcrConfig(
    device='cuda',
    deskew=False,
    line_padding=CONFIG['pipeline']['line_padding'],
    batch_size=CONFIG['pipeline']['batch_size'],
    correct_text=False,
)
detector = OpenCVDetector(detector_config)
segmentation_rows = []
for position, row in enumerate(records, 1):
    source = load_image(page_paths[row['id']])
    processed = deskew(source)
    processed_path = page_dir / (row['id'] + '.png')
    Image.fromarray(processed).save(processed_path, format='PNG')
    boxes = detector.detect(processed)
    segmentation_rows.append({
        'id': row['id'],
        'collection': row['id'].split('__')[0],
        'processed_image_sha256': sha256_file(processed_path),
        'boxes': [[box.x1, box.y1, box.x2, box.y2] for box in boxes],
        'detected_lines': len(boxes),
        'reference_nonempty_lines': sum(bool(line.strip()) for line in row['text'].splitlines()),
    })
    print(f'Segmentation {position}/36: {row["id"]} boxes={len(boxes)}', flush=True)
assert len(segmentation_rows) == 36
(work / 'segmentation-private.json').write_text(
    json.dumps(segmentation_rows, indent=2) + '\\n', encoding='utf-8'
)
print('SINGLE_SEGMENTATION_PASS_OK')
'''

    inference = '''from ocr.recognizer import Recognizer
from ocr.result import BBox

def run_model(label):
    recognizer = Recognizer(OcrConfig(
        recognizer_pl=str(models[label]),
        recognizer_en=str(models[label]),
        recognizer_fallback=str(models[label]),
        recognizer_backend='trocr',
        use_fallback_if_pl_missing=False,
        force_language='pl',
        device='cuda',
        deskew=False,
        line_padding=CONFIG['pipeline']['line_padding'],
        batch_size=CONFIG['pipeline']['batch_size'],
        correct_text=False,
    ))
    output = []
    for position, geometry in enumerate(segmentation_rows, 1):
        started = time.perf_counter()
        status = 'ok'
        error = None
        text = ''
        try:
            image = load_image(page_dir / (geometry['id'] + '.png'))
            boxes = [BBox(*values) for values in geometry['boxes']]
            decoded = recognizer.recognize_lines(image, boxes, ['pl'] * len(boxes))
            text = '\\n'.join(value.strip() for value, _ in decoded if value.strip())
        except Exception as exc:
            status = 'error'
            error = f'{type(exc).__name__}: {exc}'
        output.append({
            'id': geometry['id'],
            'status': status,
            'text': text,
            'detected_lines': geometry['detected_lines'],
            'recognized_nonempty_lines': sum(bool(line.strip()) for line in text.splitlines()),
            'elapsed_seconds': round(time.perf_counter() - started, 3),
            'error': error,
        })
        print(f'{label} {position}/36: {geometry["id"]} status={status}', flush=True)
    recognizer.close()
    del recognizer
    gc.collect()
    torch.cuda.empty_cache()
    path = work / (label + '-predictions-private.jsonl')
    path.write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\\n' for row in output),
        encoding='utf-8',
    )
    return output, path

baseline_predictions, baseline_predictions_path = run_model('baseline')
candidate_predictions, candidate_predictions_path = run_model('candidate')
'''

    evaluation = '''from jiwer import cer, wer
from training.benchmark_pages import evaluate, normalize

baseline_report = evaluate(manifest_path, baseline_predictions_path)
candidate_report = evaluate(manifest_path, candidate_predictions_path)
record_by_id = {row['id']: row for row in records}
prediction_maps = {
    'baseline': {row['id']: row for row in baseline_predictions},
    'candidate': {row['id']: row for row in candidate_predictions},
}

def collection_metrics(label):
    output = {}
    for collection in sorted({row['id'].split('__')[0] for row in records}):
        subset = [row for row in records if row['id'].split('__')[0] == collection]
        references = [normalize(row['text']) for row in subset]
        hypotheses = [
            normalize(prediction_maps[label][row['id']]['text'])
            if prediction_maps[label][row['id']]['status'] == 'ok' else ''
            for row in subset
        ]
        output[collection] = {
            'pages': len(subset),
            'cer_micro': cer(references, hypotheses),
            'wer_micro': wer(references, hypotheses),
        }
    return output

collections = {
    'baseline': collection_metrics('baseline'),
    'candidate': collection_metrics('candidate'),
}
page_rows = []
baseline_page = {row['id']: row for row in baseline_report['results']}
candidate_page = {row['id']: row for row in candidate_report['results']}
geometry_by_id = {row['id']: row for row in segmentation_rows}
for row in records:
    identifier = row['id']
    page_rows.append({
        'id': identifier,
        'collection': identifier.split('__')[0],
        'reference_characters': len(normalize(row['text'])),
        'reference_nonempty_lines': geometry_by_id[identifier]['reference_nonempty_lines'],
        'detected_lines': geometry_by_id[identifier]['detected_lines'],
        'baseline_status': baseline_page[identifier]['status'],
        'baseline_cer': baseline_page[identifier]['cer'],
        'baseline_wer': baseline_page[identifier]['wer'],
        'candidate_status': candidate_page[identifier]['status'],
        'candidate_cer': candidate_page[identifier]['cer'],
        'candidate_wer': candidate_page[identifier]['wer'],
    })

wins = {
    'candidate': sum(row['candidate_cer'] < row['baseline_cer'] - 1e-12 for row in page_rows),
    'baseline': sum(row['baseline_cer'] < row['candidate_cer'] - 1e-12 for row in page_rows),
    'ties': sum(abs(row['candidate_cer'] - row['baseline_cer']) <= 1e-12 for row in page_rows),
}
collection_cer_deltas = {
    name: collections['candidate'][name]['cer_micro'] - collections['baseline'][name]['cer_micro']
    for name in collections['baseline']
}
gates = {
    'candidate_overall_cer_improves': candidate_report['cer_micro'] < baseline_report['cer_micro'],
    'candidate_overall_wer_improves': candidate_report['wer_micro'] < baseline_report['wer_micro'],
    'maximum_collection_cer_regression_within_1pp': max(collection_cer_deltas.values()) <= 0.01 + 1e-12,
    'candidate_errors_or_missing_zero': candidate_report['errors_or_missing'] == 0,
}
gates['all_gates_passed'] = all(gates.values())
summary = {
    'schema': 'ocr-engine-historical-full-page-ab-result-v1',
    'pages': 36,
    'baseline': {key: baseline_report[key] for key in ('cer_micro', 'wer_micro', 'errors_or_missing')},
    'candidate': {key: candidate_report[key] for key in ('cer_micro', 'wer_micro', 'errors_or_missing')},
    'delta': {
        'cer_micro': candidate_report['cer_micro'] - baseline_report['cer_micro'],
        'wer_micro': candidate_report['wer_micro'] - baseline_report['wer_micro'],
    },
    'collections': collections,
    'collection_cer_deltas': collection_cer_deltas,
    'paired_page_cer_wins': wins,
    'segmentation': {
        'detected_lines': sum(row['detected_lines'] for row in segmentation_rows),
        'reference_nonempty_lines': sum(row['reference_nonempty_lines'] for row in segmentation_rows),
        'pages_with_zero_boxes': sum(row['detected_lines'] == 0 for row in segmentation_rows),
    },
    'gates': gates,
    'claim_boundary': CONFIG['claim_boundary'],
}
(work / 'metrics.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8'
)
(work / 'per-page-metrics.json').write_text(
    json.dumps(page_rows, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8'
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
'''

    evidence = '''evidence_dir = work / 'evidence'
if evidence_dir.exists():
    shutil.rmtree(evidence_dir)
evidence_dir.mkdir()
for source, name in {
    repo / 'experiments/2026-09-27/historical-full-page-ab-v1/config.json': 'experiment-config.json',
    work / 'metrics.json': 'metrics.json',
    work / 'per-page-metrics.json': 'per-page-metrics.json',
}.items():
    shutil.copyfile(source, evidence_dir / name)

environment = {
    'code_revision': CODE_REVISION,
    'dataset': CONFIG['dataset'],
    'models': CONFIG['models'],
    'pipeline': CONFIG['pipeline'],
    'gpu': torch.cuda.get_device_name(0),
    'python': sys.version,
    'packages': {
        name: importlib.metadata.version(name)
        for name in ('torch', 'transformers', 'jiwer', 'huggingface_hub',
                     'tokenizers', 'Pillow', 'opencv-python-headless')
    },
    'raw_predictions_included': False,
    'images_or_references_included': False,
    'model_weights_included': False,
    'private_token_recorded': False,
}
(evidence_dir / 'environment.json').write_text(
    json.dumps(environment, indent=2) + '\\n', encoding='utf-8'
)
checksums = {
    path.name: sha256_file(path)
    for path in sorted(evidence_dir.iterdir())
    if path.is_file()
}
(evidence_dir / 'checksums.json').write_text(
    json.dumps(checksums, indent=2) + '\\n', encoding='utf-8'
)
evidence_zip = Path(shutil.make_archive(
    str(work / 'historical-full-page-ab-v1-evidence'),
    'zip',
    evidence_dir,
))
print('Evidence:', evidence_zip)
print('No images, references, raw predictions, weights or token are included.')
'''

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "colab": {"name": "colab_historical_full_page_ab_v1.ipynb", "provenance": []},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "scope",
                "metadata": {},
                "source": [
                    "# Historical full-page recognizer A/B v1\n",
                    "Choose a GPU runtime, add the private Colab secret `HF_TOKEN`, and click **Run all**. "
                    "The notebook evaluates all 36 pinned IMPACT test pages. Each page is segmented once; "
                    "the baseline and candidate receive identical line boxes and reading order.\n",
                    "The output evidence excludes page images, references, raw predictions, model weights and "
                    "the token. This evaluates the current OpenCV full-page pipeline and is not a SOTA claim.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "install",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "%pip uninstall -q -y torchao transformers tokenizers huggingface_hub\n",
                    "%pip install -q --no-cache-dir transformers==4.57.6 tokenizers==0.22.2 "
                    "jiwer==4.0.0 huggingface_hub==0.36.2 sentencepiece==0.2.1 "
                    "pillow==11.3.0 opencv-python-headless==4.12.0.88\n",
                ],
            },
            code_cell("setup", setup),
            code_cell("inputs", inputs),
            code_cell("segmentation", segmentation),
            code_cell("inference", inference),
            code_cell("evaluation", evaluation),
            code_cell("evidence", evidence),
            {
                "cell_type": "code",
                "id": "download",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "from google.colab import files\n",
                    "files.download(str(evidence_zip))\n",
                ],
            },
        ],
    }
    Path(target).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build(Path(__file__).with_name("colab_historical_full_page_ab_v1.ipynb"))
