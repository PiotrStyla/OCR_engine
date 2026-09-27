"""Generate the frozen Colab reading-order development experiment."""
from __future__ import annotations

import json
from pathlib import Path

from training.build_historical_recognizer_colab import code_cell


CODE_REVISION = "5a6ab7b99ad56a15812c91df353866ae7cc745eb"
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-09-27/historical-layout-dev-v1/config.json"
)
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def build(target: str | Path) -> None:
    setup = f'''import gc
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
CONFIG = json.loads((repo / 'experiments/2026-09-27/historical-layout-dev-v1/config.json').read_text())
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

run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
work = Path('/content') / ('historical-layout-dev-v1-' + run_id)
work.mkdir()
print('GPU:', torch.cuda.get_device_name(0))
print('Work:', work)
'''

    inputs = '''from huggingface_hub import hf_hub_download, snapshot_download

def sha256_file(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()

model_spec = CONFIG['model']
model_dir = Path(snapshot_download(
    repo_id=model_spec['repo'],
    revision=model_spec['revision'],
    local_dir=work / 'model',
))
assert sha256_file(model_dir / 'model.safetensors') == model_spec['model_sha256']

dataset = CONFIG['dataset']
metadata_path = Path(hf_hub_download(
    dataset['repo'],
    dataset['metadata_path'],
    repo_type='dataset',
    revision=dataset['revision'],
))
assert sha256_file(metadata_path) == dataset['metadata_sha256']
records = [
    json.loads(line)
    for line in metadata_path.read_text(encoding='utf-8').splitlines()
    if line.strip()
]
assert len(records) == dataset['pages'] == 15
assert len({row['id'] for row in records}) == 15
assert sorted({row['collection'] for row in records}) == dataset['collections']
assert all(row['split'] == dataset['split'] for row in records)
assert all(row['license'] == dataset['license'] for row in records)
assert sum(row['text'].count('\\ufffd') for row in records) == dataset['replacement_characters_in_references']

page_paths = {}
xml_paths = {}
for position, row in enumerate(records, 1):
    image_name = 'pages/validation/' + row['file_name']
    page_path = Path(hf_hub_download(
        dataset['repo'], image_name, repo_type='dataset', revision=dataset['revision']))
    xml_path = Path(hf_hub_download(
        dataset['repo'], row['pagexml'], repo_type='dataset', revision=dataset['revision']))
    assert sha256_file(page_path) == row['image_sha256']
    assert sha256_file(xml_path) == row['pagexml_sha256']
    page_paths[row['id']] = page_path
    xml_paths[row['id']] = xml_path
    print(f'Inputs {position}/15: {row["id"]}', flush=True)
print('PINNED_INPUTS_OK pages=15 collections=5')
'''

    inference = '''from PIL import Image
from ocr.config import OcrConfig
from ocr.detector import sort_reading_order, sort_reading_order_columns
from ocr.opencv_detector import OpenCVDetector
from ocr.preprocess import deskew, load_image
from ocr.recognizer import Recognizer
from training.pagexml_order import order_by_pagexml_regions

runtime_config = OcrConfig(
    recognizer_pl=str(model_dir),
    recognizer_en=str(model_dir),
    recognizer_fallback=str(model_dir),
    recognizer_backend='trocr',
    use_fallback_if_pl_missing=False,
    force_language='pl',
    device='cuda',
    deskew=False,
    line_padding=CONFIG['pipeline']['line_padding'],
    batch_size=CONFIG['pipeline']['batch_size'],
    correct_text=False,
)
detector = OpenCVDetector(runtime_config)
recognizer = Recognizer(runtime_config)
page_dir = work / 'processed-pages-private'
page_dir.mkdir()
private_rows = []

for position, row in enumerate(records, 1):
    started = time.perf_counter()
    source = load_image(page_paths[row['id']])
    assert source.shape[1] == row['width'] and source.shape[0] == row['height']
    processed = deskew(source)
    processed_path = page_dir / (row['id'] + '.png')
    Image.fromarray(processed).save(processed_path, format='PNG')
    boxes = detector.detect(processed)
    assert len({box.to_tuple() for box in boxes}) == len(boxes)

    row_major = sort_reading_order(boxes, processed.shape[0])
    column_aware = sort_reading_order_columns(
        boxes, processed.shape[1], processed.shape[0])
    pagexml_oracle, pagexml_stats = order_by_pagexml_regions(
        boxes, xml_paths[row['id']].read_bytes(), processed.shape[0])
    expected_boxes = sorted(box.to_tuple() for box in boxes)
    assert all(
        sorted(box.to_tuple() for box in variant) == expected_boxes
        for variant in (row_major, column_aware, pagexml_oracle)
    )

    status = 'ok'
    error = None
    decoded_by_box = {}
    try:
        decoded = recognizer.recognize_lines(processed, boxes, ['pl'] * len(boxes))
        decoded_by_box = {
            box.to_tuple(): text.strip()
            for box, (text, _) in zip(boxes, decoded)
        }
    except Exception as exc:
        status = 'error'
        error = f'{type(exc).__name__}: {exc}'

    variants = {}
    for label, ordered in {
        'row_major': row_major,
        'column_aware_v1': column_aware,
        'pagexml_region_oracle': pagexml_oracle,
    }.items():
        variants[label] = '\\n'.join(
            decoded_by_box.get(box.to_tuple(), '')
            for box in ordered
            if decoded_by_box.get(box.to_tuple(), '')
        )
    private_rows.append({
        'id': row['id'],
        'collection': row['collection'],
        'status': status,
        'error': error,
        'reference': row['text'],
        'variants': variants,
        'detected_lines': len(boxes),
        'reference_nonempty_lines': sum(bool(line.strip()) for line in row['text'].splitlines()),
        'column_order_changed': [box.to_tuple() for box in column_aware]
            != [box.to_tuple() for box in row_major],
        'oracle_order_changed': [box.to_tuple() for box in pagexml_oracle]
            != [box.to_tuple() for box in row_major],
        'pagexml_assignment': pagexml_stats,
        'elapsed_seconds': round(time.perf_counter() - started, 3),
    })
    print(f'Inference {position}/15: {row["id"]} status={status} boxes={len(boxes)}', flush=True)

recognizer.close()
del recognizer
gc.collect()
torch.cuda.empty_cache()
(work / 'predictions-private.json').write_text(
    json.dumps(private_rows, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')
print('SINGLE_RECOGNITION_PASS_OK')
'''

    evaluation = '''from jiwer import cer, wer
from training.benchmark_pages import normalize

VARIANTS = ('row_major', 'column_aware_v1', 'pagexml_region_oracle')

def aggregate(rows, label):
    references = [normalize(row['reference']) for row in rows]
    hypotheses = [
        normalize(row['variants'][label]) if row['status'] == 'ok' else ''
        for row in rows
    ]
    return {
        'pages': len(rows),
        'cer_micro': cer(references, hypotheses),
        'wer_micro': wer(references, hypotheses),
        'errors_or_missing': sum(row['status'] != 'ok' for row in rows),
    }

overall = {label: aggregate(private_rows, label) for label in VARIANTS}
collections = {}
for label in VARIANTS:
    collections[label] = {}
    for collection in CONFIG['dataset']['collections']:
        subset = [row for row in private_rows if row['collection'] == collection]
        collections[label][collection] = aggregate(subset, label)

per_page = []
for row in private_rows:
    reference = normalize(row['reference'])
    item = {
        'id': row['id'],
        'collection': row['collection'],
        'reference_characters': len(reference),
        'reference_nonempty_lines': row['reference_nonempty_lines'],
        'detected_lines': row['detected_lines'],
        'column_order_changed': row['column_order_changed'],
        'oracle_order_changed': row['oracle_order_changed'],
        'pagexml_assignment': row['pagexml_assignment'],
        'status': row['status'],
    }
    for label in VARIANTS:
        hypothesis = normalize(row['variants'][label]) if row['status'] == 'ok' else ''
        item[label + '_cer'] = cer(reference, hypothesis)
        item[label + '_wer'] = wer(reference, hypothesis)
    per_page.append(item)

column_collection_deltas = {
    collection: (
        collections['column_aware_v1'][collection]['cer_micro']
        - collections['row_major'][collection]['cer_micro']
    )
    for collection in CONFIG['dataset']['collections']
}
gates = {
    'column_aware_overall_cer_improves': (
        overall['column_aware_v1']['cer_micro'] < overall['row_major']['cer_micro']),
    'column_aware_overall_wer_improves': (
        overall['column_aware_v1']['wer_micro'] < overall['row_major']['wer_micro']),
    'maximum_collection_cer_regression_within_1pp': (
        max(column_collection_deltas.values()) <= 0.01 + 1e-12),
    'column_aware_errors_or_missing_zero': (
        overall['column_aware_v1']['errors_or_missing'] == 0),
}
gates['all_gates_passed'] = all(gates.values())

summary = {
    'schema': 'ocr-engine-historical-layout-dev-result-v1',
    'scope': CONFIG['scope'],
    'pages': len(private_rows),
    'overall': overall,
    'delta_column_aware_minus_row_major': {
        'cer_micro': overall['column_aware_v1']['cer_micro'] - overall['row_major']['cer_micro'],
        'wer_micro': overall['column_aware_v1']['wer_micro'] - overall['row_major']['wer_micro'],
    },
    'delta_oracle_minus_row_major': {
        'cer_micro': overall['pagexml_region_oracle']['cer_micro'] - overall['row_major']['cer_micro'],
        'wer_micro': overall['pagexml_region_oracle']['wer_micro'] - overall['row_major']['wer_micro'],
    },
    'collections': collections,
    'column_collection_cer_deltas': column_collection_deltas,
    'paired_page_cer_wins': {
        'column_aware_v1': sum(
            row['column_aware_v1_cer'] < row['row_major_cer'] - 1e-12 for row in per_page),
        'row_major': sum(
            row['row_major_cer'] < row['column_aware_v1_cer'] - 1e-12 for row in per_page),
        'ties': sum(
            abs(row['column_aware_v1_cer'] - row['row_major_cer']) <= 1e-12
            for row in per_page),
    },
    'layout_diagnostics': {
        'detected_lines': sum(row['detected_lines'] for row in private_rows),
        'reference_nonempty_lines': sum(row['reference_nonempty_lines'] for row in private_rows),
        'column_order_changed_pages': sum(row['column_order_changed'] for row in private_rows),
        'oracle_order_changed_pages': sum(row['oracle_order_changed'] for row in private_rows),
        'pagexml_overlap_assigned_lines': sum(
            row['pagexml_assignment']['overlap_assigned_lines'] for row in private_rows),
        'pagexml_nearest_assigned_lines': sum(
            row['pagexml_assignment']['nearest_assigned_lines'] for row in private_rows),
    },
    'gates': gates,
    'claim_boundary': CONFIG['claim_boundary'],
}
(work / 'metrics.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')
(work / 'per-page-metrics.json').write_text(
    json.dumps(per_page, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
'''

    evidence = '''evidence_dir = work / 'evidence'
if evidence_dir.exists():
    shutil.rmtree(evidence_dir)
evidence_dir.mkdir()
for source, name in {
    repo / 'experiments/2026-09-27/historical-layout-dev-v1/config.json': 'experiment-config.json',
    work / 'metrics.json': 'metrics.json',
    work / 'per-page-metrics.json': 'per-page-metrics.json',
}.items():
    shutil.copyfile(source, evidence_dir / name)

environment = {
    'code_revision': CODE_REVISION,
    'dataset': CONFIG['dataset'],
    'model': CONFIG['model'],
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
    'pagexml_included': False,
    'model_weights_included': False,
    'private_token_recorded': False,
}
(evidence_dir / 'environment.json').write_text(
    json.dumps(environment, indent=2) + '\\n', encoding='utf-8')
checksums = {
    path.name: sha256_file(path)
    for path in sorted(evidence_dir.iterdir())
    if path.is_file()
}
(evidence_dir / 'checksums.json').write_text(
    json.dumps(checksums, indent=2) + '\\n', encoding='utf-8')
evidence_zip = Path(shutil.make_archive(
    str(work / 'historical-layout-dev-v1-evidence'), 'zip', evidence_dir))
print('Evidence:', evidence_zip)
print('No images, PAGE XML, references, raw predictions, weights or token are included.')
'''

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "colab": {"name": "colab_historical_layout_dev_v1.ipynb", "provenance": []},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "scope",
                "metadata": {},
                "source": [
                    "# Historical layout development v1\n",
                    "Choose a GPU runtime and click **Run all**. No token is required. The notebook evaluates 15 pinned public validation pages and never loads the frozen 36-page test.\n",
                    "Each page is segmented once and each line is recognized once with mixed-v3. Row-major, automatic column-aware and diagnostic PAGE-region order differ only by permutation. Historical spelling is preserved.\n",
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
    build(Path(__file__).with_name("colab_historical_layout_dev_v1.ipynb"))
