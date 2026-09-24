"""Generate the hash-pinned Colab pilot for historical-print recognition."""
from __future__ import annotations

import json
from pathlib import Path


CODE_REVISION = "009d79ecf1f342f4f16557d91bdeaa8ef5e83820"
BASE_MODEL = "PiotrSty/trocr-pl-mixed-v3"
BASE_REVISION = "85d0c91c26f8e088849096dded7c9ba10b4cd9c9"
EHRI_REVISION = "3003e8614b74a351e7d94aba4f1348368815fb70"


def code_cell(identifier: str, source: str) -> dict:
    compile(source, f"<{identifier}>", "exec")
    return {
        "cell_type": "code",
        "id": identifier,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [source],
    }


def build(target: str | Path) -> None:
    setup = f'''import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

CODE_REVISION = {CODE_REVISION!r}
BASE_MODEL = {BASE_MODEL!r}
BASE_REVISION = {BASE_REVISION!r}
EHRI_REVISION = {EHRI_REVISION!r}

repo = Path('/content/OCR_engine')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/PiotrStyla/OCR_engine.git', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', CODE_REVISION], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', CODE_REVISION], check=True)
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == CODE_REVISION
os.chdir(repo)
sys.path.insert(0, str(repo))

for name in list(sys.modules):
    if name == 'huggingface_hub' or name.startswith('huggingface_hub.') or name == 'transformers' or name.startswith('transformers.'):
        del sys.modules[name]

import torch
assert torch.cuda.is_available(), 'Select a GPU runtime, then Run all.'
assert importlib.metadata.version('transformers') == '4.57.6'
assert importlib.metadata.version('peft') == '0.19.1'
assert importlib.metadata.version('accelerate') == '1.13.0'
assert importlib.metadata.version('jiwer') == '4.0.0'
assert importlib.metadata.version('huggingface_hub') == '0.36.2'
assert importlib.metadata.version('Pillow') == '11.3.0'
assert importlib.metadata.version('opencv-python-headless') == '4.12.0.88'

run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
work = Path('/content') / ('historical-recognizer-v1-' + run_id)
work.mkdir()
print('GPU:', torch.cuda.get_device_name(0))
print('Work:', work)
'''

    corpus = '''from training.prepare_historical_line_corpus import prepare

corpus_root = work / 'historical-lines-v1'
corpus_report = prepare(
    corpus_root,
    repo / 'experiments/2026-09-24/geometry-holdout-v1/manifest.json',
)
assert corpus_report['stats']['train']['accepted_lines'] == 258
assert corpus_report['stats']['validation']['accepted_lines'] == 139
assert corpus_report['stats']['train']['accepted_regions'] == 94
assert corpus_report['stats']['validation']['accepted_regions'] == 69
assert corpus_report['quarantine_records'] == 409
print(json.dumps(corpus_report['stats'], indent=2))
print('Historical corpus verified. No PUA/U+FFFD label entered training.')
'''

    tokenizer_audit = '''import unicodedata
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
tokenizer_rows = []
roundtrip_mismatches = []
for split in ('train', 'validation'):
    for path in sorted((corpus_root / split).glob('*.txt')):
        text = path.read_text(encoding='utf-8').strip()
        token_ids = processor.tokenizer(text, truncation=False).input_ids
        decoded = processor.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        tokenizer_rows.append({'id': path.stem, 'split': split, 'tokens': len(token_ids)})
        if unicodedata.normalize('NFC', decoded) != unicodedata.normalize('NFC', text):
            roundtrip_mismatches.append(path.stem)

tokenizer_audit = {
    'labels': len(tokenizer_rows),
    'max_tokens': max(row['tokens'] for row in tokenizer_rows),
    'over_128': sum(row['tokens'] > 128 for row in tokenizer_rows),
    'roundtrip_mismatches': roundtrip_mismatches,
}
(work / 'tokenizer-audit.json').write_text(json.dumps(tokenizer_audit, indent=2) + '\\n')
assert tokenizer_audit['labels'] == 397
assert tokenizer_audit['over_128'] == 0
assert not tokenizer_audit['roundtrip_mismatches']
print(tokenizer_audit)
'''

    inputs = '''import tarfile
from huggingface_hub import hf_hub_download, snapshot_download

base_dir = Path(snapshot_download(
    repo_id=BASE_MODEL,
    revision=BASE_REVISION,
    local_dir=work / 'base-mixed-v3',
))

ehri_archive = hf_hub_download(
    'PiotrSty/ehri-pl-lines',
    'ehri-pl-lines-v1.tar.gz',
    repo_type='dataset',
    revision=EHRI_REVISION,
)
ehri_root = work / 'ehri-pl-lines-v1'
ehri_root.mkdir()
with tarfile.open(ehri_archive, 'r:gz') as archive:
    archive.extractall(ehri_root, filter='data')

evaluation_sets = {
    'historical-validation': corpus_root / 'validation',
    'real-lines-v1': repo / 'benchmarks/real-lines-v1/pairs',
    'ehri-test': ehri_root / 'test',
}
print({name: len(list(path.glob('*.png'))) for name, path in evaluation_sets.items()})
'''

    baseline = '''from dataclasses import asdict
from ocr.config import OcrConfig
from training.evaluate import evaluate

def evaluate_sets(model_path):
    results = {}
    for name, data_path in evaluation_sets.items():
        report = evaluate(
            data_path,
            str(model_path),
            batch_size=8,
            config=OcrConfig(device='cuda'),
        )
        results[name] = asdict(report)
        print(name, results[name])
        gc.collect()
        torch.cuda.empty_cache()
    return results

baseline_metrics = evaluate_sets(base_dir)
(work / 'baseline-metrics.json').write_text(
    json.dumps(baseline_metrics, indent=2) + '\\n',
    encoding='utf-8',
)
'''

    train = '''model_dir = work / 'historical-recognizer-v1-model'
subprocess.run([
    sys.executable, '-m', 'training.train_trocr_pl',
    '--train-dir', str(corpus_root / 'train'),
    '--val-dir', str(corpus_root / 'validation'),
    '--base', str(base_dir),
    '--output', str(model_dir),
    '--epochs', '12',
    '--batch-size', '4',
    '--gradient-accumulation-steps', '2',
    '--lr', '5e-5',
    '--lora-rank', '16',
    '--lora-alpha', '32',
    '--max-target-length', '128',
    '--no-4bit',
    '--seed', '42',
], check=True)
print((model_dir / 'selection.json').read_text())
print((model_dir / 'best_metrics.json').read_text())
'''

    candidate = '''candidate_metrics = evaluate_sets(model_dir)
(work / 'candidate-metrics.json').write_text(
    json.dumps(candidate_metrics, indent=2) + '\\n',
    encoding='utf-8',
)

def cer_delta(name):
    return candidate_metrics[name]['cer'] - baseline_metrics[name]['cer']

promotion = {
    'historical_validation_improved': cer_delta('historical-validation') < 0,
    'real_lines_regression_within_1pp': cer_delta('real-lines-v1') <= 0.01,
    'ehri_regression_within_2pp': cer_delta('ehri-test') <= 0.02,
    'tokenizer_roundtrip_clean': not tokenizer_audit['roundtrip_mismatches'],
    'no_target_truncation': tokenizer_audit['over_128'] == 0,
    'cer_deltas': {name: cer_delta(name) for name in evaluation_sets},
}
promotion['all_gates_passed'] = all(
    value for key, value in promotion.items()
    if key not in {'cer_deltas', 'all_gates_passed'}
)
(work / 'promotion-gates.json').write_text(
    json.dumps(promotion, indent=2) + '\\n',
    encoding='utf-8',
)
print(json.dumps(promotion, indent=2))
print('No model was published. Review the evidence ZIP first.')
'''

    evidence = '''import hashlib
import shutil

evidence_dir = work / 'evidence'
evidence_dir.mkdir()
files_to_copy = {
    corpus_root / 'report.json': 'corpus-report.json',
    corpus_root / 'manifest.jsonl': 'corpus-manifest.jsonl',
    corpus_root / 'quarantine.jsonl': 'corpus-quarantine.jsonl',
    repo / 'experiments/2026-09-24/historical-recognizer-v1/config.json': 'experiment-config.json',
    work / 'tokenizer-audit.json': 'tokenizer-audit.json',
    work / 'baseline-metrics.json': 'baseline-metrics.json',
    work / 'candidate-metrics.json': 'candidate-metrics.json',
    work / 'promotion-gates.json': 'promotion-gates.json',
    model_dir / 'run.json': 'training-run.json',
    model_dir / 'selection.json': 'selection.json',
    model_dir / 'best_metrics.json': 'best-metrics.json',
}
for source, name in files_to_copy.items():
    shutil.copyfile(source, evidence_dir / name)

environment = {
    'code_revision': CODE_REVISION,
    'base_model': BASE_MODEL,
    'base_revision': BASE_REVISION,
    'ehri_revision': EHRI_REVISION,
    'gpu': torch.cuda.get_device_name(0),
    'python': sys.version,
    'packages': {
        name: importlib.metadata.version(name)
        for name in ('torch', 'transformers', 'peft', 'accelerate', 'jiwer',
                     'huggingface_hub', 'Pillow', 'opencv-python-headless')
    },
    'raw_predictions_included': False,
    'images_or_labels_included': False,
    'model_weights_included': False,
}
(evidence_dir / 'environment.json').write_text(json.dumps(environment, indent=2) + '\\n')
checksums = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(evidence_dir.iterdir())
    if path.is_file()
}
(evidence_dir / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\\n')
evidence_zip = Path(shutil.make_archive(str(work / 'historical-recognizer-v1-evidence'), 'zip', evidence_dir))
print('Evidence:', evidence_zip)
print('Model remains local to this Colab session:', model_dir)
'''

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "colab": {"name": Path(target).name, "provenance": []},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "scope",
                "metadata": {},
                "source": [
                    "# Historical Polish recognizer v1\n",
                    "Choose a GPU runtime and click **Run all**. Do not upload a dataset manually. "
                    "The notebook downloads pinned public inputs, builds 258 train and 139 validation "
                    "line pairs, audits the tokenizer, trains a mixed-v3 specialization and downloads "
                    "a small evidence ZIP.\n",
                    "Historical spelling is preserved. Lines containing U+FFFD or private-use Unicode "
                    "are quarantined, not modernized. The geometry holdout and final test collections "
                    "are excluded from training. The notebook never publishes a model.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "install",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "%pip uninstall -y torchao\n",
                    "%pip install -q --upgrade transformers==4.57.6 peft==0.19.1 "
                    "accelerate==1.13.0 jiwer==4.0.0 huggingface_hub==0.36.2 "
                    "sentencepiece==0.2.1 pillow==11.3.0 opencv-python-headless==4.12.0.88\n",
                ],
            },
            code_cell("setup", setup),
            code_cell("corpus", corpus),
            code_cell("tokenizer", tokenizer_audit),
            code_cell("inputs", inputs),
            code_cell("baseline", baseline),
            code_cell("train", train),
            code_cell("candidate", candidate),
            code_cell("evidence", evidence),
            {
                "cell_type": "code",
                "id": "download",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": ["from google.colab import files\nfiles.download(str(evidence_zip))\n"],
            },
        ],
    }
    Path(target).write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build(Path(__file__).with_name("colab_historical_recognizer_v1.ipynb"))
