"""Build one portable Colab with isolated model environments and embedded runner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.build_historical_recognizer_colab import code_cell


ROOT = Path(__file__).resolve().parents[1]
BASE_CODE_REVISION = "ad8c3908a41949719a29ea259c5d286e078f788f"
CONFIG_PATH = ROOT / "experiments/2026-10-02/full-page-pilot-v1/config.json"

ENVIRONMENT_SETUP_SOURCE = '''import os
from pathlib import Path
import subprocess
import sys
import venv


def prepare_pilot_environment(environment):
    environment = Path(environment)
    # Colab can lack ensurepip; repair a partial venv without deleting its files.
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    prefix = subprocess.check_output(
        [str(python), '-c', 'import sys; print(sys.prefix)'], text=True).strip()
    if Path(prefix).resolve() != environment.resolve():
        raise RuntimeError('Refusing installation outside the requested model environment.')
    return python


def pilot_pip_command(python, *arguments):
    # Host pip manages the target interpreter even when its venv has no local pip.
    return [sys.executable, '-m', 'pip', '--python', str(python), *arguments]
'''


def markdown(identifier, text):
    return {"cell_type": "markdown", "id": identifier, "metadata": {}, "source": [text]}


def build(target, config_path=CONFIG_PATH):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    runner = (ROOT / "training/full_page_pilot.py").read_text(encoding="utf-8")
    runner_hash = hashlib.sha256(runner.encode()).hexdigest()
    setup = f'''import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

CONFIG = {config!r}
BASE_CODE_REVISION = {BASE_CODE_REVISION!r}
RUNNER_SOURCE = {runner!r}
RUNNER_SHA256 = {runner_hash!r}
PAGES = CONFIG['default_inference_pages']  # 2-page smoke; increase only after reviewing its evidence.
WORK = Path('/content') / CONFIG.get('work_name', 'slayer-full-page-pilot-v1')
WORK.mkdir(parents=True, exist_ok=True)
config_path = WORK / 'config.json'
if config_path.exists():
    assert json.loads(config_path.read_text()) == CONFIG, 'Use a new WORK for changed configuration.'
config_path.write_text(json.dumps(CONFIG, indent=2), encoding='utf-8')
runner_path = WORK / 'full_page_pilot.py'
assert hashlib.sha256(RUNNER_SOURCE.encode()).hexdigest() == RUNNER_SHA256
if runner_path.exists():
    assert hashlib.sha256(runner_path.read_bytes()).hexdigest() == RUNNER_SHA256, 'Use a new WORK for changed runner.'
runner_path.write_text(RUNNER_SOURCE, encoding='utf-8', newline='\\n')

repo = Path('/content/OCR_engine-full-page-pilot')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/PiotrStyla/OCR_engine.git', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', BASE_CODE_REVISION], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', BASE_CODE_REVISION], check=True)
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == BASE_CODE_REVISION, 'Dedicated checkout has changed; use a fresh session.'
sys.path.insert(0, str(repo))
spec = importlib.util.spec_from_file_location('full_page_pilot', runner_path)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
pilot.write_json(WORK / 'code-provenance.json', {{
    'base_code_revision': BASE_CODE_REVISION, 'embedded_runner_sha256': RUNNER_SHA256,
    'gpu_inference_previously_verified': False,
    'environment_bootstrap': 'venv-without-pip-host-pip-python-v2',
}})
print('WORK:', WORK)
print('PAGES:', PAGES)
'''
    inputs = '''dataset = WORK / 'dataset'
audit = pilot.stage(CONFIG, dataset, PAGES)
print(json.dumps({key: value for key, value in audit.items() if key != 'results'}, indent=2))
assert audit['gold_pages'] == 0
print('SOURCE REFERENCES ARE UNREVIEWED. These scores cannot select a production model.')
'''
    environments = ENVIRONMENT_SETUP_SOURCE + '''
import importlib.metadata
import torch
assert torch.cuda.is_available(), 'Colab: Runtime > Change runtime type > GPU.'
assert tuple(int(part) for part in importlib.metadata.version('pip').split('.')[:2]) >= (22, 3), 'Host pip >= 22.3 is required for --python.'
print('GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__)
python_by_engine = {}
for engine, model_spec in CONFIG['models'].items():
    environment = Path('/content/slayer-full-page-pilot-envs') / engine
    python = prepare_pilot_environment(environment)
    # Install into isolated venvs, never replace Transformers in the notebook kernel.
    subprocess.run(pilot_pip_command(python, 'install', '--no-cache-dir',
                                    *model_spec['packages']), check=True)
    required = {item.split('==')[0]: item.split('==')[1] for item in model_spec['packages']}
    check = ('import importlib.metadata,json; required=' + repr(required) +
             '; actual={k:importlib.metadata.version(k) for k in required}; '
             'assert actual==required,(actual,required); '
             'from transformers.generation import GenerationMixin; '
             'print(json.dumps(actual))')
    if engine == 'ovis-ocr2':
        check += '; from transformers import AutoModelForImageTextToText, AutoProcessor'
    subprocess.run([str(python), '-c', check], check=True)
    python_by_engine[engine] = python
'''
    inference = '''process_env = {**os.environ, 'PYTHONPATH': str(repo), 'TOKENIZERS_PARALLELISM': 'false'}
process_results = {}
try:
    for engine, python in python_by_engine.items():
        destination = WORK / 'predictions' / engine
        destination.mkdir(parents=True, exist_ok=True)
        with (WORK / (engine + '.log')).open('w', encoding='utf-8') as log:
            process = subprocess.Popen([
                str(python), '-u', str(runner_path), 'worker', '--config', str(config_path),
                '--engine', engine, '--inputs', str(dataset / 'inference-inputs.jsonl'),
                '--output', str(destination),
            ], env=process_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)
                    log.flush()
                process_results[engine] = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        print(engine, 'exit status:', process_results[engine])
finally:
    pilot.write_json(WORK / 'process-results.json', process_results)
    evidence_zip = pilot.package(WORK)
    print('PARTIAL_EVIDENCE_READY:', evidence_zip)
'''
    evaluation = '''summary = pilot.score(dataset / 'manifest.jsonl', WORK / 'predictions', WORK / 'scores')
for label, report in summary['reports'].items():
    print(f"{label}: CER={report['cer_micro']:.2%}, WER={report['wer_micro']:.2%}, "
          f"errors={report['errors_or_missing']}, token_limit_pages={report['token_limit_pages']}, "
          f"oracle={report['oracle_geometry']}")
print('DEVELOPMENT DIAGNOSTICS ONLY. SOTA claim:', summary['sota_claim'])

import matplotlib.pyplot as plt
labels = list(summary['reports'])
values = [summary['reports'][label]['cer_micro'] * 100 for label in labels]
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(labels, values)
ax.set_xlim(left=0)
ax.set_xlabel('CER (%) - unreviewed source references')
ax.set_title(f'Full-page pilot: {summary["pages"]} development pages; oracle is diagnostic')
for index, value in enumerate(values):
    ax.text(value, index, f' {value:.1f}%', va='center')
fig.tight_layout()
fig.savefig(WORK / 'cer-diagnostic.png', dpi=140)
plt.show()
'''
    download = '''from google.colab import files
evidence_zip = pilot.package(WORK)
print('Evidence:', evidence_zip)
print('Includes raw predictions, metrics, input provenance, errors and package versions; no scans or weights.')
files.download(str(evidence_zip))
'''
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "colab": {"name": Path(target).name, "provenance": []},
                     "accelerator": "GPU"},
        "cells": [
            markdown("scope", "# SLAYER-OCR: full-page pilot\n\n"
                     "Select a GPU runtime, then **Run all**. No uploads or HF token. Default: two historical development pages. "
                     "Our mixed-v3 pipeline is compared with OvisOCR2; source PAGE regions with automatic line segmentation are a separate diagnostic.\n\n"
                     "Source transcriptions are **not verified gold**. No model promotion or SOTA claim is possible from this pilot. "
                     "The 36-page historical test is excluded. Historical spelling is preserved. "
                     f"Ovis image budget: {config['models']['ovis-ocr2']['max_pixels']:,} pixels; processor resizing is recorded. "
                     "This memory profile still requires GPU validation. Failed pages are not quality scores.\n"),
            markdown("dependencies-heading", "## 1. Lightweight scoring dependencies\n"),
            {"cell_type": "code", "id": "dependencies", "metadata": {}, "execution_count": None,
             "outputs": [], "source": ["%pip install jiwer==4.0.0 markdown-it-py==4.0.0 pillow==11.3.0\n"]},
            markdown("setup-heading", "## 2. Frozen code and two-page scope\n"
                     "The runner is embedded with a SHA-256; existing OCR modules come from a pinned public Git commit.\n"),
            code_cell("setup", setup),
            markdown("data-heading", "## 3. Automatic HF download and reference audit\n"),
            code_cell("inputs", inputs),
            markdown("environments-heading", "## 4. Separate model environments\n"
                     "Each engine has its own Transformers installation and subprocess; the notebook kernel is not used for model loading. "
                     "Environment creation does not require ensurepip. Rerunning this cell repairs partial environment creation without deleting data.\n"),
            code_cell("environments", environments),
            markdown("inference-heading", "## 5. Inference with per-page checkpoints\n"
                     "Rerunning preserves recorded pages, including errors. To retry failed pages, use a new WORK directory. "
                     "If a later cell fails, the download cell still packages available diagnostics.\n"),
            code_cell("inference", inference),
            markdown("results-heading", "## 6. Provisional CER/WER\n"
                     "Missing or failed predictions are scored as empty. Raw Markdown is retained; only presentation markup is projected to text. "
                     "Source-region crops are diagnostic annotation-assisted inputs, not automatic layout predictions. "
                     "Tables, reading-order accuracy and hallucination rates require a later reviewed benchmark and are not claimed here.\n"),
            code_cell("evaluation", evaluation),
            markdown("download-heading", "## 7. Download evidence\n"
                     "Run this cell even after an inference or scoring error. Send back the resulting ZIP.\n"),
            code_cell("download", download),
        ],
    }
    Path(target).write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build(ROOT / 'training/colab_full_page_pilot_t4_v2.ipynb',
          ROOT / 'experiments/2026-10-03/full-page-pilot-t4-v2/config.json')
