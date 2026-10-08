"""One GPU Colab: frozen test A, SOTA measurement, no manual upload, no training."""
import argparse
import json
from pathlib import Path
import re


IMPACT_REPOSITORY = "PiotrSty/impact-print-v2"
IMPACT_REVISION = "a2480fde6f15284701458ff370b81cce50dc5c2d"
IMPACT_SHA256 = "0a9ffa126029703726fc5883a8279ddccf15763c4fdd483ed9b8cc034bdb42f0"
IMPACT_PATH = "impact-print-v2-test.tar.gz"
DEFAULT_MODELS = ("paddlevl", "qwen3vl")


def build(target, revision, *, input_spec=None, models=DEFAULT_MODELS):
    import nbformat
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Use a complete already-pushed runtime revision")
    spec = dict(input_spec or {})
    spec.setdefault("repository", IMPACT_REPOSITORY)
    spec.setdefault("revision", IMPACT_REVISION)
    spec.setdefault("sha256", IMPACT_SHA256)
    spec.setdefault("path", IMPACT_PATH)
    for key in ("revision", "sha256"):
        width = 40 if key == "revision" else 64
        if not re.fullmatch(f"[0-9a-f]{{{width}}}", spec[key]):
            raise ValueError("Invalid frozen input revision/checksum")
    if not models or any(model not in ("paddlevl", "qwen3vl") for model in models):
        raise ValueError("Unknown model selection")
    if len(set(models)) != len(models):
        raise ValueError("Duplicate model selection")

    setup = f'''from pathlib import Path
import hashlib
import os
import subprocess
import sys
import venv
from datetime import datetime, timezone

CODE_REVISION = {revision!r}
IMPACT_REPOSITORY = {spec['repository']!r}
IMPACT_REVISION = {spec['revision']!r}
IMPACT_SHA256 = {spec['sha256']!r}
IMPACT_PATH = {spec['path']!r}
MODELS = {tuple(models)!r}
repo = Path('/content/OCR_engine-sota')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/PiotrStyla/OCR_engine.git', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', 'main'], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', CODE_REVISION], check=True)
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == CODE_REVISION
environment = Path('/content/slayer-sota-env')
python = environment/'bin/python'
if not python.exists():
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir',
                'transformers==4.57.6', 'accelerate==1.13.0', 'jiwer==4.0.0',
                'huggingface_hub==0.36.2', 'pytest==8.4.2', 'nbformat==5.10.4'], check=True)
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
subprocess.run([str(python), '-m', 'pytest', 'tests/test_sota_benchmark.py',
                '-q', '-p', 'no:cacheprovider'], cwd=repo, check=True)
assert subprocess.check_output([str(python), '-c', 'import torch; print(torch.cuda.is_available())'],
                               text=True).strip() == 'True', 'Wlacz GPU (T4) w runtime Colab.'
work = Path('/content/polocrbench-sota-v1-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
work.mkdir()
print('WORKDIR', work)
'''
    inputs = '''from huggingface_hub import hf_hub_download
import json as _json
archive = Path(hf_hub_download(IMPACT_REPOSITORY, IMPACT_PATH, repo_type='dataset', revision=IMPACT_REVISION))
assert hashlib.sha256(archive.read_bytes()).hexdigest() == IMPACT_SHA256, 'Checksum archiwum nie zgadza sie.'
staged = work/'benchmark'
subprocess.run([str(python), '-m', 'training.stage_impact_benchmark', '--archive', str(archive),
                '--output', str(staged)], cwd=repo, check=True)
rows = [_json.loads(line) for line in (staged/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
assert len(rows) == 36, f'Oczekiwano 36 stron testu A, jest {len(rows)}.'
print('INPUT_READY', staged, len(rows), 'stron')
'''
    paddle = '''subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', 'paddlex', 'paddlepaddle-gpu'], check=True)
versions = subprocess.check_output([str(python), '-c',
    'import importlib.metadata as m; print({p: m.version(p) for p in ("paddlex", "paddlepaddle", "paddleocr", "transformers", "torch")})'],
    text=True)
print('STACK', versions)
'''
    smoke = '''import json as _json
for model in MODELS:
    out = work/'smoke'/model
    command = [str(python), '-m', 'training.run_sota_benchmark', '--model', model,
               '--benchmark', str(staged), '--output', str(out), '--limit', '1']
    result = subprocess.run(command, cwd=repo)
    assert result.returncode == 0, f'Smoke test nie przeszedl dla {model}. Wklej blad; nie uruchamiaj pelnego pomiaru.'
    smoke_pred = _json.loads((out/'predictions.jsonl').read_text(encoding='utf-8').splitlines()[0])
    assert smoke_pred['status'] == 'ok' and smoke_pred['text'].strip(), f'Pusty wynik smoke dla {model}.'
    print('SMOKE_OK', model)
'''
    full = '''import json as _json
summary = {}
for model in MODELS:
    out = work/'runs'/model
    command = [str(python), '-m', 'training.run_sota_benchmark', '--model', model,
               '--benchmark', str(staged), '--output', str(out)]
    result = subprocess.run(command, cwd=repo)
    assert result.returncode == 0, f'Pomiar nie ukonczyl sie dla {model}.'
    score = _json.loads((out/'score.json').read_text(encoding='utf-8'))
    summary[model] = {'cer_micro': score['cer_micro'], 'wer_micro': score['wer_micro'],
                      'structure_similarity': score['structure_similarity'],
                      'errors_or_missing': score['errors_or_missing']}
    print('MODEL_RESULT', model, summary[model])
(work/'summary.json').write_text(_json.dumps(summary, indent=2), encoding='utf-8')
'''
    package = '''import json as _json
import shutil
from google.colab import files
evidence = work/'evidence'
evidence.mkdir()
for model in MODELS:
    for name in ('predictions.jsonl', 'score.json', 'run.json'):
        shutil.copyfile(work/'runs'/model/name, evidence/f'{model}-{name}')
shutil.copyfile(staged/'verification.json', evidence/'staged-verification.json')
shutil.copyfile(work/'summary.json', evidence/'summary.json')
(evidence/'receipt.json').write_text(_json.dumps({
    'protocol_version': 'polocrbench-sota-measurement-v1', 'code_revision': CODE_REVISION,
    'impact_repository': IMPACT_REPOSITORY, 'impact_revision': IMPACT_REVISION,
    'impact_sha256': IMPACT_SHA256, 'models': list(MODELS),
    'measurement_only': True, 'training_performed': False}, indent=2), encoding='utf-8')
zip_path = work/'polocrbench-sota-measurement-v1-evidence.zip'
shutil.make_archive(str(zip_path.with_suffix('')), 'zip', evidence)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    shutil.copyfile(zip_path, '/content/drive/MyDrive/'+zip_path.name)
    print('KOPIA_ZAPASOWA', '/content/drive/MyDrive/'+zip_path.name)
except Exception as error:
    print('Brak kopii na Dysku:', error)
print('Gotowy ZIP:', zip_path)
files.download(str(zip_path))
'''
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(
            '# PolOCRBench test A: pomiar SOTA v1\n\n'
            'Wybierz **GPU T4** i **Uruchom wszystko**. Niczego nie wgrywaj.\n\n'
            'Zamrożone 36 stron testu A (IMPACT history_print), ścieżka zero-shot: '
            + ", ".join(models) + '. Plik promptu i szablony są niezmienne '
            '(`benchmarks/polocrbench/prompts/zero_shot_prompt_v1.md`).\n\n'
            'To **pomiar**, nie trening: bez strojenia wag, adapterów, korekt i bez '
            'promocji jakichkolwiek artefaktów. Wynik: `polocrbench-sota-measurement-v1-evidence.zip` '
            '(predykcje, metryki, wersje, paragon wejścia) + kopia na Dysku.'),
        nbformat.v4.new_markdown_cell('## 1. Środowisko, przypięty kod i testy'),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_markdown_cell(
            '## 2. Zamrożone wejście: 36 stron testu A\n\n'
            'Archiwum `impact-print-v2-test` z przypiętą rewizją i sumą SHA-256; '
            'staging weryfikuje sumy wszystkich obrazów względem zamrożonego manifestu.'),
        nbformat.v4.new_code_cell(inputs),
        nbformat.v4.new_markdown_cell(
            '## 3. Stos modelowy GPU\n\n'
            'Wersje faktycznie uruchomione trafiają do `run.json` każdego pomiaru.'),
        nbformat.v4.new_code_cell(paddle),
        nbformat.v4.new_markdown_cell('## 4. Smoke test (1 strona na model)'),
        nbformat.v4.new_code_cell(smoke),
        nbformat.v4.new_markdown_cell(
            '## 5. Pełny pomiar\n\n'
            'Każdy model w osobnym procesie (pamięć GPU zwalniana między przebiegami).'),
        nbformat.v4.new_code_cell(full),
        nbformat.v4.new_markdown_cell(
            '## 6. Podsumowanie i paczka dowodowa\n\n'
            'Przekaż `polocrbench-sota-measurement-v1-evidence.zip` do audytu. '
            'Kopia na Dysku jest zapasowa; nie zamykaj sesji przed pobraniem.'),
        nbformat.v4.new_code_cell(package)], metadata={
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
            'colab': {'name': 'colab_sota_testA.ipynb', 'provenance': []}})
    nbformat.validate(notebook)
    nbformat.write(notebook, Path(target))
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output", default="notebooks/colab_sota_testA.ipynb")
    parser.add_argument("--input-receipt")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = parser.parse_args()
    spec = json.loads(Path(args.input_receipt).read_text(encoding="utf-8")) if args.input_receipt else None
    build(args.output, args.code_revision, input_spec=spec, models=tuple(args.models.split(",")))
