"""Build one no-upload Colab comparing Qwen full-page OCR with retained Ovis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.build_full_page_pilot_colab import ENVIRONMENT_SETUP_SOURCE, markdown
from training.build_historical_recognizer_colab import code_cell


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT/'experiments/2026-10-03/full-page-comparison-v5/config.json'


def build(target, code_revision, config_path=CONFIG_PATH):
    if len(code_revision) != 40 or any(char not in '0123456789abcdef' for char in code_revision):
        raise ValueError('Use a full, already-published Git commit')
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    relative_config = config_path.relative_to(ROOT).as_posix()
    setup = f'''import json
import os
from pathlib import Path
import subprocess
import sys

CODE_REVISION = {code_revision!r}
CONFIG = {config!r}
repo = Path('/content')/('OCR_engine-' + CONFIG['work_name'])
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/PiotrStyla/OCR_engine.git', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', 'main'], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', CODE_REVISION], check=True)
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == CODE_REVISION
sys.path.insert(0, str(repo))
from training import full_page_comparison as comparison
from training.full_page_pilot import digest, write_json
assert json.loads((repo/{relative_config!r}).read_text()) == CONFIG
WORK = Path('/content')/CONFIG['work_name']
WORK.mkdir(parents=True, exist_ok=True)
config_path = WORK/'config.json'
if config_path.exists():
    assert json.loads(config_path.read_text()) == CONFIG, 'Use a fresh session for a different configuration.'
write_json(config_path, CONFIG)
write_json(WORK/'code-provenance.json', {{
    'code_revision': CODE_REVISION,
    'runner_sha256': digest(repo/'training/full_page_pilot.py'),
    'comparison_sha256': digest(repo/'training/full_page_comparison.py'),
    'gpu_run_previously_verified': False,
}})
print('WORK:', WORK)
'''
    inputs = '''dataset = WORK/'dataset'
staged = comparison.stage_bundle(CONFIG, dataset)
print(json.dumps(staged, indent=2))
assert staged['pages'] == 15 and staged['gold_pages'] == 0
print('Model inputs contain images only, without reference text, notes or source boxes.')
'''
    environment = ENVIRONMENT_SETUP_SOURCE + '''
import importlib.metadata
import torch

assert torch.cuda.is_available(), 'Colab: Runtime > Change runtime type > GPU.'
assert tuple(int(p) for p in importlib.metadata.version('pip').split('.')[:2]) >= (22, 3)
spec = CONFIG['models']['qwen3-vl-4b']
python = prepare_pilot_environment('/content/' + CONFIG['work_name'] + '-env')
command = pilot_pip_command(python, 'install', '--no-cache-dir', *spec['packages'])
installation = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(WORK/'installation.log').write_text(installation.stdout, encoding='utf-8')
print(installation.stdout[-12000:])
installation.check_returncode()
required = {item.split('==')[0]: item.split('==')[1] for item in spec['packages']}
probe = ('import importlib.metadata,json,torch; required=' + repr(required) +
    '; actual={k:importlib.metadata.version(k) for k in required}; '
    'assert actual==required,(actual,required); '
    'from transformers import Qwen3VLForConditionalGeneration,AutoProcessor,BitsAndBytesConfig; '
    'from transformers.generation import GenerationMixin; import bitsandbytes; '
    'assert torch.cuda.is_available(); '
    'print(json.dumps(dict(packages=actual,gpu=torch.cuda.get_device_name(0),torch=torch.__version__)))')
preflight = subprocess.run([str(python), '-c', probe], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
(WORK/'preflight.log').write_text(preflight.stdout, encoding='utf-8')
print(preflight.stdout)
preflight.check_returncode()
write_json(WORK/'runtime.json', {'gpu': torch.cuda.get_device_name(0),
    'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': sys.version,
    'environment': str(python), 'kernel_transformers_replaced': False})
'''
    inference = '''destination = WORK/'predictions/qwen3-vl-4b'
destination.mkdir(parents=True, exist_ok=True)
command = [str(python), '-u', '-m', 'training.full_page_pilot', 'worker',
    '--config', str(config_path), '--engine', 'qwen3-vl-4b',
    '--inputs', str(dataset/'inference-inputs.jsonl'), '--output', str(destination)]
process = None
try:
    with (WORK/'qwen3-vl-4b.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command,
            env={**os.environ, 'PYTHONPATH': str(repo), 'TOKENIZERS_PARALLELISM': 'false'},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end='')
                log.write(line)
                log.flush()
            exit_status = process.wait()
            write_json(WORK/'process-result.json', {'exit_status': exit_status})
            print('Worker exit status:', exit_status)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
finally:
    print('PARTIAL_EVIDENCE_READY:', comparison.package_comparison(WORK))
'''
    evaluation = '''summary = comparison.compare(CONFIG, dataset,
    WORK/'predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl', WORK/'scores')
print('Development diagnostics only; not gold, teacher approval or SOTA.')
for label, report in summary['reports'].items():
    print(f"{label}: CER={report['cer_micro']:.2%}, WER={report['wer_micro']:.2%}, "
          f"errors/missing={report['errors_or_missing']}, EOS={report['eos_pages']}, "
          f"capped={report['token_limit_pages']}")
from IPython.display import display, Markdown
import csv
with (WORK/'scores/per-page.csv').open(encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
table = '| Silnik | Strona | CER | Koniec | Limit |\\n|---|---|---:|---|---|\\n'
for row in rows:
    table += f"| {row['engine']} | {row['id']} | {float(row['cer']):.2%} | {row['finish_reason']} | {row['token_limit_reached']} |\\n"
display(Markdown(table))
'''
    chart = '''import matplotlib.pyplot as plt
labels = list(summary['reports'])
values = [summary['reports'][label]['cer_micro']*100 for label in labels]
fig, ax = plt.subplots(figsize=(9, 3))
ax.barh(labels, values, color=['#53758e', '#b25346'])
ax.set_xlim(0, max(values+[1])*1.2)
ax.set_xlabel('CER (%)')
ax.set_title('15 stron: prowizoryczne referencje v2, wszystkie wyniki')
for index,value in enumerate(values):
    ax.text(value, index, f' {value:.2f}%', va='center')
fig.tight_layout()
fig.savefig(WORK/'cer-comparison.png', dpi=140)
plt.show()
'''
    download = '''from google.colab import files
from training.full_page_comparison import package_comparison
archive = package_comparison(WORK)
print('Pobierz i prześlij do Codex:', archive.name)
print('ZIP zawiera predykcje, metryki, logi i pochodzenie; bez skanów i wag.')
files.download(str(archive))
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
                     'colab': {'name':Path(target).name, 'provenance':[]}, 'accelerator':'GPU'},
        'cells': [
            markdown('scope', '# OCR całych stron: Qwen kontra zapisany Ovis\n\n'
                'Wybierz GPU (np. T4), następnie **Uruchom wszystko**. Niczego nie wgrywaj. '
                'Notebook pobierze dokładny komplet 15 stron v2 z HF. Ovis nie jest uruchamiany ponownie.\n\n'
                'To nowy, jeszcze niewykonany test GPU. Referencje są prowizoryczne, nie gold. '
                'Zachowujemy dawną pisownię, wszystkie błędy i wyjścia na limicie. '
                'Prompt Qwen obejmuje tekst peryferyjny, którego zakres w referencjach nie jest jeszcze uzgodniony. '
                'Porównujemy profile inferencji, nie izolowany wpływ samych wag.\n'),
            markdown('deps-heading', '## 1. Lekkie zależności metryk\n'),
            code_cell('dependencies', "import subprocess\nimport sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', 'jiwer==4.0.0', 'markdown-it-py==4.0.0', 'pillow==11.3.0', 'matplotlib==3.10.8'], check=True)\n"),
            markdown('setup-heading', '## 2. Przypięty kod i konfiguracja\n'), code_cell('setup', setup),
            markdown('data-heading', '## 3. Pobranie danych i kontrola sum\n'), code_cell('inputs', inputs),
            markdown('environment-heading', '## 4. Oddzielne środowisko modelu\n'
                'FP16 + NF4 4-bit + SDPA. Bez ensurepip i podmieniania Transformers w kernelu.\n'),
            code_cell('environment', environment),
            markdown('inference-heading', '## 5. Odczyt wszystkich 15 stron\n'
                'Wyniki zapisują się po każdej stronie. Ponowne uruchomienie zachowuje także błędy; '
                'nowa sesja służy do ponowienia całego eksperymentu.\n'), code_cell('inference', inference),
            markdown('evaluation-heading', '## 6. CER/WER i diagnostyka\n'
                'Brakujące i nieudane predykcje są liczone jako pusty tekst. '
                'Kolejność przestrzenna, pominięcia regionów i halucynacje wymagają osobnej oceny. '
                'Czas i VRAM są raportowane, ale nieporównywalne bez wspólnego runtime.\n'),
            code_cell('evaluation', evaluation), code_cell('chart', chart),
            markdown('download-heading', '## 7. Pobierz dowody\n'
                'Uruchom tę komórkę również po błędzie instalacji, inferencji lub metryk, jeśli krok 2 się zakończył. '
                'Prześlij plik **full-page-comparison-v5-evidence.zip**.\n'), code_cell('download', download)]}
    Path(target).write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args()
    build(ROOT/'training/colab_full_page_comparison_v5.ipynb', args.code_revision)
