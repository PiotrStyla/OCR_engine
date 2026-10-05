"""Build one pinned Colab for training-source OCR teacher proposals."""
import argparse
import json
from pathlib import Path
import re

from training.build_full_page_pilot_colab import ENVIRONMENT_SETUP_SOURCE, markdown
from training.build_historical_recognizer_colab import code_cell


def build(target, config_path, code_revision):
    if not re.fullmatch('[0-9a-f]{40}', code_revision):
        raise ValueError('A published full code SHA is required')
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    archive_name = config.get('evidence_archive_name', 'recognizer-data-v3-teacher-evidence.zip')
    setup = f'''import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

CONFIG = {config!r}
CODE_REVISION = {code_revision!r}
WORK = Path('/content') / ('recognizer-data-v3-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
WORK.mkdir(parents=True)
repo = Path('/content/OCR_engine-recognizer-data-v3')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/PiotrStyla/OCR_engine.git', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', 'main'], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', CODE_REVISION], check=True)
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == CODE_REVISION
sys.path.insert(0, str(repo))
from training.recognizer_data_pilot import stage, combine, package, ENGINES
from training.full_page_pilot import digest, fetch, write_json, read_rows
EVIDENCE_ARCHIVE_NAME = {archive_name!r}
write_json(WORK / 'config.json', CONFIG)
write_json(WORK / 'code-provenance.json', {{'code_revision': CODE_REVISION,
    'runner_sha256': digest(repo / 'training/recognizer_data_pilot.py'),
    'qwen_runner_sha256': digest(repo / 'training/full_page_pilot.py'),
    'reference_text_sent_to_models': False, 'gpu_execution_validated_locally': False}})
print('WORK:', WORK)
'''
    inputs = '''archive = fetch(CONFIG, CONFIG['dataset']['path'], WORK / 'input.zip', CONFIG['dataset']['archive_sha256'])
dataset = WORK / 'dataset'
audit = stage(CONFIG, archive, dataset)
write_json(WORK / 'staging-report.json', audit)
print(json.dumps(audit, indent=2))
'''
    environment = ENVIRONMENT_SETUP_SOURCE + '''
import traceback
python = None
bootstrap = {'status': 'error'}
try:
    import torch
    assert torch.cuda.is_available(), 'Select a GPU runtime in Colab.'
    candidate_python = prepare_pilot_environment('/content/recognizer-data-v3-env')
    command = pilot_pip_command(candidate_python, 'install', '--no-cache-dir', *CONFIG['packages'])
    installed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (WORK / 'installation.log').write_text(installed.stdout, encoding='utf-8')
    assert installed.returncode == 0, installed.stdout[-6000:]
    required = {item.split('==')[0]: item.split('==')[1] for item in CONFIG['packages']}
    preflight = ('import importlib.metadata,json,torch; required=' + repr(required) +
        '; actual={k:importlib.metadata.version(k) for k in required}; '
        'assert actual==required,(actual,required); '
        'from transformers import Qwen3VLForConditionalGeneration,TrOCRProcessor,VisionEncoderDecoderModel; '
        'from transformers.generation import GenerationMixin; '
        'assert torch.cuda.is_available(); '
        'print(json.dumps({"packages":actual,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0)}))')
    checked = subprocess.run([str(candidate_python), '-c', preflight], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
    (WORK / 'preflight.log').write_text(checked.stdout, encoding='utf-8')
    assert checked.returncode == 0, checked.stdout[-6000:]
    python = candidate_python
    bootstrap = {'status': 'ok', 'interpreter': str(python), 'gpu': torch.cuda.get_device_name(0)}
except Exception:
    bootstrap['error'] = traceback.format_exc()
    print(bootstrap['error'])
write_json(WORK / 'bootstrap.json', bootstrap)
print('BOOTSTRAP:', bootstrap['status'])
'''
    inference = '''runs = {}
for engine in ENGINES:
    if python is None:
        runs[engine] = {'status': 'not-started', 'reason': 'environment preflight failed'}
        continue
    command = [str(python), '-u', '-m', 'training.recognizer_data_pilot', '--config', str(WORK / 'config.json'),
               '--engine', engine, '--inputs', str(dataset / 'inference-inputs.jsonl'),
               '--output', str(WORK / 'predictions' / engine)]
    env = dict(os.environ, PYTHONPATH=str(repo), TOKENIZERS_PARALLELISM='false')
    with (WORK / (engine + '.log')).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='')
        code = process.wait()
    runs[engine] = {'exit_code': code, 'status': 'finished' if code == 0 else 'failed'}
    write_json(WORK / 'worker-exits.json', runs)
write_json(WORK / 'worker-exits.json', runs)
'''
    consensus = '''try:
    report = combine(CONFIG, dataset, WORK / 'predictions', WORK / 'combined')
    print(json.dumps(report, ensure_ascii=False, indent=2))
except Exception:
    import traceback
    write_json(WORK / 'combine-error.json', {'error': traceback.format_exc()})
    print(traceback.format_exc())
result_archive = package(WORK)
print('EVIDENCE:', result_archive)
'''
    download = '''from google.colab import files
result_archive = package(WORK)
files.download(str(result_archive))
'''
    if 'evidence_archive_name' in config:
        consensus = consensus.replace('package(WORK)', 'package(WORK, EVIDENCE_ARCHIVE_NAME)')
        download = download.replace('package(WORK)', 'package(WORK, EVIDENCE_ARCHIVE_NAME)')
    selection_note = ('To kolejna partia nowych linii: wszystkie wczesniej przejrzane ID i wycinki sa wykluczone. '
        if config.get('selection') else '')
    cells = [markdown('scope', '# DATA ENGINE: recognizer v3 / teacher pilot\n\n'
        'Wybierz GPU i uruchom wszystkie komorki. Dane pobiora sie automatycznie z HF; nie wgrywaj zadnych ZIP-ow. '
        f"{config['dataset']['lines']} wycinki pochodza ze zbioru treningowego, nie z 15 stron walidacji ani testu. "
        + selection_note +
        'Qwen i TrOCR pracuja kolejno w osobnych procesach, bez etykiet referencyjnych. '
        'Notebook nie trenuje modelu: przygotowuje propozycje do kontroli obrazu, granic linii i pisowni. '
        'Zgodnosc modeli nie stanowi prawdy ani zgody na trening. Zachowujemy historyczne znaki, bez modernizacji. '
        'Profil wycinkow nie zostal jeszcze sprawdzony na GPU. W razie bledow ostatnia komorka pobiera logi i czesciowe wyniki.'),
        code_cell('setup', setup), code_cell('inputs', inputs), code_cell('environment', environment),
        code_cell('inference', inference), code_cell('consensus', consensus),
        markdown('return', f'## Pobierz wynik\n\nPrzekaz plik `{archive_name}`. '
                 'Nie zawiera wag ani skanow; obrazy mozna odtworzyc z przypietego pakietu HF. '
                 'Bledy, brakujace odpowiedzi i limity pozostaja w raporcie.'), code_cell('download', download)]
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'name': Path(target).name}}}
    Path(target).write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.output, args.config, args.code_revision)
