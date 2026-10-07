"""Generate one pinned Colab notebook with automatic inputs and one result ZIP."""
import argparse
from pathlib import Path
import re

DATA_REVISION = '3a301fd12a6272b28774d1888213ddf528b47f24'
DATA_SHA256 = '3e39b7d02b90a0a741adde506c0467698c46fe86c82846d3cc0e648eebaae32e'


def build(target, code_revision, version='v1'):
    import nbformat

    if not re.fullmatch('[0-9a-f]{40}', code_revision):
        raise ValueError('Use the complete, already pushed runtime commit SHA')
    if version not in ('v1', 'v2'):
        raise ValueError('Unknown notebook version')
    stem = 'recognizer-reviewed-colab-' + version
    setup = f'''from pathlib import Path
import os
import subprocess
import sys
import venv

CODE_REVISION = {code_revision!r}
DATA_REVISION = {DATA_REVISION!r}
DATA_SHA256 = {DATA_SHA256!r}
repo = Path('/content/OCR_engine-reviewed')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/PiotrStyla/OCR_engine.git', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', 'main'], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', CODE_REVISION], check=True)
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == CODE_REVISION
environment = Path('/content/slayer-reviewed-training-env')
python = environment/'bin/python'
if not python.exists():
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
subprocess.run([str(python), '-c', "import torch; assert torch.cuda.is_available(), 'Wybierz GPU T4 w ustawieniach Colaba i uruchom wszystko.'; print('GPU:', torch.cuda.get_device_name(0))"], check=True)
torch_release = subprocess.check_output([str(python), '-c', "import torch; print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))"], text=True).strip()
torchao_versions = {{'2.10': '0.16.0', '2.11': '0.17.0'}}
assert torch_release in torchao_versions, f'Niezweryfikowana wersja Torch: {{torch_release}}. Nie zmieniam CUDA.'
packages = ['transformers==4.57.6', 'tokenizers==0.22.2', 'huggingface_hub==0.36.2',
            'peft==0.19.1', 'accelerate==1.13.0', 'jiwer==4.0.0',
            'pillow==11.3.0', 'sentencepiece==0.2.1']
subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', *packages], check=True)
subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', '--no-deps',
                'torchao=='+torchao_versions[torch_release]], check=True)
subprocess.run([str(python), '-c', "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; from transformers.generation import GenerationMixin; import peft, jiwer; print('TRAINING_IMPORTS_OK')"], check=True)
subprocess.run([str(python), '-m', 'training.preflight_reviewed_recognizer',
                '--output', str(environment/'adapter-preflight.json')], cwd=repo, check=True)
print('Gotowe. Dane zostana pobrane automatycznie; niczego nie wgrywaj.')
'''
    if version == 'v2':
        setup = setup.replace('/content/OCR_engine-reviewed', '/content/OCR_engine-reviewed-v2')
        setup = setup.replace('/content/slayer-reviewed-training-env', '/content/slayer-reviewed-training-v2-env')
        setup = setup.replace("'sentencepiece==0.2.1'", "'sentencepiece==0.2.1', 'pytest==8.4.2'")
        setup += '''os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
subprocess.run([str(python), '-m', 'pytest', 'tests/test_training_protocol.py',
                '-q'], cwd=repo, check=True)
os.environ['HF_HUB_DISABLE_XET'] = '1'
'''
    training = '''from google.colab import files

RESULT_DIRECTORY = None
EVIDENCE_ZIP = None
command = [str(python), '-m', 'training.run_reviewed_recognizer_colab',
           '--dataset-revision', DATA_REVISION, '--dataset-sha256', DATA_SHA256]
process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end='', flush=True)
    if line.startswith('RESULT_DIRECTORY '):
        RESULT_DIRECTORY = Path(line.strip().split(' ', 1)[1])
    elif line.startswith('EVIDENCE_ZIP '):
        EVIDENCE_ZIP = Path(line.strip().split(' ', 1)[1])
return_code = process.wait()
if return_code:
    if EVIDENCE_ZIP is not None and EVIDENCE_ZIP.is_file():
        files.download(str(EVIDENCE_ZIP))
    raise RuntimeError('Trening nie zakonczyl sie. Zapisano ZIP diagnostyczny; nie jest to model.')
assert RESULT_DIRECTORY is not None, 'Brak potwierdzenia zakonczenia treningu.'
print('TRENING_ZAKONCZONY', RESULT_DIRECTORY)
'''
    if version == 'v2':
        training = training.replace("'training.run_reviewed_recognizer_colab',\n           '--dataset-revision', DATA_REVISION, '--dataset-sha256', DATA_SHA256",
                                    "'training.run_reviewed_recognizer_colab_v2'")
    download = '''assert RESULT_DIRECTORY is not None, 'Najpierw musi zakonczyc sie komorka treningu.'
result_zip = RESULT_DIRECTORY/'recognizer-reviewed-colab-v1-result.zip'
assert result_zip.is_file(), 'Brak kompletnego archiwum wyniku.'
print('Model i raport w jednym pliku:', result_zip)
files.download(str(result_zip))
'''
    title = ('# TrOCR V2: ochrona zwyklego druku\n\n'
        'Wybierz **GPU T4** i **Uruchom wszystko**. Nie wgrywaj zadnych plikow.\n\n'
        'Trzy warianty po 3 epoki: LR 1e-5 / replay 500, LR 3e-6 / replay 500, '
        'LR 3e-6 / replay 2000. Wszystkie zaczynaja od tej samej przypietej bazy. '
        '70 linii historycznych w treningu; kontrola: 9 historycznych i 75 zwyklych.\n\n'
        'Kandydat musi poprawic historyczny i laczny CER, bez regresji CER/WER zwyklego '
        'druku i bez dodatkowych znakow zastepczych. W przeciwnym razie pozostaje baza. '
        'Stara pisownia pozostaje bez zmian. To wybor na zbiorze development, nie dowod SOTA.\n\n'
        'Raport pobierze sie automatycznie. Dla duzego ZIP-a z wagami jest opcjonalny zapis '
        'na Dysku Google w ostatniej komorce. Nie wymaga to ponownego treningu.')
    if version == 'v2':
        download = '''assert RESULT_DIRECTORY is not None, 'Najpierw musi zakonczyc sie trening.'
assert EVIDENCE_ZIP is not None and EVIDENCE_ZIP.is_file(), 'Brak raportu.'
result_zip = RESULT_DIRECTORY/'recognizer-reviewed-colab-v2-result.zip'
assert result_zip.is_file(), 'Brak kompletnego archiwum wyniku.'
import json
selection = json.loads((RESULT_DIRECTORY/'evidence/selection.json').read_text())
print('Wybor:', selection['selected'])
print('CER: baza i warianty (historyczny / zwykly / laczny)')
baseline = json.loads((RESULT_DIRECTORY/'evidence/baseline-metrics.json').read_text())
for name, metrics in [('unchanged-baseline', baseline)] + [(row['id'], row['metrics']) for row in selection['candidates']]:
    print(name, ' / '.join(f"{100*metrics[domain]['cer']:.3f}%" for domain in
          ('historical-development', 'ordinary-development', 'combined')))
for row in selection['candidates']:
    print(row['id'], 'przechodzi' if row['eligible'] else 'odrzucony: '+', '.join(row['reasons']))
print('Kompletny pakiet:', result_zip, 'Bajty:', result_zip.stat().st_size)
print('Jesli pozostaje baza, ZIP zawiera raport i przypieta tozsamosc bazy, bez nowych wag.')
files.download(str(EVIDENCE_ZIP))
'''
        backup = '''BACKUP_TO_DRIVE = False
# Aby zachowac duzy pakiet, zmien powyzej na True i uruchom tylko te komorke.
if BACKUP_TO_DRIVE:
    from google.colab import drive
    import hashlib
    import shutil
    drive.mount('/content/drive')
    destination = Path('/content/drive/MyDrive/OCR_engine')/RESULT_DIRECTORY.name
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/result_zip.name
    def sha256(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                h.update(chunk)
        return h.hexdigest()
    expected = sha256(result_zip)
    if target.exists():
        assert sha256(target) == expected, 'Na Drive istnieje inny plik; nie nadpisuje go.'
    else:
        shutil.copyfile(result_zip, target)
    assert sha256(target) == expected, 'Kopia na Drive nie przeszla kontroli SHA256.'
    print('Zapisano kompletny pakiet na Drive:', target, 'SHA256:', expected)
else:
    print('Wagi sa w sesji Colaba. Aby zachowac je przed jej zakonczeniem, ustaw BACKUP_TO_DRIVE=True i uruchom tylko te komorke.')
'''
        compile(backup, 'backup', 'exec')
    for name, code in [('setup', setup), ('training', training), ('download', download)]:
        compile(code, name, 'exec')
    cells = [
        nbformat.v4.new_markdown_cell(title if version == 'v2' else '# Trening TrOCR: sprawdzone transkrypcje\n\n'
            'Wybierz **GPU T4** i **Uruchom wszystko**. Nie wgrywaj zadnych plikow.\n\n'
            '70 sprawdzonych linii treningowych, 500 probek zwyklego druku, 3 epoki LoRA. '
            'Kontrola: 9 linii historycznych z oddzielnej kolekcji i 75 zwyklych linii. '
            'Ryzykowne rodziny dziel sa wykluczone. To eksperyment, nie niezalezny benchmark SOTA. '
            'Oryginalne teksty i stare glify pozostaja bez modernizacji.\n\n'
            'Po treningu pobierze sie jeden ZIP z wagami i raportem.'),
        nbformat.v4.new_markdown_cell('## 1. Srodowisko'),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_markdown_cell('## 2. Dane, kontrola, trening i porownanie'),
        nbformat.v4.new_code_cell(training),
        nbformat.v4.new_markdown_cell('## 3. Pobranie modelu i raportu'),
        nbformat.v4.new_code_cell(download)]
    if version == 'v2':
        cells += [nbformat.v4.new_markdown_cell('## 4. Zachowaj pelny pakiet na Drive (opcjonalnie)'),
                  nbformat.v4.new_code_cell(backup)]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
            'accelerator': 'GPU', 'colab': {'name': 'colab_recognizer_reviewed_training_' + version + '.ipynb'},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    for index, cell in enumerate(notebook.cells):
        cell.id = 'reviewed-training-'+version+'-'+str(index)
    nbformat.validate(notebook)
    nbformat.write(notebook, Path(target))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', default=str(Path(__file__).with_name('colab_recognizer_reviewed_training_v1.ipynb')))
    parser.add_argument('--version', choices=('v1', 'v2'), default='v1')
    args = parser.parse_args()
    build(args.output, args.code_revision, args.version)
