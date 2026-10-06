"""Generate one pinned Colab notebook with automatic inputs and one result ZIP."""
import argparse
from pathlib import Path
import re

DATA_REVISION = '3a301fd12a6272b28774d1888213ddf528b47f24'
DATA_SHA256 = '3e39b7d02b90a0a741adde506c0467698c46fe86c82846d3cc0e648eebaae32e'


def build(target, code_revision):
    import nbformat

    if not re.fullmatch('[0-9a-f]{40}', code_revision):
        raise ValueError('Use the complete, already pushed runtime commit SHA')
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
packages = ['transformers==4.57.6', 'tokenizers==0.22.2', 'huggingface_hub==0.36.2',
            'peft==0.19.1', 'accelerate==1.13.0', 'jiwer==4.0.0',
            'pillow==11.3.0', 'sentencepiece==0.2.1']
subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', *packages], check=True)
subprocess.run([str(python), '-c', "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; from transformers.generation import GenerationMixin; import peft, jiwer; print('TRAINING_IMPORTS_OK')"], check=True)
print('Gotowe. Dane zostana pobrane automatycznie; niczego nie wgrywaj.')
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
    download = '''assert RESULT_DIRECTORY is not None, 'Najpierw musi zakonczyc sie komorka treningu.'
result_zip = RESULT_DIRECTORY/'recognizer-reviewed-colab-v1-result.zip'
assert result_zip.is_file(), 'Brak kompletnego archiwum wyniku.'
print('Model i raport w jednym pliku:', result_zip)
files.download(str(result_zip))
'''
    for name, code in [('setup', setup), ('training', training), ('download', download)]:
        compile(code, name, 'exec')
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# Trening TrOCR: sprawdzone transkrypcje\n\n'
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
        nbformat.v4.new_code_cell(download)], metadata={
            'accelerator': 'GPU', 'colab': {'name': 'colab_recognizer_reviewed_training_v1.ipynb'},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    for index, cell in enumerate(notebook.cells):
        cell.id = 'reviewed-training-'+str(index)
    nbformat.validate(notebook)
    nbformat.write(notebook, Path(target))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', default=str(Path(__file__).with_name('colab_recognizer_reviewed_training_v1.ipynb')))
    args = parser.parse_args()
    build(args.output, args.code_revision)
