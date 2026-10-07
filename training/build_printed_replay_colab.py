"""One CPU Colab, frozen input, no manual upload and no training side effects."""
import argparse
from pathlib import Path
import re


DATA_REVISION = "d942013b20616ff81223d1f20c6879746444e149"
DATA_SHA256 = "6510428da9c19dd76cc4e5ba2f1c0e45409119b97c608ddcff22f71e84a334eb"
DATA_PATH = "data/printed-replay-source-pilot-v1-20261007/printed-replay-source-pilot-v1.zip"


def build(target, revision):
    import nbformat
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Use a complete already-pushed runtime revision")
    setup = f'''from pathlib import Path
import hashlib
import os
import subprocess
import sys
import venv
from datetime import datetime, timezone

CODE_REVISION = {revision!r}
DATA_REVISION = {DATA_REVISION!r}
DATA_SHA256 = {DATA_SHA256!r}
DATA_PATH = {DATA_PATH!r}
repo = Path('/content/OCR_engine-printed-replay')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/PiotrStyla/OCR_engine.git', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', 'main'], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', CODE_REVISION], check=True)
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == CODE_REVISION
subprocess.run(['apt-get', 'update', '-qq'], check=True)
subprocess.run(['apt-get', 'install', '-y', '-qq', 'djvulibre-bin', 'tesseract-ocr', 'tesseract-ocr-pol'], check=True)
environment = Path('/content/slayer-printed-replay-env')
python = environment/'bin/python'
if not python.exists():
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir',
                'pillow==11.3.0', 'huggingface_hub==0.36.2', 'beautifulsoup4==4.13.5',
                'pytest==8.4.2'], check=True)
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
subprocess.run([str(python), '-m', 'pytest', 'tests/test_printed_replay_pilot.py',
                '-q', '-p', 'no:cacheprovider'], cwd=repo, check=True)
work = Path('/content/printed-replay-pilot-v1-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
work.mkdir()
'''
    inputs = '''import zipfile
download_code = """from huggingface_hub import hf_hub_download
import sys
print(hf_hub_download('PiotrSty/slayer-ocr-datasets', sys.argv[2], repo_type='dataset', revision=sys.argv[1]))
"""
archive = Path(subprocess.check_output([str(python), '-c', download_code, DATA_REVISION, DATA_PATH], text=True).strip())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == DATA_SHA256, 'Checksum danych nie zgadza sie.'
source_root = work/'input'
source_root.mkdir()
with zipfile.ZipFile(archive) as zipped:
    for member in zipped.infolist():
        path = Path(member.filename)
        assert not path.is_absolute() and '..' not in path.parts and '\\\\' not in member.filename
        assert ((member.external_attr >> 16) & 0o170000) != 0o120000, 'Symlink w ZIP'
    zipped.extractall(source_root)
print('INPUT_READY', source_root)
'''
    mining = '''from google.colab import files
EVIDENCE_ZIP = None
command = [str(python), '-m', 'training.mine_printed_replay', '--source-root', str(source_root),
           '--output', str(work/'mining')]
process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, bufsize=1)
for line in process.stdout:
    print(line, end='', flush=True)
    if line.startswith('EVIDENCE_ZIP '):
        EVIDENCE_ZIP = Path(line.strip().split(' ', 1)[1])
returncode = process.wait()
assert returncode == 0, 'Ekstrakcja nie zakonczona. Wklej blad; nie uruchamiaj treningu.'
assert EVIDENCE_ZIP is not None and EVIDENCE_ZIP.is_file(), 'Brak kompletnego ZIP-a.'
print('Gotowy ZIP:', EVIDENCE_ZIP)
files.download(str(EVIDENCE_ZIP))
'''
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# Real printed replay: pilot V1\n\n'
            'Wybierz **CPU** i **Uruchom wszystko**. Niczego nie wgrywaj. '
            '12 stron, 2 osobne ksiazki; oryginalne skany i uwierzytelnione teksty.\n\n'
            'To przygotowanie kandydatow danych, **nie trening**. Stara pisownia pozostaje '
            'bez zmian. ZIP zawiera wycinki, teksty, odrzucenia i pochodzenie. '
            'Kandydaci nie sa automatycznie zatwierdzonymi danymi treningowymi.'),
        nbformat.v4.new_markdown_cell('## 1. Srodowisko i testy'),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_markdown_cell('## 2. Automatyczne pobranie sprawdzonych zrodel'),
        nbformat.v4.new_code_cell(inputs),
        nbformat.v4.new_markdown_cell('## 3. Oryginalne skany, linie i ZIP\n\n'
            'Po pobraniu przekaz `printed-replay-pilot-v1-evidence.zip` do audytu. '
            'Jesli przegladarka blokuje pobieranie: panel Pliki, katalog printed-replay-pilot-v1-..., '
            'mining, prawy przycisk na ZIP-ie, Pobierz. Nie zamykaj sesji przed pobraniem.'),
        nbformat.v4.new_code_cell(mining)], metadata={
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
            'colab': {'name': 'printed-replay-pilot-v1.ipynb', 'provenance': []}})
    nbformat.validate(notebook)
    nbformat.write(notebook, Path(target))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', default='training/colab_printed_replay_pilot_v1.ipynb')
    args = parser.parse_args()
    build(args.output, args.code_revision)
