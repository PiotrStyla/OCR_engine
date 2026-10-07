"""One CPU Colab, frozen input, no manual upload and no training side effects."""
import argparse
from pathlib import Path
import re


DATA_REVISION = "d942013b20616ff81223d1f20c6879746444e149"
DATA_SHA256 = "6510428da9c19dd76cc4e5ba2f1c0e45409119b97c608ddcff22f71e84a334eb"
DATA_PATH = "data/printed-replay-source-pilot-v1-20261007/printed-replay-source-pilot-v1.zip"


def build(target, revision, version="v1", *, input_spec=None):
    import nbformat
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Use a complete already-pushed runtime revision")
    if version not in ("v1", "v2", "expansion-v1"):
        raise ValueError("Unknown notebook version")
    if version == "expansion-v1" and input_spec is None:
        raise ValueError("Expansion requires an explicit frozen input receipt")
    spec = input_spec or {"revision": DATA_REVISION, "sha256": DATA_SHA256, "path": DATA_PATH}
    if not re.fullmatch("[0-9a-f]{40}", spec["revision"]) or not re.fullmatch("[0-9a-f]{64}", spec["sha256"]):
        raise ValueError("Invalid frozen input revision/checksum")
    setup = f'''from pathlib import Path
import hashlib
import os
import subprocess
import sys
import venv
from datetime import datetime, timezone

CODE_REVISION = {revision!r}
DATA_REVISION = {spec['revision']!r}
DATA_SHA256 = {spec['sha256']!r}
DATA_PATH = {spec['path']!r}
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
                'pytest==8.4.2', 'nbformat==5.10.4'], check=True)
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
subprocess.run([str(python), '-m', 'pytest', 'tests/test_printed_replay_pilot.py',
                '-q', '-p', 'no:cacheprovider'], cwd=repo, check=True)
work = Path('/content/printed-replay-pilot-v1-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
work.mkdir()
'''
    inputs = '''import zipfile
from pathlib import PurePosixPath
download_code = """from huggingface_hub import hf_hub_download
import sys
print(hf_hub_download('PiotrSty/slayer-ocr-datasets', sys.argv[2], repo_type='dataset', revision=sys.argv[1]))
"""
archive = Path(subprocess.check_output([str(python), '-c', download_code, DATA_REVISION, DATA_PATH], text=True).strip())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == DATA_SHA256, 'Checksum danych nie zgadza sie.'
source_root = work/'input'
source_root.mkdir()
with zipfile.ZipFile(archive) as zipped:
    members = zipped.infolist()
    assert len({m.filename for m in members}) == len(members), 'Duplicate ZIP member'
    assert sum(m.file_size for m in members) <= 256_000_000, 'Oversized input archive'
    for member in members:
        path = PurePosixPath(member.orig_filename)
        assert member.orig_filename == member.filename and ':' not in member.filename
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
    if version in ("v2", "expansion-v1"):
        setup = setup.replace("printed-replay-pilot-v1", "printed-replay-pilot-v2")
        mining = mining.replace("'--output', str(work/'mining')]",
                                "'--output', str(work/'mining'), '--protocol', 'v2']")
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
    if version in ("v2", "expansion-v1"):
        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                cell.source = cell.source.replace("pilot V1", "pilot V2").replace(
                    "printed-replay-pilot-v1", "printed-replay-pilot-v2")
        notebook.cells[0].source += ("\n\nV2 zachowuje dokladne linie nawet przy niskim "
            "pokryciu strony i dolacza pelne obrazy oraz TSV. Nie zmniejsza progu "
            "pewnosci slow ani nie dopuszcza automatycznie danych do treningu.")
        notebook.metadata["colab"]["name"] = "printed-replay-pilot-v2.ipynb"
    if version == "expansion-v1":
        config_path, config_hash = spec["works_config"], spec["works_config_sha256"]
        if not re.fullmatch("[0-9a-f]{64}", config_hash):
            raise ValueError("Invalid work configuration checksum")
        notebook.cells[0].source = ("# Printed replay: expansion V1\n\n"
            "Wybierz **CPU**, potem **Uruchom wszystko**. Niczego nie wgrywaj. "
            f"Zamrozona partia: {spec['pages']} stron, {spec['work_families']} rodzin zrodel. "
            "Osobne rodziny kontrolne nie trafiaja do treningu.\n\n"
            "To wydobywanie kandydatow replay, **nie trening**. Zachowujemy stara pisownie. "
            "Dane obejmuja dawny druk, nie reprezentatywny zbior wspolczesnych dokumentow.")
        notebook.cells[2].source = notebook.cells[2].source.replace(
            "printed-replay-pilot-v2-", "printed-replay-expansion-v1-").replace(
            "'tests/test_printed_replay_pilot.py',", "'tests/test_printed_replay_pilot.py', 'tests/test_printed_replay_expansion.py',")
        notebook.cells[4].source += f"\nWORKS_CONFIG = repo/{config_path!r}\nassert hashlib.sha256(WORKS_CONFIG.read_bytes()).hexdigest() == {config_hash!r}, 'Work config checksum mismatch'\n"
        notebook.cells[6].source = notebook.cells[6].source.replace(
            "'--protocol', 'v2']", "'--protocol', 'v2', '--works-config', str(WORKS_CONFIG)]").replace(
            "print('Gotowy ZIP:', EVIDENCE_ZIP)",
            "import shutil\nexport_zip = work/'printed-replay-expansion-v1-evidence.zip'\nshutil.copyfile(EVIDENCE_ZIP, export_zip)\nEVIDENCE_ZIP = export_zip\nprint('Gotowy ZIP:', EVIDENCE_ZIP)")
        notebook.cells[5].source = ("## 3. Native skany, linie i ZIP\n\n"
            "Zwracany plik: `printed-replay-expansion-v1-evidence.zip`. "
            "Zawiera wszystkie native strony, TSV, wycinki, odrzucenia i pochodzenie. "
            "Jesli automatyczne pobieranie jest blokowane, pobierz ZIP z panelu Pliki "
            "w katalogu printed-replay-expansion-v1-...; nie zamykaj sesji wczesniej.")
        notebook.metadata["colab"]["name"] = "printed-replay-expansion-v1.ipynb"
    nbformat.validate(notebook)
    nbformat.write(notebook, Path(target))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output')
    parser.add_argument('--version', choices=('v1', 'v2', 'expansion-v1'), default='v1')
    parser.add_argument('--input-receipt')
    args = parser.parse_args()
    default = 'training/colab_printed_replay_expansion_v1.ipynb' if args.version == 'expansion-v1' else f'training/colab_printed_replay_pilot_{args.version}.ipynb'
    target = args.output or default
    import json
    spec = json.loads(Path(args.input_receipt).read_text(encoding="utf-8")) if args.input_receipt else None
    build(target, args.code_revision, args.version, input_spec=spec)
