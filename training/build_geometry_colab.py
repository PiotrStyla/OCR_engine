"""Generate a small Colab notebook with automatically downloaded paired inputs."""
import hashlib
import json
from pathlib import Path

DATA_COMMIT = '95786263807ff1d640db5c2fc3a16b64ab4c0e3d'
OLD_HASH = '913ded2bd5d742099567a2652460baaf3c4a8bb16625492d0931607396dbdd55'
NEW_HASH = 'de90e9b90eb658341e01a16a24e2eb101de4c7cfb50ac1bdee7724878090f278'


def build(target):
    root = Path(__file__).resolve().parent
    runner = (root / 'kaggle_body_dev_diagnostic.py').read_text(encoding='utf-8')
    runner = runner.replace("Path('/kaggle/working')", "Path('/content')").replace(
        'Enable GPU and Internet in Kaggle', 'Select GPU runtime in Colab, then Run all')
    runner += '\n    return archive\n'
    helpers = (root / 'geometry_comparison.py').read_text(encoding='utf-8')
    prefix = runner + '\n' + helpers
    checksum = hashlib.sha256(prefix.encode()).hexdigest()
    tail = f'''
from urllib.request import urlopen
ROOT_URL = 'https://raw.githubusercontent.com/PiotrStyla/OCR_engine/{DATA_COMMIT}/'
with urlopen(ROOT_URL + 'experiments/2026-09-21/body-dev-diagnostic/body-dev-input.zip', timeout=120) as response:
    old_data = response.read()
with urlopen(ROOT_URL + 'experiments/2026-09-24/body-auto-geometry/body-auto-geometry-input.zip', timeout=120) as response:
    new_data = response.read()
payload = pair_inputs(old_data, '{OLD_HASH}', new_data, '{NEW_HASH}', load_input)
_base_metrics = metrics
def metrics(rows, predictions):
    return paired_metrics(rows, predictions, _base_metrics)
result_archive = run(payload, digest(payload), '{checksum}')
'''
    code = prefix + tail
    compile(code, '<geometry-colab>', 'exec')
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'colab': {'name': 'colab_auto_geometry.ipynb', 'provenance': []}, 'accelerator': 'GPU'},
        'cells': [
            {'cell_type': 'markdown', 'id': 'scope', 'metadata': {}, 'source': [
                '# Automatyczne ramki: porownanie na tych samych 63 liniach\n',
                'GPU, potem Uruchom wszystko. Dane pobieraja sie automatycznie z GitHuba. '
                'Nic nie dodawaj recznie. Ostatnia komorka pobiera ZIP.\n',
                'Kazdy model odczyta oryginalne 63 wycinki i wariant 31 nowych ramek + 32 oryginalnych fallbackow. '
                'Referencje i historyczna pisownia pozostaja bez zmian. '
                'Osobne wyniki dla obu wariantow; to eksperyment deweloperski, nie trening ani dowod SOTA. '
                'Nowe ramki nie sa recznie zatwierdzone.\n',
                'Zrodlo: IMPACT/PSNC, PiotrSty/impact-psnc-polish-ocr, CC-BY-3.0. '
                'Zmiany: wycinki PNG i robocze transkrypcje; teraz zmieniono tylko geometrie.\n']},
            {'cell_type': 'code', 'id': 'install', 'metadata': {}, 'execution_count': None, 'outputs': [],
             'source': ['%pip install -q transformers==4.57.6 jiwer==4.0.0 huggingface_hub==0.36.0 sentencepiece==0.2.1']},
            {'cell_type': 'code', 'id': 'run', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [code]},
            {'cell_type': 'code', 'id': 'download', 'metadata': {}, 'execution_count': None, 'outputs': [],
             'source': ['from google.colab import files\nfiles.download(str(result_archive))\n']},
        ]}
    target = Path(target)
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    build(Path(__file__).with_name('colab_auto_geometry.ipynb'))
