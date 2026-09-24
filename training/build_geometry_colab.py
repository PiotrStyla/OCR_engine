"""Generate a small Colab notebook with automatically downloaded paired inputs."""
import hashlib
import json
from pathlib import Path

DATA_COMMIT = '95786263807ff1d640db5c2fc3a16b64ab4c0e3d'
OLD_HASH = '913ded2bd5d742099567a2652460baaf3c4a8bb16625492d0931607396dbdd55'
NEW_HASH = 'de90e9b90eb658341e01a16a24e2eb101de4c7cfb50ac1bdee7724878090f278'
BANDS_COMMIT = 'ffa73b902140e790df180bfcd3f0bca4fc2a2612'
BANDS_HASH = '730a7cfe4a0c2d2a6575f53fdc7d65a36fb2bf31fa8e9f7d9d734cc514e34431'


def build(target, *, bands=False):
    root = Path(__file__).resolve().parent
    runner = (root / 'kaggle_body_dev_diagnostic.py').read_text(encoding='utf-8')
    runner = runner.replace("Path('/kaggle/working')", "Path('/content')").replace(
        'Enable GPU and Internet in Kaggle', 'Select GPU runtime in Colab, then Run all')
    runner += '\n    return archive\n'
    helpers = (root / 'geometry_comparison.py').read_text(encoding='utf-8')
    environment = (root / 'colab_environment.py').read_text(encoding='utf-8')
    prefix = runner + '\n' + helpers + '\n' + environment
    checksum = hashlib.sha256(prefix.encode()).hexdigest()
    commit = BANDS_COMMIT if bands else DATA_COMMIT
    new_hash = BANDS_HASH if bands else NEW_HASH
    folder = 'body-geometry-bands' if bands else 'body-auto-geometry'
    changed, fallback = (33, 30) if bands else (31, 32)
    tail = f'''
check_colab_environment()
from urllib.request import urlopen
ROOT_URL = 'https://raw.githubusercontent.com/PiotrStyla/OCR_engine/{commit}/'
with urlopen(ROOT_URL + 'experiments/2026-09-21/body-dev-diagnostic/body-dev-input.zip', timeout=120) as response:
    old_data = response.read()
with urlopen(ROOT_URL + 'experiments/2026-09-24/{folder}/body-auto-geometry-input.zip', timeout=120) as response:
    new_data = response.read()
payload = pair_inputs(old_data, '{OLD_HASH}', new_data, '{new_hash}', load_input)
_base_metrics = metrics
def metrics(rows, predictions):
    return paired_metrics(rows, predictions, _base_metrics)
result_archive = run(payload, digest(payload), '{checksum}')
with zipfile.ZipFile(result_archive) as evidence:
    final_report = json.loads(evidence.read('report.json'))
failed = []
for model, arms in final_report['results'].items():
    for variant, scores in arms.items():
        score = scores['all_draft_lines']
        if score['errors']:
            failed.append(f"{{model}} / {{variant}}: {{score['errors']}} errors")
        else:
            print(f"{{model}} / {{variant}}: CER={{score['cer']:.2%}}, WER={{score['wer']:.2%}}")
if failed:
    print('INCOMPLETE RUN. Error-derived CER is not model quality. Download the diagnostic ZIP.\\n' + '\\n'.join(failed))
else:
    print('COMPLETE: both models finished both arms. Draft development result, not a benchmark.')
'''
    code = prefix + tail
    compile(code, '<geometry-colab>', 'exec')
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'colab': {'name': Path(target).name, 'provenance': []}, 'accelerator': 'GPU'},
        'cells': [
            {'cell_type': 'markdown', 'id': 'scope', 'metadata': {}, 'source': [
                '# ' + ('Granice wierszy i interpunkcja' if bands else 'Automatyczne ramki') + ': porownanie 63 linii\n',
                'GPU, potem Uruchom wszystko. Dane pobieraja sie automatycznie z GitHuba. '
                'Nic nie dodawaj recznie. Ostatnia komorka pobiera ZIP. '
                'Przy pracy w starej sesji po instalacji zrestartuj sesje i wybierz Uruchom wszystko.\n',
                f'Kazdy model odczyta oryginalne 63 wycinki i wariant {changed} nowych ramek + {fallback} oryginalnych fallbackow. '
                'Referencje i historyczna pisownia pozostaja bez zmian. '
                'Osobne wyniki dla obu wariantow; to eksperyment deweloperski, nie trening ani dowod SOTA. '
                'Nowe ramki nie sa recznie zatwierdzone.\n',
                'Zrodlo: IMPACT/PSNC, PiotrSty/impact-psnc-polish-ocr, CC-BY-3.0. '
                'Zmiany: wycinki PNG i robocze transkrypcje; teraz zmieniono tylko geometrie.\n',
                ('Wariant eksperymentalny wybiela piksele poza granicami wiersza. Nie zmienia tekstow wzorcowych. '
                 'Nie porownuj podzbioru 33 linii bezposrednio ze starym podzbiorem 31; porownuj wszystkie 63.\n') if bands else '']},
            {'cell_type': 'code', 'id': 'install', 'metadata': {}, 'execution_count': None, 'outputs': [],
             'source': ['%pip install -q --upgrade transformers==4.57.6 jiwer==4.0.0 huggingface_hub==0.36.0 sentencepiece==0.2.1']},
            {'cell_type': 'code', 'id': 'run', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [code]},
            {'cell_type': 'code', 'id': 'download', 'metadata': {}, 'execution_count': None, 'outputs': [],
             'source': ['from google.colab import files\nfiles.download(str(result_archive))\n']},
        ]}
    target = Path(target)
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    build(Path(__file__).with_name('colab_auto_geometry.ipynb'))
    build(Path(__file__).with_name('colab_geometry_bands.ipynb'), bands=True)
