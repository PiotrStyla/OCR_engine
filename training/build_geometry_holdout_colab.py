"""Build the self-contained, hash-pinned geometry holdout Colab notebook."""
import hashlib
import json
from pathlib import Path

MANIFEST_COMMIT = '63c922624e8aeed10113a1060615f1dc74b962eb'
MANIFEST_SHA256 = '0b5459ea56d352de532b2cdc2284bd701b1c98458c67afd9f2220b6f0a6a7b18'


def build(target):
    root = Path(__file__).resolve().parent
    detector = (root / 'body_line_geometry.py').read_text(encoding='utf-8')
    environment = (root / 'colab_environment.py').read_text(encoding='utf-8')
    runner = (root / 'geometry_holdout_runner.py').read_text(encoding='utf-8')
    runner = runner.replace('from training.body_line_geometry import crop_line_band, detect_lines\n', '')
    runner = runner.replace(
        'from training.kaggle_printed_dev_control import DATASET, REVISION, MODELS, FROZEN\n',
        "DATASET = 'PiotrSty/impact-psnc-polish-ocr'\n"
        "REVISION = 'c7cb156fb95d2880699c33725bbaf1fbc1008fea'\n"
        "MODELS = {'microsoft/trocr-base-printed': '93450be3f1ed40a930690d951ef3932687cc1892', "
        "'PiotrSty/trocr-pl-mixed-v3': '85d0c91c26f8e088849096dded7c9ba10b4cd9c9'}\n"
        "FROZEN = {'NA2_FT', 'Nowiny_z_Rakuz_FT', 'Powodzenia_FT'}\n")
    prefix = detector + '\n' + environment + '\n' + runner
    runner_sha256 = hashlib.sha256(prefix.encode('utf-8')).hexdigest()
    tail = f'''
check_colab_environment({{'opencv-python-headless': '4.12.0.88'}})
MANIFEST_URL = ('https://raw.githubusercontent.com/PiotrStyla/OCR_engine/'
                '{MANIFEST_COMMIT}/experiments/2026-09-24/geometry-holdout-v1/manifest.json')
with urlopen(MANIFEST_URL, timeout=120) as response:
    manifest_data = response.read()
result_archive = run(manifest_data, '{MANIFEST_SHA256}', '{runner_sha256}')
with zipfile.ZipFile(result_archive) as evidence:
    final_report = json.loads(evidence.read('report.json'))
failed = []
for model, result in final_report['results'].items():
    for variant in VARIANTS:
        score_result = result[variant]['all_regions']
        if score_result['errors']:
            failed.append(f"{{model}} / {{variant}}: {{score_result['errors']}} region errors")
        else:
            print(f"{{model}} / {{variant}}: CER={{score_result['cer']:.2%}}, "
                  f"WER={{score_result['wer']:.2%}}, lines={{score_result['detected_lines']}}")
    print(model, result['comparison'])
if failed:
    print('INCOMPLETE RUN. Error-derived CER is not model quality. Download the diagnostic ZIP.\\n' +
          '\\n'.join(failed))
else:
    print('COMPLETE: 12/12 regions retained in both variants. Development holdout, not SOTA evidence.')
'''
    code = prefix + tail
    compile(code, '<geometry-holdout-colab>', 'exec')
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'colab': {'name': Path(target).name, 'provenance': []}, 'accelerator': 'GPU'},
                'cells': [
                    {'cell_type': 'markdown', 'id': 'scope', 'metadata': {}, 'source': [
                        '# Geometria OCR: zamrozona proba 12 nowych kolekcji\n',
                        'Wybierz GPU i kliknij **Uruchom wszystko**. Niczego nie wgrywaj recznie. '
                        'Ostatnia komorka pobierze ZIP wynikow.\n',
                        'Proba zostala przypieta na GitHubie przed OCR: 12 regionow, 12 nowych kolekcji, '
                        '67 wierszy referencyjnych. Obrazy sa pobierane z przypietej rewizji Hugging Face '
                        'i sprawdzane SHA256. Oba warianty same wykrywaja linie; liczba linii referencyjnych '
                        'nie steruje segmentacja i zaden region nie wypada z mianownika.\n',
                        'Pisownia historyczna pozostaje bez zmian. Referencje sa upstream i nie przeszly '
                        'recznej weryfikacji. To test deweloperski geometrii, nie benchmark modelu ani dowod SOTA.\n']},
                    {'cell_type': 'code', 'id': 'install', 'metadata': {}, 'execution_count': None, 'outputs': [],
                     'source': ['%pip install -q --upgrade transformers==4.57.6 jiwer==4.0.0 '
                                'huggingface_hub==0.36.0 sentencepiece==0.2.1 '
                                'opencv-python-headless==4.12.0.88']},
                    {'cell_type': 'code', 'id': 'run', 'metadata': {}, 'execution_count': None,
                     'outputs': [], 'source': [code]},
                    {'cell_type': 'code', 'id': 'download', 'metadata': {}, 'execution_count': None,
                     'outputs': [], 'source': ['from google.colab import files\nfiles.download(str(result_archive))\n']},
                ]}
    target = Path(target)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    build(Path(__file__).with_name('colab_geometry_holdout.ipynb'))
