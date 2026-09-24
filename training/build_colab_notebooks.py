"""Derive Colab notebooks without changing OCR inputs, weights or scoring."""
import hashlib
import json
from pathlib import Path


def convert(source_path, target_path, runner_path):
    source_path, target_path, runner_path = map(Path, (source_path, target_path, runner_path))
    notebook = json.loads(source_path.read_text(encoding='utf-8'))
    runner = runner_path.read_text(encoding='utf-8')
    code = ''.join(notebook['cells'][-1]['source'])
    if not code.startswith(runner):
        raise ValueError('Embedded runner differs from source; review before converting')
    adapted = runner.replace("Path('/kaggle/working')", "Path('/content')").replace(
        "Enable GPU and Internet in Kaggle", "Select a GPU runtime in Colab, then Run all")
    adapted += '\n    return archive\n'
    tail = code[len(runner):]
    old_hash = hashlib.sha256(runner.encode('utf-8')).hexdigest()
    new_hash = hashlib.sha256(adapted.encode('utf-8')).hexdigest()
    if old_hash not in tail:
        raise ValueError('Missing expected runner hash')
    tail = tail.replace(old_hash, new_hash)
    if tail.count('\nrun(') != 1:
        raise ValueError('Expected one top-level run call')
    tail = tail.replace('\nrun(', '\nresult_archive = run(')
    code = adapted + tail
    compile(code, str(target_path), 'exec')
    ab = 'crop_ab' in source_path.stem
    notebook['cells'][0]['source'] = [
        '# OCR: Google Colab' + (' / test A/B' if ab else ' / 63 linie') + '\n',
        'Wybierz GPU w ustawieniach srodowiska wykonawczego, potem Uruchom wszystko. '
        'Nie dodawaj recznie datasetu ani ZIP-a. '
        + ('Dwa obrazy A/B sa osadzone w notebooku. ' if ab else 'Komplet danych pobiera sie automatycznie z przypietego commita GitHub. ')
        + 'Ostatnia komorka pobiera ZIP wynikow; kopia pozostaje w /content. '
        'Nie zamykaj sesji przed pobraniem wynikow.\n',
        'To diagnostyka na roboczych referencjach, nie benchmark ani trening. '
        'Zachowujemy historyczna pisownie, akcenty i dlugie s. '
        'Modele, dane, FP32 i metryki jak w wersji Kaggle; wersje srodowiska sa zapisywane.\n',
        'Zrodlo obrazow: IMPACT/PSNC, PiotrSty/impact-psnc-polish-ocr, CC-BY-3.0. '
        'Zmiany: wycinki PNG i robocze transkrypcje.\n',
    ]
    notebook['cells'][-1]['source'] = [code]
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    notebook['cells'].append({'id': 'download', 'cell_type': 'code', 'metadata': {},
        'execution_count': None, 'outputs': [], 'source': [
            'from google.colab import files\n',
            'if not result_archive.is_file():\n    raise FileNotFoundError(result_archive)\n',
            'files.download(str(result_archive))\n']})
    notebook['metadata']['colab'] = {'name': target_path.name, 'provenance': []}
    notebook['metadata']['accelerator'] = 'GPU'
    notebook['metadata']['ocr_provenance'] = {'source_notebook': source_path.name,
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'colab_runner_sha256': new_hash}
    target_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    for name in ['body_dev_diagnostic', 'body_crop_ab']:
        convert(root / f'kaggle_{name}.ipynb', root / f'colab_{name}.ipynb',
                root / 'kaggle_body_dev_diagnostic.py')
