"""Build a private, self-contained Kaggle notebook; never publish its embedded images."""
import argparse
import base64
import json
from pathlib import Path

from training.kaggle_body_dev_diagnostic import pack, load_input, digest


def build(manifest, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / 'body-dev-input.zip'
    checksum = pack(manifest, archive)
    rows, _, _ = load_input(archive.read_bytes(), checksum)
    script = Path(__file__).with_name('kaggle_body_dev_diagnostic.py').read_text(encoding='utf-8')
    encoded = base64.b64encode(archive.read_bytes()).decode('ascii')
    source = script + '\n\nINPUT_SHA256 = ' + repr(checksum) + '\nINPUT_BASE64 = ' + repr(encoded) + '\nrun(base64.b64decode(INPUT_BASE64), INPUT_SHA256, ' + repr(digest(script.encode('utf-8'))) + ')\n'
    cells = [
        {'id': 'scope', 'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Diagnostyka roboczych transkrypcji\n',
            'Notebook zawiera prywatna kopie wycinkow i poprawek. Pozostaw notebook prywatny na Kaggle. Nie publikuj go na GitHub.\n',
            'Wlacz GPU + Internet, potem Run All. Dane sa osadzone: nie dodawaj datasetu. Wynik ZIP pobierz z ostatniej komorki.\n',
            'To nie benchmark: 63 propozycje linii, w tym 7 do wyjasnienia. Osobne metryki dla 63 i 56 linii. Bez treningu.']},
        {'id': 'install', 'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
         'source': ['%pip install -q transformers==4.57.6 jiwer==4.0.0 huggingface_hub==0.36.0 sentencepiece==0.2.1']},
        {'id': 'run', 'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [source]},
    ]
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': '3.12'}}}
    target = output / 'kaggle_body_dev_PRIVATE.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    compile(source, str(target), 'exec')
    (output / 'runner.py').write_text(script, encoding='utf-8')
    summary = {'input_sha256': checksum, 'runner_sha256': digest(script.encode('utf-8')),
               'notebook_sha256': digest(target.read_bytes()), 'lines': len(rows),
               'uncertain_lines': sum(r['source_review_decision'] == 'needs-review' for r in rows)}
    (output / 'build.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.output), indent=2))
