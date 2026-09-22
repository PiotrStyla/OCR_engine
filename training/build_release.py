"""Assemble a PolOCRBench release: HuggingFace layout, AmuEval TSVs, checksums.

Each ``--split name=DIR`` points at a generator data directory
(``training.generate_documents`` layout: ``images/``, ``manifest-A/B/C.jsonl``,
``generation.json``). Split names carry their role:

- ``train`` — public with ground truth (training material);
- ``testA`` — public images and ``in-*.tsv`` ids only, ground truth private
  (development/leaderboard set);
- ``testB`` — fully private (hidden generalization set, ranking).

Output layout::

    hf/train/            images/ + manifest-*.jsonl + generation.json
    hf/testA/            images/ + in-<subtask>.tsv
    private/<split>/     manifest-*.jsonl + generation.json (+ testB images/)
    amueval/<split>/<subtask>/{in.tsv,expected.tsv,sample-out.tsv}
    DATASET_CARD.md  LICENSE-DATA.txt  RELEASE.json

``expected.tsv`` is packed from the gold manifests with the public
``training.submission_tsv`` rules; ``sample-out.tsv`` is an empty-cell
template (pack real baseline predictions with ``training.submission_tsv``).
The cross-split integrity gate (``training.check_split_integrity``) refuses a
release with duplicates unless ``--allow-violations`` (the decision is recorded
in ``RELEASE.json``). ``--holdout-field/--holdout-split`` additionally enforce
the Test B generalization claim at build time.

Usage:
  python -m training.build_release --version 0.1 --split train=data/train \\
      --split testA=data/testA --split testB=data/testB --output releases/0.1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from training.check_split_integrity import check, load_split
from training.stage_impact_benchmark import digest
from training.submission_tsv import pack, validate_tsv
from training.validate_submission import load_jsonl

PROTOCOL_VERSION = 'polocrbench-release-v1'
ROLES = {'train': 'public-gt', 'testA': 'public-ids', 'testB': 'private'}
SUBTASKS = ('A', 'B', 'C')
_PAYLOAD_KEYS = {'A': 'text', 'B': 'html', 'C': 'fields'}

DATASET_CARD = """---
language: [pl]
license: cc-by-4.0
task_categories: [image-to-text, object-detection]
tags: [ocr, document-understanding, tables, key-information-extraction, polish]
---

# PolOCRBench {version} — Polish Document Understanding

Trudne polskie dokumenty (współczesne druki, druki historyczne, pismo ręczne,
zdjęcia telefonu) i trzy podzadania z obrazu strony: **A** transkrypcja do
Markdown (CER/WER), **B** ekstrakcja tabel do HTML (TEDS), **C** ekstrakcja
informacji kluczowych do JSON (field-level F1). Wynik zbiorczy to średnia
(1 − CER, TEDS, F1).

## Zawartość

| Katalog | Co | GT |
| --- | --- | --- |
| `train/` | strony treningowe + manifesty A/B/C | tak |
| `testA/` | strony zbioru dev + `in-*.tsv` | ukryte (leaderboard) |
| `testB/` | ukryty zbiór generalizacji (ranking główny) | ukryte |

Formaty, miary i normalizacja: `docs/POLOCRBENCH_SUBTASKS_BC.md` w repo
https://github.com/PiotrStyla/OCR_engine (skrypty ewaluacyjne MIT);
zgłoszenia w formacie AmuEval (`out.tsv`, kolejność `in.tsv`).

## Licencja

Dane: CC BY 4.0. Skrypty ewaluacyjne: MIT.
"""

LICENSE_DATA = """PolOCRBench data — CC BY 4.0 (Creative Commons Attribution 4.0 International)
https://creativecommons.org/licenses/by/4.0/

You are free to share and adapt the material for any purpose, even commercially,
under the terms of attribution.
"""

LICENSE_CODE = """PolOCRBench evaluation and tooling scripts — MIT License.

Copyright (c) 2026 OCR Engine / PolOCRBench

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def discover(data_dir):
    data_dir = Path(data_dir)
    manifests = {subtask: data_dir / f'manifest-{subtask}.jsonl' for subtask in SUBTASKS}
    missing = [subtask for subtask, path in manifests.items() if not path.exists()]
    if missing:
        raise ValueError(f'{data_dir}: missing manifests {missing}')
    return {'dir': data_dir,
            'manifests': {subtask: path for subtask, path in manifests.items()
                          if load_jsonl(path)},
            'images': data_dir / 'images',
            'generation': data_dir / 'generation.json'
            if (data_dir / 'generation.json').exists() else None}


def gold_predictions(manifest, subtask):
    """Manifest rows are valid prediction records for submission_tsv.pack."""
    return manifest


def build_split(name, data, output, version):
    """Write hf/private/amueval parts of one split; returns its summary."""
    role = ROLES[name]
    summary = {'role': role, 'subtasks': {}}
    public_dir = output / 'hf' / name
    private_dir = output / 'private' / name
    if role != 'private':
        shutil.copytree(data['images'], public_dir / 'images')
    else:
        shutil.copytree(data['images'], private_dir / 'images')
    if role == 'public-gt':
        targets = [public_dir]
    elif role == 'public-ids':
        targets = [private_dir]
    else:
        targets = [private_dir]
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        for manifest in data['manifests'].values():
            shutil.copy2(manifest, target / manifest.name)
        if data['generation'] is not None:
            shutil.copy2(data['generation'], target / 'generation.json')
    for subtask, manifest in data['manifests'].items():
        rows = load_jsonl(manifest)
        package = output / 'amueval' / name / subtask
        package.mkdir(parents=True, exist_ok=True)
        in_tsv = package / 'in.tsv'
        in_tsv.write_text('\n'.join(row['id'] for row in rows) + '\n',
                          encoding='utf-8', newline='\n')
        expected = package / 'expected.tsv'
        pack(in_tsv, gold_predictions(manifest, subtask), expected, subtask)
        sample = package / 'sample-out.tsv'
        sample.write_text('\n'.join([''] * len(rows)) + '\n', encoding='utf-8', newline='\n')
        validate_tsv(in_tsv, expected)
        validate_tsv(in_tsv, sample)
        if role != 'private':
            (public_dir / f'in-{subtask}.tsv').write_text(
                in_tsv.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
        summary['subtasks'][subtask] = {'records': len(rows),
                                        'manifest_sha256': digest(manifest)}
    return summary


def _tree_digests(root):
    digests = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            digests[str(path.relative_to(root)).replace('\\', '/')] = digest(path)
    return digests


def build_release(splits, output, version, allow_violations=False,
                  holdout_field=None, holdout_split=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        if 'A' not in split['manifests']:
            raise ValueError(f'{name}: manifest-A is required for the integrity gate')
    integrity = check([load_split(name, split['manifests']['A']) for name, split in splits.items()],
                      holdout_field=holdout_field, holdout_split=holdout_split)
    if integrity['violations'] and not allow_violations:
        raise ValueError(f"Split integrity violations: {integrity['violations']} "
                         f'(rerun with --allow-violations to record the decision)')
    summary = {name: build_split(name, data, output, version) for name, data in splits.items()}
    (output / 'DATASET_CARD.md').write_text(DATASET_CARD.format(version=version),
                                            encoding='utf-8', newline='\n')
    (output / 'LICENSE-DATA.txt').write_text(LICENSE_DATA, encoding='utf-8', newline='\n')
    (output / 'LICENSE-CODE.txt').write_text(LICENSE_CODE, encoding='utf-8', newline='\n')
    report = {'schema': PROTOCOL_VERSION, 'version': version,
              'roles': {name: ROLES[name] for name in splits},
              'splits': summary,
              'integrity': integrity,
              'integrity_overridden': bool(integrity['violations'] and allow_violations),
              'files': _tree_digests(output)}
    (output / 'RELEASE.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding='utf-8', newline='\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--split', action='append', required=True,
                        help='name=DIR with generator output (train, testA, testB)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-violations', action='store_true')
    parser.add_argument('--holdout-field')
    parser.add_argument('--holdout-split')
    args = parser.parse_args()
    splits = {}
    for spec in args.split:
        name, _, directory = spec.partition('=')
        if name not in ROLES or not directory:
            parser.error('--split expects train|testA|testB=DIR')
        if name in splits:
            parser.error(f'duplicate split: {name}')
        splits[name] = discover(directory)
    if not splits:
        parser.error('at least one --split is required')
    try:
        report = build_release(splits, args.output, args.version,
                               allow_violations=args.allow_violations,
                               holdout_field=args.holdout_field,
                               holdout_split=args.holdout_split)
    except (FileExistsError, ValueError) as error:
        print(json.dumps({'error': str(error)}, ensure_ascii=False))
        sys.exit(1)
    print(json.dumps({'version': report['version'], 'roles': report['roles'],
                      'records': {name: {task: data['records'] for task, data in
                                         summary['subtasks'].items()}
                                  for name, summary in report['splits'].items()},
                      'integrity_violations': report['integrity']['violations'],
                      'integrity_overridden': report['integrity_overridden']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
