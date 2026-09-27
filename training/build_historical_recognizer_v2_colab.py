"""Generate the frozen Colab replay experiment for historical recognition v2."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from training.build_historical_recognizer_colab import build as build_v1


CODE_REVISION = "8b950b8f5c42af259269fb0bb5a706f1c6724148"
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-09-27/historical-recognizer-v2/config.json"
)
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
BASE_MODEL = CONFIG["base_model"]
BASE_REVISION = CONFIG["base_revision"]
HISTORICAL_REVISION = CONFIG["sources"]["historical"]["revision"]
EHRI_REVISION = CONFIG["sources"]["ehri"]["revision"]
SYN_REVISION = CONFIG["sources"]["synthetic"]["revision"]


def _cell(notebook: dict, identifier: str) -> dict:
    return next(cell for cell in notebook["cells"] if cell.get("id") == identifier)


def _set_source(notebook: dict, identifier: str, source: str) -> None:
    if identifier != "scope":
        compile(source, f"<{identifier}>", "exec")
    _cell(notebook, identifier)["source"] = [source]


def build(target: str | Path) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        draft = Path(temporary) / "v1.ipynb"
        build_v1(draft)
        notebook = json.loads(draft.read_text(encoding="utf-8"))

    notebook["metadata"]["colab"]["name"] = "colab_historical_recognizer_v2.ipynb"
    _set_source(
        notebook,
        "scope",
        """# Historical Polish recognizer v2: frozen replay\n
Choose a GPU runtime and click **Run all**. The notebook downloads only pinned
public inputs and executes the single recipe frozen before training in
`docs/HISTORICAL_RECOGNIZER_V2.md`: 258 historical lines repeated twice, 349
EHRI training lines and a deterministic 500-line synthetic replay sample.\n
Historical spelling is preserved. Before training, the notebook audits target
lengths, tokenizer round trips and exact image-hash overlap against all three
evaluation sets. It downloads an evidence ZIP and creates model weights only
if every promotion gate passes. It never publishes either artifact.\n
""",
    )

    setup = f'''import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

CODE_REVISION = {CODE_REVISION!r}
BASE_MODEL = {BASE_MODEL!r}
BASE_REVISION = {BASE_REVISION!r}
HISTORICAL_REVISION = {HISTORICAL_REVISION!r}
EHRI_REVISION = {EHRI_REVISION!r}
SYN_REVISION = {SYN_REVISION!r}

repo = Path('/content/OCR_engine')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/PiotrStyla/OCR_engine.git', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', CODE_REVISION], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', CODE_REVISION], check=True)
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == CODE_REVISION
os.chdir(repo)
sys.path.insert(0, str(repo))

for name in list(sys.modules):
    if (name == 'huggingface_hub' or name.startswith('huggingface_hub.')
            or name == 'transformers' or name.startswith('transformers.')
            or name == 'tokenizers' or name.startswith('tokenizers.')):
        del sys.modules[name]

import torch
assert torch.cuda.is_available(), 'Select a GPU runtime, then Run all.'
assert importlib.metadata.version('transformers') == '4.57.6'
assert importlib.metadata.version('tokenizers') == '0.22.2'
assert importlib.metadata.version('peft') == '0.19.1'
assert importlib.metadata.version('accelerate') == '1.13.0'
assert importlib.metadata.version('jiwer') == '4.0.0'
assert importlib.metadata.version('huggingface_hub') == '0.36.2'
assert importlib.metadata.version('Pillow') == '11.3.0'
assert importlib.metadata.version('opencv-python-headless') == '4.12.0.88'
from transformers.generation import GenerationMixin
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
assert GenerationMixin is not None and TrOCRProcessor is not None and VisionEncoderDecoderModel is not None
print('TRANSFORMERS_IMPORT_PREFLIGHT_OK')

run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
work = Path('/content') / ('historical-recognizer-v2-' + run_id)
work.mkdir()
print('GPU:', torch.cuda.get_device_name(0))
print('Work:', work)
'''
    _set_source(notebook, "setup", setup)

    inputs = '''import hashlib
import shutil
import tarfile
from huggingface_hub import hf_hub_download, snapshot_download
from training.protocol import pair_manifest

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

base_dir = Path(snapshot_download(
    repo_id=BASE_MODEL,
    revision=BASE_REVISION,
    local_dir=work / 'base-mixed-v3',
))

ehri_archive = Path(hf_hub_download(
    'PiotrSty/ehri-pl-lines',
    'ehri-pl-lines-v1.tar.gz',
    repo_type='dataset',
    revision=EHRI_REVISION,
))
ehri_root = work / 'ehri-pl-lines-v1'
ehri_root.mkdir()
with tarfile.open(ehri_archive, 'r:gz') as archive:
    archive.extractall(ehri_root, filter='data')

syn_archive = Path(hf_hub_download(
    'PiotrSty/ocr-pl-lines',
    'ocr-pl-lines-v1.tar.gz',
    repo_type='dataset',
    revision=SYN_REVISION,
))
syn_root = work / 'ocr-pl-lines-v1'
syn_root.mkdir()
with tarfile.open(syn_archive, 'r:gz') as archive:
    archive.extractall(syn_root, filter='data')

historical_train = pair_manifest(corpus_root / 'train')
ehri_train = pair_manifest(ehri_root / 'train')
synthetic_train = pair_manifest(syn_root / 'train')
assert len(historical_train) == 258
assert len(ehri_train) == 349
assert len(synthetic_train) == 2000

selected = sorted(
    synthetic_train,
    key=lambda row: hashlib.sha256(
        ('historical-replay-v2:' + row['id']).encode('utf-8')
    ).hexdigest(),
)[:500]
replay_root = work / 'synthetic-replay-500'
replay_root.mkdir()
selected_manifest = []
for rank, row in enumerate(selected):
    for suffix in ('.png', '.txt'):
        shutil.copyfile(
            syn_root / 'train' / (row['id'] + suffix),
            replay_root / (row['id'] + suffix),
        )
    selected_manifest.append({
        'selection_rank': rank,
        'id': row['id'],
        'selection_sha256': hashlib.sha256(
            ('historical-replay-v2:' + row['id']).encode('utf-8')
        ).hexdigest(),
        'image_sha256': row['image_sha256'],
        'text_sha256': row['text_sha256'],
    })
synthetic_replay = pair_manifest(replay_root)
assert len(synthetic_replay) == 500

evaluation_sets = {
    'historical-validation': corpus_root / 'validation',
    'real-lines-v1': repo / 'benchmarks/real-lines-v1/pairs',
    'ehri-test': ehri_root / 'test',
}
evaluation_counts = {
    name: len(pair_manifest(path)) for name, path in evaluation_sets.items()
}
assert evaluation_counts == {
    'historical-validation': 139,
    'real-lines-v1': 75,
    'ehri-test': 81,
}

train_domains = {
    'historical-train': historical_train,
    'ehri-train': ehri_train,
    'synthetic-replay': synthetic_replay,
}
seen_train_hashes = {}
for domain, rows in train_domains.items():
    for row in rows:
        previous = seen_train_hashes.setdefault(row['image_sha256'], (domain, row['id']))
        if previous != (domain, row['id']):
            raise ValueError(
                f'Exact train image overlap: {previous} / {(domain, row["id"])}'
            )

overlap_audit = {}
for train_name, train_rows in train_domains.items():
    train_hashes = {row['image_sha256'] for row in train_rows}
    for eval_name, eval_path in evaluation_sets.items():
        eval_rows = pair_manifest(eval_path)
        overlap = sorted(train_hashes & {row['image_sha256'] for row in eval_rows})
        overlap_audit[f'{train_name}::{eval_name}'] = overlap
        if overlap:
            raise ValueError(f'Exact train/evaluation image overlap: {train_name} / {eval_name}')

replay_manifest = {
    'schema': 'ocr-engine-historical-replay-v2',
    'code_revision': CODE_REVISION,
    'historical_revision': HISTORICAL_REVISION,
    'ehri_revision': EHRI_REVISION,
    'synthetic_revision': SYN_REVISION,
    'source_archive_sha256': {
        'ehri': sha256_file(ehri_archive),
        'synthetic': sha256_file(syn_archive),
    },
    'selection': "sort by sha256('historical-replay-v2:' + id), take first 500",
    'counts': {
        'historical_distinct': 258,
        'historical_repeats': 2,
        'ehri': 349,
        'synthetic_source': 2000,
        'synthetic_selected': 500,
        'distinct_train': 1107,
        'effective_examples_per_epoch': 1365,
        'evaluation': evaluation_counts,
    },
    'exact_image_overlap': overlap_audit,
    'selected_synthetic': selected_manifest,
}
(work / 'replay-manifest.json').write_text(
    json.dumps(replay_manifest, ensure_ascii=False, indent=2) + '\\n',
    encoding='utf-8',
)
assert all(not values for values in overlap_audit.values())
print(json.dumps(replay_manifest['counts'], indent=2))
print('Exact image-overlap audit passed.')
'''
    _set_source(notebook, "inputs", inputs)

    tokenizer = '''import unicodedata
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
tokenizer_rows = []
roundtrip_mismatches = []
audit_sets = {
    'train-historical': corpus_root / 'train',
    'train-ehri': ehri_root / 'train',
    'train-synthetic': replay_root,
    'eval-historical': corpus_root / 'validation',
    'eval-real-lines': repo / 'benchmarks/real-lines-v1/pairs',
    'eval-ehri': ehri_root / 'test',
}

def normalize_audit_text(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())

for domain, directory in audit_sets.items():
    for path in sorted(directory.glob('*.txt')):
        text = path.read_text(encoding='utf-8').strip()
        token_ids = processor.tokenizer(text, truncation=False).input_ids
        decoded = processor.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        tokenizer_rows.append({
            'id': path.stem,
            'domain': domain,
            'tokens': len(token_ids),
        })
        if normalize_audit_text(decoded) != normalize_audit_text(text):
            roundtrip_mismatches.append({'domain': domain, 'id': path.stem})

tokenizer_audit = {
    'labels': len(tokenizer_rows),
    'max_tokens': max(row['tokens'] for row in tokenizer_rows),
    'over_128': sum(row['tokens'] > 128 for row in tokenizer_rows),
    'roundtrip_mismatches': roundtrip_mismatches,
    'domain_counts': {
        domain: sum(row['domain'] == domain for row in tokenizer_rows)
        for domain in audit_sets
    },
}
(work / 'tokenizer-audit.json').write_text(
    json.dumps(tokenizer_audit, ensure_ascii=False, indent=2) + '\\n',
    encoding='utf-8',
)
assert tokenizer_audit['labels'] == 1402
assert tokenizer_audit['over_128'] == 0
assert not tokenizer_audit['roundtrip_mismatches']
print(tokenizer_audit)
'''
    _set_source(notebook, "tokenizer", tokenizer)

    cells = notebook["cells"]
    tokenizer_cell = cells.pop(cells.index(_cell(notebook, "tokenizer")))
    inputs_index = cells.index(_cell(notebook, "inputs"))
    cells.insert(inputs_index + 1, tokenizer_cell)

    train = '''model_dir = work / 'historical-recognizer-v2-model'
subprocess.run([
    sys.executable, '-m', 'training.train_trocr_pl',
    '--train-dir',
    str(corpus_root / 'train'),
    str(corpus_root / 'train'),
    str(ehri_root / 'train'),
    str(replay_root),
    '--val-dir', str(corpus_root / 'validation'),
    '--base', str(base_dir),
    '--output', str(model_dir),
    '--epochs', '4',
    '--batch-size', '4',
    '--gradient-accumulation-steps', '2',
    '--lr', '2e-5',
    '--lora-rank', '16',
    '--lora-alpha', '32',
    '--max-target-length', '128',
    '--no-4bit',
    '--seed', '42',
], check=True)
print((model_dir / 'selection.json').read_text())
print((model_dir / 'best_metrics.json').read_text())
'''
    _set_source(notebook, "train", train)

    evidence = '''import hashlib
import shutil

model_archive = None
model_archive_sha256 = None
if promotion['all_gates_passed']:
    model_package_dir = work / 'historical-recognizer-v2-package'
    model_package_dir.mkdir()
    for source in sorted(model_dir.iterdir()):
        if source.is_file():
            shutil.copyfile(source, model_package_dir / source.name)
    model_archive = Path(shutil.make_archive(
        str(work / 'historical-recognizer-v2-model'),
        'zip',
        model_package_dir,
    ))
    model_archive_sha256 = hashlib.sha256(model_archive.read_bytes()).hexdigest()

evidence_dir = work / 'evidence'
evidence_dir.mkdir()
files_to_copy = {
    corpus_root / 'report.json': 'corpus-report.json',
    corpus_root / 'manifest.jsonl': 'corpus-manifest.jsonl',
    corpus_root / 'quarantine.jsonl': 'corpus-quarantine.jsonl',
    repo / 'experiments/2026-09-27/historical-recognizer-v2/config.json': 'experiment-config.json',
    work / 'replay-manifest.json': 'replay-manifest.json',
    work / 'tokenizer-audit.json': 'tokenizer-audit.json',
    work / 'baseline-metrics.json': 'baseline-metrics.json',
    work / 'candidate-metrics.json': 'candidate-metrics.json',
    work / 'promotion-gates.json': 'promotion-gates.json',
    model_dir / 'run.json': 'training-run.json',
    model_dir / 'selection.json': 'selection.json',
    model_dir / 'best_metrics.json': 'best-metrics.json',
}
for source, name in files_to_copy.items():
    shutil.copyfile(source, evidence_dir / name)

environment = {
    'code_revision': CODE_REVISION,
    'base_model': BASE_MODEL,
    'base_revision': BASE_REVISION,
    'historical_revision': HISTORICAL_REVISION,
    'ehri_revision': EHRI_REVISION,
    'synthetic_revision': SYN_REVISION,
    'gpu': torch.cuda.get_device_name(0),
    'python': sys.version,
    'packages': {
        name: importlib.metadata.version(name)
        for name in ('torch', 'transformers', 'peft', 'accelerate', 'jiwer',
                     'huggingface_hub', 'tokenizers', 'Pillow', 'opencv-python-headless')
    },
    'raw_predictions_included': False,
    'images_or_labels_included': False,
    'model_weights_included': False,
    'model_archive_contents': 'Merged final model and root configuration files; no checkpoints or adapter.',
    'model_archive_created': model_archive is not None,
    'model_archive_sha256': model_archive_sha256,
}
(evidence_dir / 'environment.json').write_text(json.dumps(environment, indent=2) + '\\n')
checksums = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(evidence_dir.iterdir())
    if path.is_file()
}
(evidence_dir / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\\n')
evidence_zip = Path(shutil.make_archive(
    str(work / 'historical-recognizer-v2-evidence'),
    'zip',
    evidence_dir,
))
print('Evidence:', evidence_zip)
if model_archive is None:
    print('Promotion gates failed; model archive was not created.')
else:
    print('Promotion gates passed. Model archive:', model_archive)
'''
    _set_source(notebook, "evidence", evidence)

    Path(target).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build(Path(__file__).with_name("colab_historical_recognizer_v2.ipynb"))
