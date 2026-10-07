"""Three bounded, paired Colab experiments with protected development slices."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from training.audit_slayer_rfdetr_layout import safe_extract_zip
from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.merge_recognizer_reviewed_pool import verified_package
from training.reviewed_recognizer_selection import normalize, select, summarize
from training.run_reviewed_recognizer_colab import evaluate, preflight
from training.slayer_vision_onnx_smoke import safe_extract_tar

STEM = 'recognizer-reviewed-colab-v2'
CONFIG = Path(__file__).resolve().parents[1] / 'experiments/2026-10-07' / STEM / 'config.json'


def run_logged(command, repo, log_path):
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError('Training failed; inspect ' + log_path.name)


def package_model(directory, archive_path):
    files = {path.name: digest(path) for path in directory.iterdir() if path.is_file()}
    write_json(directory / 'checksums.json', files)
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in (*files, 'checksums.json'):
            archive.write(directory / name, name)


def run(output_root='/content'):
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import TrOCRProcessor
    from training.protocol import pair_manifest

    if not torch.cuda.is_available():
        raise RuntimeError('Select GPU T4 and Run all; CPU training is disabled.')
    repo = Path(__file__).resolve().parents[1]
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    work = Path(output_root) / (STEM + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    evidence = work / 'evidence'
    evidence.mkdir(parents=True)
    model_archive = None
    try:
        shutil.copyfile(CONFIG, evidence / 'experiment-config.json')
        source_commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        write_json(evidence / 'environment.json', dict(code_revision=source_commit, python=sys.version,
            gpu=torch.cuda.get_device_name(0), config_sha256=digest(CONFIG),
            packages={name: importlib.metadata.version(name) for name in
                      ('torch', 'torchao', 'transformers', 'peft', 'accelerate', 'jiwer', 'huggingface_hub')}))
        dataset = Path(hf_hub_download('PiotrSty/slayer-ocr-datasets',
            'data/recognizer-reviewed-colab-input-v1-20261006/recognizer-reviewed-colab-input-v1-20261006.zip',
            repo_type='dataset', revision=cfg['dataset_revision']))
        if digest(dataset) != cfg['dataset_sha256']:
            raise ValueError('Frozen reviewed input ZIP hash mismatch')
        corpus = safe_extract_zip(dataset, work / 'corpus')
        verified_package(corpus)
        if digest(corpus / 'config.json') != cfg['input_config_sha256']:
            raise ValueError('Frozen input policy hash mismatch')
        source = json.loads((corpus / 'config.json').read_text(encoding='utf-8'))
        for name in ('config.json', 'report.json', 'manifest.jsonl', 'source-ledger.jsonl', 'checksums.json'):
            shutil.copyfile(corpus / name, evidence / ('corpus-' + name))
        synthetic = source['synthetic_replay']
        archive = Path(hf_hub_download(synthetic['repo'], synthetic['filename'],
                                     repo_type='dataset', revision=synthetic['revision']))
        if digest(archive) != cfg['synthetic_archive_sha256']:
            raise ValueError('Frozen synthetic archive hash mismatch')
        synthetic_root = work / 'synthetic-source'
        safe_extract_tar(archive, synthetic_root)
        matches = [path for path in synthetic_root.rglob('train')
                   if path.is_dir() and list(path.glob('*.png'))]
        if len(matches) != 1:
            raise ValueError('Expected exactly one synthetic training split')
        synthetic_train = matches[0]
        rows = pair_manifest(synthetic_train)
        if len(rows) != 2000:
            raise ValueError('Synthetic training count drift')
        ordered = sorted(rows, key=lambda row: hashlib.sha256(
            (cfg['synthetic_order_prefix'] + row['id']).encode()).hexdigest())
        replays = {}
        for count in sorted({variant['replay_count'] for variant in cfg['variants']}):
            if not 0 < count <= len(rows):
                raise ValueError('Invalid replay size')
            replay = work / f'replay-{count}'
            replay.mkdir()
            for row in ordered[:count]:
                for suffix in ('.png', '.txt'):
                    shutil.copyfile(synthetic_train / (row['id'] + suffix), replay / (row['id'] + suffix))
            replays[count] = replay
        development = {'historical-development': corpus / 'development',
                       'ordinary-development': repo / 'benchmarks/real-lines-v1/pairs'}
        directories = {'reviewed': corpus / 'train', 'synthetic': synthetic_train, **development}
        manifests = {name: pair_manifest(path) for name, path in directories.items()}
        if {name: len(rows) for name, rows in manifests.items()} != {
                'reviewed': 70, 'synthetic': 2000, 'historical-development': 9, 'ordinary-development': 75}:
            raise ValueError('Training/development counts changed')
        train_hashes = [row['image_sha256'] for name in ('reviewed', 'synthetic') for row in manifests[name]]
        dev_hashes = [row['image_sha256'] for name in development for row in manifests[name]]
        if (len(set(train_hashes)) != len(train_hashes) or len(set(dev_hashes)) != len(dev_hashes)
                or set(train_hashes) & set(dev_hashes)):
            raise ValueError('Exact training duplicate or development overlap')
        write_json(evidence / 'input-manifests.json', manifests)
        write_json(evidence / 'replay-manifests.json', {
            str(count): pair_manifest(path) for count, path in replays.items()})
        base = snapshot_download(source['base_model'], revision=source['base_revision'],
            local_dir=work / 'base', allow_patterns=['*.json', '*.safetensors', '*.bin', '*.txt', '*.model'])
        processor = TrOCRProcessor.from_pretrained(base)
        recipe = cfg['training']
        token_audit = []
        for domain, directory in directories.items():
            for path in sorted(directory.glob('*.txt')):
                text = path.read_text(encoding='utf-8').strip()
                ids = processor.tokenizer(text, truncation=False).input_ids
                decoded = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if len(ids) > recipe['max_target_length'] or normalize(decoded) != normalize(text):
                    raise ValueError(f'Tokenizer length/round-trip failure: {domain}/{path.stem}')
                token_audit.append(dict(domain=domain, id=path.stem, tokens=len(ids)))
        write_rows(evidence / 'tokenizer-audit.jsonl', token_audit)
        write_json(evidence / 'adapter-preflight.json', preflight(base, corpus / 'train', processor, recipe))
        print('FULL_MODEL_ADAPTER_AND_EVAL_LOSS_PREFLIGHT_OK', flush=True)
        baseline_path = evidence / 'baseline-predictions.jsonl'
        evaluate(base, development, processor, baseline_path)
        baseline_rows = read_rows(baseline_path)
        baseline = summarize(baseline_rows)
        write_json(evidence / 'baseline-metrics.json', baseline)
        policy_path = evidence / 'selection-policy.json'
        write_json(policy_path, dict(baseline=baseline, policy=cfg['selection']))
        validation = work / 'validation'
        validation.mkdir()
        for domain, directory in development.items():
            for path in directory.glob('*'):
                if path.suffix in ('.png', '.txt'):
                    shutil.copyfile(path, validation / (domain + '__' + path.name))
        candidates = []
        for variant in cfg['variants']:
            identifier = variant['id']
            model = work / identifier
            print('VARIANT', identifier, flush=True)
            command = [sys.executable, '-m', 'training.train_trocr_pl', '--train-dir',
                *([str(corpus / 'train')] * recipe['historical_repeats']), str(replays[variant['replay_count']]),
                '--val-dir', str(validation), '--base', str(base), '--output', str(model),
                '--epochs', str(recipe['epochs']), '--batch-size', str(recipe['batch_size']),
                '--gradient-accumulation-steps', str(recipe['gradient_accumulation_steps']),
                '--lr', str(variant['learning_rate']), '--lora-rank', str(recipe['lora_rank']),
                '--lora-alpha', str(recipe['lora_alpha']), '--max-target-length', str(recipe['max_target_length']),
                '--seed', str(recipe['seed']), '--selection-policy', str(policy_path), '--no-4bit']
            run_logged(command, repo, evidence / (identifier + '-training.log'))
            predictions_path = evidence / (identifier + '-predictions.jsonl')
            evaluate(model, development, processor, predictions_path)
            candidate_rows = read_rows(predictions_path)
            if [(row['domain'], row['id'], row['reference']) for row in candidate_rows] != [
                    (row['domain'], row['id'], row['reference']) for row in baseline_rows]:
                raise ValueError('Paired prediction identities/references changed')
            metrics = summarize(candidate_rows)
            candidates.append(dict(id=identifier, metrics=metrics))
            write_json(evidence / (identifier + '-metrics.json'), metrics)
            for name in ('run.json', 'selection.json', 'best_metrics.json', 'trainer_state.json'):
                shutil.copyfile(model / name, evidence / (identifier + '-' + name))
            for path in model.glob('checkpoint-evaluation-*.jsonl'):
                shutil.copyfile(path, evidence / (identifier + '-' + path.name))
            write_json(evidence / 'partial-selection.json', select(baseline, candidates))
        selection = select(baseline, candidates)
        selection.update(base_model=source['base_model'], base_revision=source['base_revision'],
                         input_dataset_revision=cfg['dataset_revision'], input_dataset_sha256=cfg['dataset_sha256'],
                         old_v1_eval_loss_comparable=False, generation_references_sent_to_model=False)
        write_json(evidence / 'selection.json', selection)
        if not selection['baseline_retained']:
            selected = work / 'selected-model'
            selected.mkdir()
            for path in (work / selection['selected']).iterdir():
                if path.is_file() and path.suffix in ('.safetensors', '.json', '.txt', '.model'):
                    shutil.copyfile(path, selected / path.name)
            write_json(selected / 'guarded-selection.json', selection)
            model_archive = work / (STEM + '-model.zip')
            package_model(selected, model_archive)
        print('SELECTED', selection['selected'], flush=True)
        print('Model bazowy pozostaje dostepny; nie jest to promocja produkcyjna ani SOTA.', flush=True)
    except Exception as exc:
        write_json(evidence / 'failure.json', dict(type=type(exc).__name__, message=str(exc),
            completed=False, production_promoted=False, sota_claim=False))
        raise
    finally:
        write_json(evidence / 'checksums.json', {path.name: digest(path) for path in evidence.iterdir()
                                               if path.is_file() and path.name != 'checksums.json'})
        evidence_archive = Path(shutil.make_archive(str(work / (STEM + '-evidence')), 'zip', evidence))
        print('EVIDENCE_ZIP', evidence_archive, flush=True)
    payloads = [evidence_archive] + ([model_archive] if model_archive else [])
    write_json(work / 'result-checksums.json', {path.name: digest(path) for path in payloads})
    with zipfile.ZipFile(work / (STEM + '-result.zip'), 'x', zipfile.ZIP_STORED) as archive:
        for path in (*payloads, work / 'result-checksums.json'):
            archive.write(path, path.name)
    return work


if __name__ == '__main__':
    print('RESULT_DIRECTORY', run(), flush=True)
