"""Run bounded exploratory TrOCR LoRA training on pinned human-reviewed inputs."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unicodedata
import zipfile

from training.audit_slayer_rfdetr_layout import safe_extract_zip
from training.full_page_pilot import digest, write_json, write_rows
from training.merge_recognizer_reviewed_pool import verified_package
from training.slayer_vision_onnx_smoke import safe_extract_tar


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def evaluate(model_path, directories, processor, output):
    import torch
    from jiwer import cer, wer
    from PIL import Image
    from transformers import VisionEncoderDecoderModel
    from training.protocol import configure_generation

    model = VisionEncoderDecoderModel.from_pretrained(model_path).to('cuda').eval()
    configure_generation(model, processor.tokenizer)
    results, predictions = {}, []
    for name, directory in directories.items():
        refs, hyps = [], []
        for path in sorted(Path(directory).glob('*.png')):
            reference = path.with_suffix('.txt').read_text(encoding='utf-8')
            with Image.open(path) as image:
                pixels = processor(images=image.convert('RGB'), return_tensors='pt').pixel_values.to('cuda')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                generated = model.generate(pixels)
            hypothesis = processor.tokenizer.decode(generated[0], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            refs.append(normalize(reference))
            hyps.append(normalize(hypothesis))
            predictions.append({'domain': name, 'id': path.stem, 'reference': reference,
                                'prediction': hypothesis})
        if not refs:
            raise ValueError('Empty development domain')
        results[name] = {'cer': cer(refs, hyps), 'wer': wer(refs, hyps), 'lines': len(refs)}
    write_rows(output, predictions)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run(dataset_revision, dataset_sha256, output_root='/content'):
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import TrOCRProcessor
    from training.protocol import pair_manifest

    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab, then Run all; CPU training is disabled.')
    work = Path(output_root)/('recognizer-reviewed-colab-v1-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    work.mkdir(parents=True)
    evidence = work/'evidence'
    evidence.mkdir()
    try:
        dataset = Path(hf_hub_download('PiotrSty/slayer-ocr-datasets',
            'data/recognizer-reviewed-colab-input-v1-20261006/recognizer-reviewed-colab-input-v1-20261006.zip',
            repo_type='dataset', revision=dataset_revision))
        if digest(dataset) != dataset_sha256:
            raise ValueError('Pinned training ZIP hash mismatch')
        corpus = safe_extract_zip(dataset, work/'corpus')
        verified_package(corpus)
        cfg = json.loads((corpus/'config.json').read_text(encoding='utf-8'))
        code_config = Path(__file__).resolve().parents[1]/'experiments/2026-10-06/recognizer-reviewed-colab-v1/config.json'
        if digest(code_config) != digest(corpus/'config.json'):
            raise ValueError('Code/data training policy mismatch')
        for name in ('config.json', 'report.json', 'manifest.jsonl', 'source-ledger.jsonl', 'checksums.json'):
            shutil.copyfile(corpus/name, evidence/('corpus-'+name))
        synthetic = cfg['synthetic_replay']
        archive = Path(hf_hub_download(synthetic['repo'], synthetic['filename'],
            repo_type='dataset', revision=synthetic['revision']))
        synthetic_root = work/'synthetic-source'
        safe_extract_tar(archive, synthetic_root)
        matches = [p for p in synthetic_root.rglob('train') if p.is_dir() and list(p.glob('*.png'))]
        if len(matches) != 1:
            raise ValueError('Expected one synthetic training split')
        synthetic_train = matches[0]
        candidates = pair_manifest(synthetic_train)
        if len(candidates) != 2000:
            raise ValueError('Synthetic source count drift')
        selected = sorted(candidates, key=lambda r: hashlib.sha256(
            ('reviewed-colab-v1:'+r['id']).encode()).hexdigest())[:synthetic['count']]
        replay = work/'replay'
        replay.mkdir()
        for row in selected:
            for suffix in ('.png', '.txt'):
                shutil.copyfile(synthetic_train/(row['id']+suffix), replay/(row['id']+suffix))
        repo = Path(__file__).resolve().parents[1]
        ordinary = repo/'benchmarks/real-lines-v1/pairs'
        development = {'historical-development': corpus/'development', 'ordinary-development': ordinary}
        train = {'reviewed': corpus/'train', 'synthetic': replay}
        manifests = {name: pair_manifest(path) for name, path in {**train, **development}.items()}
        train_hashes = [r['image_sha256'] for name in train for r in manifests[name]]
        dev_hashes = [r['image_sha256'] for name in development for r in manifests[name]]
        if len(train_hashes) != len(set(train_hashes)) or set(train_hashes) & set(dev_hashes):
            raise ValueError('Exact training duplicate or development overlap')
        if {name: len(rows) for name, rows in manifests.items()} != {
                'reviewed': 70, 'synthetic': 500, 'historical-development': 9, 'ordinary-development': 75}:
            raise ValueError('Training/development count drift')
        base = snapshot_download(cfg['base_model'], revision=cfg['base_revision'],
            local_dir=work/'base', allow_patterns=['*.json', '*.safetensors', '*.bin', '*.txt', '*.model'])
        processor = TrOCRProcessor.from_pretrained(base)
        token_audit = []
        for domain, directory in {**train, **development}.items():
            for path in sorted(directory.glob('*.txt')):
                text = path.read_text(encoding='utf-8').strip()
                ids = processor.tokenizer(text, truncation=False).input_ids
                decoded = processor.tokenizer.decode(ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)
                if len(ids) > cfg['training']['max_target_length'] or normalize(text) != normalize(decoded):
                    raise ValueError(f'Tokenizer length/round-trip failure: {domain}/{path.stem}')
                token_audit.append({'domain': domain, 'id': path.stem, 'tokens': len(ids)})
        write_rows(evidence/'tokenizer-audit.jsonl', token_audit)
        write_json(evidence/'input-manifests.json', manifests)
        write_json(evidence/'environment.json', {'gpu': torch.cuda.get_device_name(0), 'python': sys.version,
            'code_revision': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
            'dataset_revision': dataset_revision, 'dataset_sha256': dataset_sha256,
            'synthetic_archive_sha256': digest(archive), 'base_revision': cfg['base_revision'],
            'packages': {n: importlib.metadata.version(n) for n in
                ('torch', 'transformers', 'peft', 'accelerate', 'jiwer', 'huggingface_hub')}})
        baseline = evaluate(base, development, processor, evidence/'baseline-predictions.jsonl')
        write_json(evidence/'baseline-metrics.json', baseline)
        validation = work/'validation'
        validation.mkdir()
        for domain, directory in development.items():
            for path in directory.glob('*'):
                if path.suffix in ('.png', '.txt'):
                    shutil.copyfile(path, validation/(domain+'__'+path.name))
        recipe = cfg['training']
        model = work/'model'
        command = [sys.executable, '-m', 'training.train_trocr_pl', '--train-dir',
            *([str(corpus/'train')]*recipe['historical_repeats']), str(replay),
            '--val-dir', str(validation), '--base', str(base), '--output', str(model),
            '--epochs', str(recipe['epochs']), '--batch-size', str(recipe['batch_size']),
            '--gradient-accumulation-steps', str(recipe['gradient_accumulation_steps']),
            '--lr', str(recipe['learning_rate']), '--lora-rank', str(recipe['lora_rank']),
            '--lora-alpha', str(recipe['lora_alpha']), '--max-target-length', str(recipe['max_target_length']),
            '--seed', str(recipe['seed']), '--no-4bit']
        with (evidence/'training.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError('Training failed; inspect training.log in the evidence ZIP.')
        candidate = evaluate(model, development, processor, evidence/'candidate-predictions.jsonl')
        write_json(evidence/'candidate-metrics.json', candidate)
        for name in ('run.json', 'selection.json', 'best_metrics.json'):
            shutil.copyfile(model/name, evidence/('training-'+name))
        write_json(evidence/'result.json', {'status': 'trained-experimental-candidate',
            'historical_cer_delta': candidate['historical-development']['cer']-baseline['historical-development']['cer'],
            'ordinary_cer_delta': candidate['ordinary-development']['cer']-baseline['ordinary-development']['cer'],
            'independent_benchmark': False, 'sota_claim': False, 'production_promoted': False,
            'selection_uses_development_data': True})
        model_files = {p.name: digest(p) for p in model.iterdir() if p.is_file()}
        write_json(model/'checksums.json', model_files)
        with zipfile.ZipFile(work/'recognizer-reviewed-colab-v1-model.zip', 'x', zipfile.ZIP_DEFLATED) as stream:
            for name in [*model_files, 'checksums.json']:
                stream.write(model/name, name)
    except Exception as exc:
        write_json(evidence/'failure.json', {'type': type(exc).__name__, 'message': str(exc),
            'training_success': False, 'sota_claim': False})
        raise
    finally:
        write_json(evidence/'checksums.json', {p.name: digest(p) for p in evidence.iterdir()
            if p.is_file() and p.name != 'checksums.json'})
        archive = Path(shutil.make_archive(str(work/'recognizer-reviewed-colab-v1-evidence'), 'zip', evidence))
        print('EVIDENCE_ZIP', archive, flush=True)
    result_files = [work/'recognizer-reviewed-colab-v1-model.zip', archive]
    write_json(work/'result-checksums.json', {p.name: digest(p) for p in result_files})
    with zipfile.ZipFile(work/'recognizer-reviewed-colab-v1-result.zip', 'x', zipfile.ZIP_STORED) as stream:
        for path in [*result_files, work/'result-checksums.json']:
            stream.write(path, path.name)
    return work


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-revision', required=True)
    parser.add_argument('--dataset-sha256', required=True)
    parser.add_argument('--output-root', default='/content')
    print('RESULT_DIRECTORY', run(**vars(parser.parse_args())), flush=True)
