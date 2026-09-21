"""Offline preflight by default; explicit --execute runs pinned Kraken on CUDA."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from training.stage_impact_benchmark import digest
from training.validate_submission import load_jsonl, unique_ids

KRAKEN_VERSION = '7.1.1'


def load_cases(manifest):
    manifest = Path(manifest)
    rows = load_jsonl(manifest)
    if not rows:
        raise ValueError('Empty manifest')
    unique_ids(rows, 'Manifest')
    cases = []
    for row in rows:
        path = (manifest.parent / row['image']).resolve()
        if digest(path) != row['sha256']:
            raise ValueError(f"Image checksum mismatch: {row['id']}")
        # Do not expose reference text to the inference function.
        cases.append({'id': row['id'], 'path': path, 'sha256': row['sha256']})
    return cases


def run_cases(cases, predict, output, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    run = {**metadata, 'started_utc': datetime.now(timezone.utc).isoformat(),
           'pages_expected': len(cases), 'pages_completed': 0, 'error_pages': 0, 'state': 'running'}

    def save_state():
        temporary = output / 'run.json.tmp'
        temporary.write_text(json.dumps(run, indent=2), encoding='utf-8', newline='\n')
        temporary.replace(output / 'run.json')

    save_state()
    try:
        with (output / 'predictions.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for case in cases:
                started = time.perf_counter()
                record = {'id': case['id'], 'source_sha256': case['sha256']}
                try:
                    if digest(case['path']) != case['sha256']:
                        raise ValueError('Image changed since preflight')
                    text = predict(case['path'])
                    if not isinstance(text, str):
                        raise TypeError('Recognizer must return text')
                    record.update(status='ok', text=text)
                except Exception as error:
                    record.update(status='error', text='', error_type=type(error).__name__)
                    run['error_pages'] += 1
                record['elapsed_seconds'] = time.perf_counter() - started
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                run['pages_completed'] += 1
                save_state()
                print(f"{record['id']}: {record['status']}", flush=True)
        run['state'] = 'completed'
    except BaseException:
        run['state'] = 'interrupted'
        raise
    finally:
        run['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if (output / 'predictions.jsonl').exists():
            run['predictions_sha256'] = digest(output / 'predictions.jsonl')
        save_state()
    return run


def create_predictor(recognizer, segmenter):
    if importlib.metadata.version('kraken') != KRAKEN_VERSION:
        raise RuntimeError(f'Expected kraken=={KRAKEN_VERSION}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from PIL import Image
    from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel
    from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; no implicit laptop CPU fallback')
    torch.ones(1, device='cuda').sum().item()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    segmentation = SegmentationTaskModel.load_model(str(segmenter))
    recognition = RecognitionTaskModel.load_model(str(recognizer))
    seg_config = SegmentationInferenceConfig(accelerator='cuda', device=[0])
    rec_config = RecognitionInferenceConfig(accelerator='cuda', device=[0])

    def predict(path):
        with Image.open(path) as source:
            source.seek(0)
            image = source.convert('L')
        with torch.inference_mode():
            regions = segmentation.predict(image, seg_config)
            records = recognition.predict(image, regions, rec_config)
            return '\n'.join(record.prediction for record in records)

    return predict, {'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                     'segmentation_config': repr(seg_config), 'recognition_config': repr(rec_config),
                     'seed': 0, 'deterministic_algorithms': True,
                     'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
                     'installed_distributions': sorted([
                         {'name': dist.metadata['Name'], 'version': dist.version}
                         for dist in importlib.metadata.distributions() if dist.metadata['Name']],
                         key=lambda item: item['name'].lower()),
                     'packages': {name: importlib.metadata.version(name) for name in ('kraken', 'torch', 'Pillow')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--recognizer')
    parser.add_argument('--recognizer-sha256')
    parser.add_argument('--segmenter')
    parser.add_argument('--segmenter-sha256')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Use a new output directory')
    cases = load_cases(args.manifest)
    report = {'engine': 'kraken', 'required_version': KRAKEN_VERSION,
              'manifest_sha256': digest(args.manifest), 'pages': len(cases),
              'runner_sha256': digest(__file__), 'python': platform.python_version(),
              'preprocessing': 'frame 0, grayscale L, no deskew/binarization/resize; native line order',
              'reference_text_sent_to_model': False}
    try:
        report['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
            cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        report['code_commit'] = None
    if not args.execute:
        output = Path(args.output)
        output.mkdir(parents=True)
        report['state'] = 'preflight-only'
        (output / 'preflight.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        return
    for label in ('recognizer', 'segmenter'):
        path, expected = getattr(args, label), getattr(args, label + '_sha256')
        if not path or not expected or digest(path) != expected.lower():
            raise ValueError(f'{label}: explicit local weights and matching SHA-256 required')
        report[label] = {'filename': Path(path).name, 'sha256': expected.lower()}
    predict, environment = create_predictor(args.recognizer, args.segmenter)
    run_cases(cases, predict, args.output, {**report, **environment})


if __name__ == '__main__':
    main()
