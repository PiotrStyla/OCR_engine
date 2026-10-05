"""Two-family OCR proposals on training-source crops, never automatic gold labels."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import stat
import sys
import time
import zipfile

from training.benchmark_pages import normalize
from training.full_page_pilot import digest, read_rows, safe_relative, validate_inputs, write_json, write_rows

ENGINES = ('qwen3-vl-4b', 'trocr-mixed-v3')
FILES = {'qwen3-vl-4b': 'qwen3-vl-4b-full-page.jsonl', 'trocr-mixed-v3': 'trocr-lines.jsonl'}


def stage(config, archive, output):
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise FileExistsError('Use a new staged input directory')
    if digest(archive) != config['dataset']['archive_sha256']:
        raise ValueError('Training-source bundle checksum mismatch')
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or sum(item.file_size for item in members) > 50_000_000:
            raise ValueError('Duplicate or oversized input bundle')
        for item in members:
            if safe_relative(item.filename) != item.filename:
                raise ValueError('Noncanonical input member')
            if item.is_dir() or stat.S_ISLNK(item.external_attr >> 16) or Path(item.filename).suffix not in ('.png', '.json', '.jsonl'):
                raise ValueError('Unsafe or unexpected input member')
        payload = {name: stream.read(name) for name in names}
        checksums = json.loads(payload['checksums.json'])
        import hashlib
        if set(checksums) != set(names)-{'checksums.json'} or any(
                hashlib.sha256(payload[name]).hexdigest() != sha for name, sha in checksums.items()):
            raise ValueError('Incomplete or incorrect bundle checksums')
        if hashlib.sha256(payload['manifest.jsonl']).hexdigest() != config['dataset']['manifest_sha256']:
            raise ValueError('Frozen training candidate manifest mismatch')
        rows = [json.loads(line) for line in payload['manifest.jsonl'].splitlines() if line.strip()]
        if len(rows) != config['dataset']['lines'] or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Unexpected line coverage')
        for row in rows:
            if (row['source_split'] != 'train' or row['split'] != 'training-review'
                    or row['eligible_for_training'] is not False or row['final_test'] is not False
                    or row['collection'] in config['dataset']['forbidden_collections']
                    or row['dataset'] != config['dataset']['source_repo']
                    or row['revision'] != config['dataset']['source_revision']):
                raise ValueError('Validation/test data or unapproved training label')
            image = safe_relative(row['image'])
            if hashlib.sha256(payload[image]).hexdigest() != row['sha256']:
                raise ValueError('Crop hash mismatch')
            from PIL import Image
            import io
            with Image.open(io.BytesIO(payload[image])) as scan:
                if scan.size != (row['width'], row['height']):
                    raise ValueError('Crop dimensions mismatch')
        expected = [{key: row[key] for key in ('id', 'image', 'sha256', 'width', 'height')}
                    | {'source_regions': []} for row in rows]
        actual = [json.loads(line) for line in payload['inference-inputs.jsonl'].splitlines() if line.strip()]
        if actual != expected:
            raise ValueError('Reference-free worker inputs mismatch')
    output.mkdir(parents=True)
    for name, data in payload.items():
        path = output/safe_relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    validate_inputs(output/'inference-inputs.jsonl')
    return {'lines': len(rows), 'labels_sent_to_models': False,
            'training_examples_created': 0, 'input_sha256': digest(output/'inference-inputs.jsonl')}


def load_trocr(spec):
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    if spec['dtype'] != 'float32':
        raise ValueError('This pilot requires FP32 TrOCR inference')
    folder = Path(snapshot_download(spec['repo'], revision=spec['revision']))
    if digest(folder/'model.safetensors') != spec['weights_sha256']:
        raise ValueError('TrOCR checkpoint hash mismatch')
    processor = TrOCRProcessor.from_pretrained(folder)
    model = VisionEncoderDecoderModel.from_pretrained(folder, torch_dtype=torch.float32).to('cuda').eval()
    eos, pad = processor.tokenizer.sep_token_id, processor.tokenizer.pad_token_id
    start = model.generation_config.decoder_start_token_id
    if any(value is None for value in (start, eos, pad)):
        raise ValueError('Explicit TrOCR decoder/EOS/PAD tokens required')

    def infer(path, _row):
        with Image.open(path) as scan:
            image = scan.convert('RGB')
        pixels = processor(images=image, return_tensors='pt').pixel_values.to('cuda')
        with torch.inference_mode():
            output = model.generate(pixels, max_new_tokens=spec['max_new_tokens'], num_beams=spec['num_beams'],
                do_sample=False, early_stopping=True, eos_token_id=eos, pad_token_id=pad,
                decoder_start_token_id=start)
        sequence = output[0].tolist()
        if not sequence or sequence[0] != start:
            raise ValueError('TrOCR decoder start mismatch')
        tokens = sequence[1:]
        ended = eos in tokens
        if ended and any(token != pad for token in tokens[tokens.index(eos)+1:]):
            raise ValueError('Unexpected tokens after TrOCR EOS')
        capped = len(tokens) >= spec['max_new_tokens'] and not ended
        return {'text': processor.decode(sequence, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False), 'finish_reason': 'eos' if ended else ('length' if capped else 'unknown'),
            'generated_tokens': len(tokens), 'token_limit_reached': capped,
            'generation_trace': {'generated_token_ids': tokens, 'eos_token_ids': [eos]},
            'input_geometry': {'original_width': image.width, 'original_height': image.height,
                               'processor_tensor_shape': list(pixels.shape)}}
    return infer


def worker(engine, config, inputs, output):
    if engine == 'qwen3-vl-4b':
        from training.full_page_pilot import run_worker
        return run_worker(engine, config, inputs, output)
    if engine != 'trocr-mixed-v3':
        raise ValueError('Unknown teacher')
    import torch
    inputs, output = Path(inputs), Path(output)
    rows = validate_inputs(inputs)
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    spec = config['models'][engine]
    identity = {'engine': engine, 'spec': spec, 'input_sha256': digest(inputs), 'runner_sha256': digest(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text()) != identity:
        raise ValueError('Existing worker identity mismatch')
    write_json(output/'identity.json', identity)
    destination = output/FILES[engine]
    existing = read_rows(destination) if destination.exists() else []
    if len({row['id'] for row in existing}) != len(existing) or {row['id'] for row in existing}-{row['id'] for row in rows}:
        raise ValueError('Invalid saved worker coverage')
    pending = [row for row in rows if row['id'] not in {item['id'] for item in existing}]
    if not pending:
        return {'lines': len(rows), 'errors': sum(row['status'] != 'ok' for row in existing)}
    torch.manual_seed(42)
    started, load_error = time.perf_counter(), None
    try:
        infer = load_trocr(spec)
    except Exception as error:
        infer, load_error = None, f'{type(error).__name__}: {error}'
    load_seconds = time.perf_counter()-started
    for index, row in enumerate(pending, 1):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            if load_error:
                raise RuntimeError(load_error)
            torch.cuda.synchronize()
            prediction = {'status': 'ok', **infer(inputs.parent/row['image'], row)}
            torch.cuda.synchronize()
        except Exception as error:
            prediction = {'status': 'error', 'text': '', 'error': f'{type(error).__name__}: {error}'}
        record = {'id': row['id'], 'elapsed_seconds': time.perf_counter()-started,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(), **prediction}
        with destination.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            stream.flush()
        print(f'{engine} {index}/{len(pending)} {row["id"]}', flush=True)
    write_json(output/'environment.json', {'gpu': torch.cuda.get_device_name(0), 'python': sys.version,
        'load_error': load_error, 'model_load_seconds': load_seconds,
        'reference_text_sent_to_model': False,
        'packages': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'tokenizers', 'huggingface_hub', 'Pillow')}})
    return {'lines': len(rows), 'errors': sum(row['status'] != 'ok' for row in read_rows(destination))}


def healthy(row):
    return (row.get('status') == 'ok' and row.get('finish_reason') == 'eos'
            and row.get('token_limit_reached') is False and isinstance(row.get('text'), str)
            and bool(normalize(row['text'])))


def combine(config, dataset, prediction_root, output):
    dataset, prediction_root, output = map(Path, (dataset, prediction_root, output))
    rows = read_rows(dataset/'manifest.jsonl')
    if digest(dataset/'manifest.jsonl') != config['dataset']['manifest_sha256']:
        raise ValueError('Changed training-source manifest')
    ids = {row['id'] for row in rows}
    predictions = {}
    for engine in ENGINES:
        directory = prediction_root/engine
        identity_path = directory/'identity.json'
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding='utf-8'))
            runner = Path(__file__).with_name('full_page_pilot.py') if engine == 'qwen3-vl-4b' else Path(__file__)
            if (identity.get('engine') != engine or identity.get('runner_sha256') != digest(runner)
                    or identity.get('spec') != config['models'][engine]
                    or identity.get('input_sha256') != digest(dataset/'inference-inputs.jsonl')):
                raise ValueError('Teacher provenance mismatch')
        path = directory/FILES[engine]
        raw = read_rows(path) if path.exists() else []
        if raw and not identity_path.exists():
            raise ValueError('Predictions without teacher identity')
        if len(raw) != len({row['id'] for row in raw}) or {row['id'] for row in raw}-ids:
            raise ValueError('Duplicate or unknown teacher predictions')
        predictions[engine] = {row['id']: row for row in raw}
    output.mkdir(parents=True, exist_ok=True)
    proposals = []
    for row in rows:
        votes = {engine: predictions[engine].get(row['id'], {'status': 'missing', 'text': ''}) for engine in ENGINES}
        valid = all(healthy(vote) for vote in votes.values())
        texts = [normalize(vote['text']) if isinstance(vote.get('text'), str) else '' for vote in votes.values()]
        status = ('teacher-agreement-proposal' if texts[0] == texts[1] else 'teacher-disagreement') if valid else 'teacher-abstention'
        proposal = {'id': row['id'], 'source_split': 'train', 'collection': row['collection'],
            'page_id': row['page_id'], 'image_sha256': row['sha256'], 'source_text': row['text'],
            'teachers': votes, 'status': status,
            'proposed_text': votes[ENGINES[0]]['text'] if status == 'teacher-agreement-proposal' else None,
            'source_agreement': valid and all(text == normalize(row['text']) for text in texts),
            'review_required': True, 'line_geometry_verified': False,
            'eligible_for_training': False, 'gold_label': False}
        proposals.append(proposal)
    write_rows(output/'proposals.jsonl', proposals)
    report = {'schema': 'slayer-recognizer-data-teacher-pilot-v1', 'lines': len(rows),
        'statuses': dict(Counter(row['status'] for row in proposals)),
        'source_agreement_proposals': sum(row['source_agreement'] for row in proposals),
        'teacher_health': {engine: {'returned': len(predictions[engine]),
            'healthy': sum(healthy(row) for row in predictions[engine].values()),
            'missing': len(ids-set(predictions[engine]))} for engine in ENGINES},
        'models': config['models'], 'input_sha256': digest(dataset/'inference-inputs.jsonl'),
        'source_manifest_sha256': digest(dataset/'manifest.jsonl'),
        'training_examples_created': 0, 'gold_labels_created': 0, 'gpu_execution_validated_locally': False,
        'normalization_for_agreement': 'NFC and whitespace only; raw output retained separately',
        'claim_boundary': 'Training-source proposal generation only. Agreement is not truth; no benchmark, model training, automatic labels or SOTA claim.'}
    write_json(output/'report.json', report)
    return report


def package(work):
    work = Path(work)
    archive = work/'recognizer-data-v3-teacher-evidence.zip'
    files = [path for path in work.rglob('*') if path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.csv')
             and path.name != 'checksums.json']
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            stream.write(path, path.relative_to(work).as_posix())
        stream.writestr('checksums.json', json.dumps({path.relative_to(work).as_posix(): digest(path) for path in files}, indent=2))
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--engine', choices=ENGINES, required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    print(json.dumps(worker(args.engine, config, args.inputs, args.output)), flush=True)
