"""Private SLAYER-OCR layout-teacher pilot for one pinned teacher per run.

The module keeps heavy imports inside adapter functions so its selection,
mapping and parsing contracts can be tested on CPU without model dependencies.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


PROPOSAL_SCHEMA = 'slayer-layout-teacher-proposal-v1'
PROMPT = """Detect every page-layout region in this historical printed document.
Return only a JSON array. Each element must be:
{"label": one of ["text_region", "heading", "table", "figure", "caption", "marginalia", "header", "footer", "page_number"], "bbox_2d": [x1, y1, x2, y2]}.
Coordinates must be normalized integers from 0 to 1000 relative to the displayed image.
Use tight, mutually exclusive region boxes. Do not nest text_region inside another region.
Do not transcribe, modernize or correct any text. Do not add explanations or Markdown."""

LABELS = {
    'text': 'text_region', 'plain_text': 'text_region', 'content': 'text_region',
    'paragraph': 'text_region', 'abstract': 'text_region', 'reference': 'text_region',
    'reference_content': 'text_region', 'list': 'text_region', 'list_item': 'text_region',
    'algorithm': 'text_region', 'formula': 'text_region', 'display_formula': 'text_region',
    'inline_formula': 'text_region', 'title': 'heading', 'doc_title': 'heading',
    'document_title': 'heading', 'paragraph_title': 'heading', 'section_header': 'heading',
    'section_title': 'heading', 'table': 'table', 'picture': 'figure', 'image': 'figure',
    'figure': 'figure', 'chart': 'figure', 'caption': 'caption',
    'figure_caption': 'caption', 'table_caption': 'caption', 'figure_title': 'caption',
    'table_title': 'caption', 'footnote': 'marginalia', 'aside_text': 'marginalia',
    'marginal_note': 'marginalia', 'marginalia': 'marginalia', 'header': 'header',
    'page_header': 'header', 'footer': 'footer', 'page_footer': 'footer',
    'page_number': 'page_number', 'number': 'page_number',
}
SCORE_KINDS = {'model-confidence', 'neutral-unavailable'}

DOCLAYOUT_NAMES = {
    0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure',
    4: 'figure_caption', 5: 'table', 6: 'table_caption',
    7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption',
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def rank(value, salt):
    return sha256_bytes((salt + '\0' + value).encode('utf-8'))


def select_pages(rows, count, salt):
    if not 1 <= count <= len(rows):
        raise ValueError('Invalid page count')
    if len({row.get('id') for row in rows}) != len(rows):
        raise ValueError('Duplicate page ID')
    groups = defaultdict(list)
    for row in rows:
        if row.get('split') != 'train' or row.get('license') != 'CC-BY-3.0':
            raise ValueError('Pilot accepts only CC-BY-3.0 train pages')
        groups[row['collection']].append(row)
    for collection in groups:
        groups[collection].sort(key=lambda row: rank(row['id'], salt))
    order = sorted(groups, key=lambda collection: rank(collection, salt))
    selected = []
    while len(selected) < count:
        added = False
        for collection in order:
            if groups[collection] and len(selected) < count:
                selected.append(groups[collection].pop(0))
                added = True
        if not added:
            raise ValueError('Insufficient pages')
    return selected


def normalize_label(value):
    if not isinstance(value, str):
        return None
    key = re.sub(r'_+', '_', re.sub(r'[^a-z0-9]+', '_', value.casefold())).strip('_')
    return LABELS.get(key)


def _json_payload(text):
    fenced = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find('['), candidate.rfind(']')
        if start < 0 or end <= start:
            raise ValueError('Qwen output has no JSON array')
        data = json.loads(candidate[start:end + 1])
    if isinstance(data, dict):
        data = data.get('detections') or data.get('objects') or data.get('results')
    if not isinstance(data, list):
        raise ValueError('Qwen output is not a detection list')
    return data


def parse_qwen_grounding(text, width, height):
    detections = []
    for index, item in enumerate(_json_payload(text)):
        if not isinstance(item, dict):
            raise ValueError(f'Invalid Qwen detection at index {index}')
        box = item.get('bbox_2d') or item.get('bbox')
        if (not isinstance(box, list) or len(box) != 4 or
                any(not isinstance(value, (int, float)) for value in box)):
            raise ValueError(f'Invalid Qwen bbox at index {index}')
        values = [float(value) for value in box]
        if not (0 <= values[0] < values[2] <= 1000 and 0 <= values[1] < values[3] <= 1000):
            raise ValueError(f'Qwen bbox outside normalized range at index {index}')
        detections.append({
            'raw_label': item.get('label'),
            'bbox_xyxy': [values[0] * width / 1000, values[1] * height / 1000,
                          values[2] * width / 1000, values[3] * height / 1000],
            'score': 0.5,
            'score_kind': 'neutral-unavailable',
        })
    return detections


def _valid_box(box, width, height):
    return (isinstance(box, (list, tuple)) and len(box) == 4 and
            all(isinstance(value, (int, float)) for value in box) and
            0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def canonicalize(raw, width, height):
    accepted, unmapped = [], []
    for index, item in enumerate(raw):
        label = normalize_label(item.get('raw_label'))
        box = item.get('bbox_xyxy')
        reason = None
        if label is None:
            reason = 'unmapped-label'
        elif not _valid_box(box, width, height):
            reason = 'invalid-bbox'
        score = item.get('score')
        if reason is None and (not isinstance(score, (int, float)) or not 0 <= score <= 1):
            reason = 'invalid-score'
        score_kind = item.get('score_kind', 'model-confidence')
        if reason is None and score_kind not in SCORE_KINDS:
            reason = 'invalid-score-kind'
        if reason:
            unmapped.append({'index': index, 'reason': reason, 'raw': item})
            continue
        accepted.append({
            'label': label,
            'bbox_xyxy': [float(value) for value in box],
            'score': float(score),
            'score_kind': score_kind,
            'raw_label': item.get('raw_label'),
        })
    return accepted, unmapped


def make_proposal(page, teacher_id, teacher, run_id, prompt_sha256, detections,
                  *, status='ok', error=None):
    return {
        'schema': PROPOSAL_SCHEMA,
        'page_id': page['id'],
        'image': {'file_name': page['file_name'], 'sha256': page['image_sha256'],
                  'width': page['width'], 'height': page['height']},
        'teacher': {'id': teacher_id, 'revision': teacher['revision'],
                    'run_id': run_id, 'prompt_sha256': prompt_sha256},
        'detections': [dict({'id': f'{page["id"]}:{teacher_id}:{index:04d}'}, **item)
                       for index, item in enumerate(detections)],
        'status': status,
        'error': error,
    }


def _safe_dataset_path(row, split):
    relative = PurePosixPath(row['file_name'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe dataset image path')
    return (PurePosixPath('pages') / split / relative).as_posix()


def download_inputs(config, work, count):
    from huggingface_hub import hf_hub_download

    dataset = config['dataset']
    metadata = Path(hf_hub_download(
        dataset['repo'], dataset['metadata_path'], repo_type='dataset',
        revision=dataset['revision']))
    if sha256_file(metadata) != dataset['metadata_sha256']:
        raise ValueError('Dataset metadata checksum mismatch')
    rows = [json.loads(line) for line in metadata.read_text(encoding='utf-8').splitlines()
            if line.strip()]
    if len(rows) != dataset['available_pages']:
        raise ValueError('Dataset page count changed')
    selected = select_pages(rows, count, dataset['rank_salt'])
    image_dir = work / 'private-images'
    image_dir.mkdir()
    pages = []
    for position, row in enumerate(selected, 1):
        source = Path(hf_hub_download(
            dataset['repo'], _safe_dataset_path(row, 'train'), repo_type='dataset',
            revision=dataset['revision']))
        if sha256_file(source) != row['image_sha256']:
            raise ValueError(f'Image checksum mismatch: {row["id"]}')
        suffix = Path(row['file_name']).suffix or '.jpg'
        target = image_dir / f'{position:04d}{suffix}'
        shutil.copyfile(source, target)
        pages.append({key: row[key] for key in (
            'id', 'collection', 'width', 'height', 'image_sha256', 'file_name', 'license')} |
            {'local_path': str(target)})
        print(f'Input {position}/{len(selected)}: {row["id"]}', flush=True)
    public_selection = [{key: page[key] for key in page if key != 'local_path'} for page in pages]
    (work / 'selection.json').write_text(
        json.dumps(public_selection, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return pages


def _load_qwen(spec):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForImageTextToText.from_pretrained(
        spec['model_repo'], revision=spec['revision'], device_map='auto',
        torch_dtype=torch.float16, quantization_config=quantization)
    processor = AutoProcessor.from_pretrained(
        spec['model_repo'], revision=spec['revision'],
        min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)

    def infer(path):
        from PIL import Image
        image = Image.open(path).convert('RGB')
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to(model.device)
        output = model.generate(**inputs, max_new_tokens=spec['max_new_tokens'], do_sample=False)
        text = processor.batch_decode(
            output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        return parse_qwen_grounding(text, image.width, image.height), text

    return model, infer


def _load_doclayout(spec, model_dir):
    from doclayout_yolo import YOLOv10
    checkpoint = model_dir / 'doclayout_yolo_docstructbench_imgsz1024.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = YOLOv10(str(checkpoint))

    def infer(path):
        result = model.predict(
            str(path), imgsz=spec['image_size'], conf=spec['threshold'],
            iou=spec['nms_iou'], device='cuda', verbose=False)[0]
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        classes = result.boxes.cls.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        raw = [{'raw_label': DOCLAYOUT_NAMES.get(int(cls), f'class_{int(cls)}'),
                'bbox_xyxy': box, 'score': score, 'score_kind': 'model-confidence'}
               for box, cls, score in zip(boxes, classes, scores)]
        return raw, None

    return model, infer


def _load_surya(model_dir):
    from surya.fast_layout import FastLayoutPredictor
    model = FastLayoutPredictor(checkpoint=str(model_dir))

    def infer(path):
        from PIL import Image
        prediction = model([Image.open(path).convert('RGB')])[0]
        raw = []
        for box in prediction.bboxes:
            confidence = getattr(box, 'confidence', None)
            raw.append({
                'raw_label': box.label,
                'bbox_xyxy': list(box.bbox),
                'score': float(confidence) if confidence is not None else 0.5,
                'score_kind': ('model-confidence' if confidence is not None
                               else 'neutral-unavailable'),
            })
        return raw, None

    return model, infer


def _load_teacher(teacher_id, spec, work):
    from huggingface_hub import snapshot_download
    if teacher_id == 'qwen3-vl-4b':
        return _load_qwen(spec)
    model_dir = Path(snapshot_download(
        spec['model_repo'], revision=spec['revision'], local_dir=work / 'model'))
    if teacher_id == 'doclayout-yolo':
        return _load_doclayout(spec, model_dir)
    if teacher_id == 'surya-layout2':
        return _load_surya(model_dir)
    raise ValueError(f'Unknown teacher: {teacher_id}')


def _package_versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def run(teacher_id, config, pages=None, output_root='/content'):
    import torch

    if teacher_id not in config['teachers']:
        raise ValueError(f'Unknown teacher: {teacher_id}')
    pages = pages or config['dataset']['selected_pages']
    if not 1 <= pages <= config['dataset']['selected_pages']:
        raise ValueError('pages outside frozen pilot range')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime, then Run all')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    work = Path(output_root) / f'slayer-layout-{teacher_id}-{run_id}'
    work.mkdir()
    spec = config['teachers'][teacher_id]
    selected = download_inputs(config, work, pages)
    model, infer = _load_teacher(teacher_id, spec, work)
    prompt_material = PROMPT if teacher_id == 'qwen3-vl-4b' else json.dumps(
        {'adapter': teacher_id, 'spec': spec, 'label_map': LABELS}, sort_keys=True)
    prompt_sha256 = sha256_bytes(prompt_material.encode('utf-8'))
    proposal_path = work / 'teacher-proposals.jsonl'
    raw_path = work / 'raw-model-output.jsonl'
    unmapped_path = work / 'unmapped.jsonl'
    completed = errors = accepted_count = unmapped_count = 0
    started = time.perf_counter()
    with (proposal_path.open('x', encoding='utf-8', newline='\n') as proposals,
          raw_path.open('x', encoding='utf-8', newline='\n') as raw_stream,
          unmapped_path.open('x', encoding='utf-8', newline='\n') as unmapped_stream):
        for position, page in enumerate(selected, 1):
            status, error, raw_text = 'ok', None, None
            try:
                raw, raw_text = infer(page['local_path'])
                detections, rejected = canonicalize(raw, page['width'], page['height'])
                completed += 1
                accepted_count += len(detections)
                unmapped_count += len(rejected)
                for item in rejected:
                    unmapped_stream.write(json.dumps(
                        {'page_id': page['id'], **item}, ensure_ascii=False) + '\n')
            except Exception as exc:  # every page remains represented
                status, error, detections = 'error', f'{type(exc).__name__}: {exc}', []
                errors += 1
            proposals.write(json.dumps(make_proposal(
                page, teacher_id, spec, run_id, prompt_sha256, detections,
                status=status, error=error), ensure_ascii=False) + '\n')
            proposals.flush()
            if raw_text is not None:
                raw_stream.write(json.dumps(
                    {'page_id': page['id'], 'output': raw_text}, ensure_ascii=False) + '\n')
                raw_stream.flush()
            print(f'Teacher {position}/{len(selected)}: {page["id"]} status={status} '
                  f'objects={len(detections)}', flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    report = {
        'schema': 'slayer-layout-teacher-run-v1', 'state': 'completed',
        'teacher_id': teacher_id, 'teacher': spec, 'run_id': run_id,
        'prompt_sha256': prompt_sha256, 'pages_expected': len(selected),
        'pages_completed': completed, 'error_pages': errors,
        'accepted_detections': accepted_count, 'unmapped_detections': unmapped_count,
        'elapsed_seconds': round(time.perf_counter() - started, 3),
        'gpu': torch.cuda.get_device_name(0), 'python': sys.version,
        'packages': _package_versions([
            'torch', 'transformers', 'bitsandbytes', 'accelerate',
            'doclayout-yolo', 'surya-ocr', 'huggingface-hub', 'Pillow']),
        'dataset_revision': config['dataset']['revision'],
        'dataset_metadata_sha256': config['dataset']['metadata_sha256'],
        'reference_text_sent_to_teacher': False,
        'images_or_references_in_evidence_zip': False,
        'automatic_publication': False,
    }
    (work / 'run.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (work / 'experiment-config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    evidence = work / 'evidence'
    evidence.mkdir()
    for path in (proposal_path, raw_path, unmapped_path, work / 'selection.json',
                 work / 'run.json', work / 'experiment-config.json'):
        shutil.copyfile(path, evidence / path.name)
    checksums = {path.name: sha256_file(path) for path in sorted(evidence.iterdir())}
    (evidence / 'checksums.json').write_text(
        json.dumps(checksums, indent=2) + '\n', encoding='utf-8')
    archive = Path(shutil.make_archive(str(work / f'{teacher_id}-evidence'), 'zip', evidence))
    return archive
