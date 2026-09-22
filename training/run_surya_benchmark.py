"""Surya 2 baseline for PolOCRBench subtasks A (text) and B (table HTML).

Requires ``pip install surya-ocr`` plus an inference backend (vllm on NVIDIA or
llama.cpp's ``llama-server`` on CPU; ``SuryaInferenceManager`` auto-spawns one).
Subtask C is out of scope for Surya (it does not extract key information) — use
``training.run_vision_baseline`` or the Qwen-VL Kaggle script for KIE.

One Surya call set per page image:

- B: ``TableRecPredictor.predict_full`` tables are aligned to the manifest's
  table slots in reading order (top-left of table crops); surplus tables are
  counted in ``run.json`` and deficits become empty payloads;
- A: ``RecognitionPredictor`` blocks in reading order; Table blocks serialize
  as rows of cell texts (the synthetic ground-truth convention), other blocks
  are tag-stripped and whitespace-normalized.

Cost reporting is per page (Surya recognizes a page once even when it fills
several table slots; ``elapsed_seconds`` lands on the first slot row only).
Dry run by default (standard library only).

Usage:
  python -m training.run_surya_benchmark --subtask B --manifest manifest-B.jsonl \
      --output runs/surya-b --execute
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from training.run_vision_baseline import PROTOCOL_VERSION, PAYLOAD_KEYS
from training.stage_impact_benchmark import digest
from training.validate_submission import load_jsonl, unique_ids

_SUBTASKS = ('A', 'B')
_TAGS = re.compile(r'<[^>]+>')


def _get(item, name, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def load_page_cases(manifest):
    """Group manifest rows by page image; B rows become ordered table slots."""
    manifest = Path(manifest)
    rows = load_jsonl(manifest)
    if not rows:
        raise ValueError('Empty manifest')
    unique_ids(rows, 'Manifest')
    pages, order = {}, []
    for row in rows:
        path = (manifest.parent / row['image']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Image checksum mismatch: {row["id"]}')
        key = str(path)
        if key not in pages:
            pages[key] = {'path': path, 'rows': []}
            order.append(pages[key])
        pages[key]['rows'].append(row)
    for page in order:
        page['rows'].sort(key=lambda row: row.get('table_index', 0))
    return order


def align_tables(tables, count):
    """Reading-order HTML list padded/trimmed to the manifest slot count."""
    ordered = sorted(tables, key=lambda table: (_get(table, 'image_bbox') or [0, 0])[1::-1])
    htmls = [(_get(table, 'html') or '') for table in ordered]
    surplus = max(0, len(htmls) - count)
    return (htmls[:count] + [''] * max(0, count - len(htmls))), surplus


def blocks_to_text(blocks):
    """Subtask A text: table rows as cell lines, other blocks tag-stripped."""
    from training.table_eval import row_texts
    from training.transcription_eval import normalize
    lines = []
    for block in blocks:
        if _get(block, 'skipped', False) or _get(block, 'error', False):
            continue
        html = _get(block, 'html') or ''
        if _get(block, 'label') == 'Table':
            lines.extend(row_texts(html))
        elif html.strip():
            lines.append(normalize(_TAGS.sub(' ', html)))
    return '\n'.join(lines)


def create_predictors():
    from surya.inference import SuryaInferenceManager
    from surya.recognition import RecognitionPredictor
    from surya.table_rec import TableRecPredictor
    manager = SuryaInferenceManager()
    return RecognitionPredictor(manager), TableRecPredictor(manager)


def predict_page(recognition, table_rec, subtask, path):
    """One Surya pass per page: (payload, elapsed). Payload is text or [html, ...]."""
    from PIL import Image
    image = Image.open(path)
    start = time.perf_counter()
    if subtask == 'B':
        results = table_rec.predict_full([image])
        tables = results[0] if results else []
        payload = [_get(table, 'html') or '' for table in tables]
    else:
        pages = recognition([image])
        payload = blocks_to_text(_get(pages[0], 'blocks') or [])
    return payload, time.perf_counter() - start


def run_pages(subtask, pages, predict, output, metadata):
    """One case per page; B fills every manifest table slot of that page."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload_key = PAYLOAD_KEYS[subtask]
    run = {**metadata, 'started_utc': datetime.now(timezone.utc).isoformat(),
           'pages_expected': len(pages), 'pages_completed': 0, 'error_pages': 0,
           'surplus_tables': 0, 'state': 'running'}

    def save_state():
        temporary = output / 'run.json.tmp'
        temporary.write_text(json.dumps(run, indent=2), encoding='utf-8', newline='\n')
        temporary.replace(output / 'run.json')

    save_state()
    elapsed_total, elapsed_max, timed = 0.0, 0.0, 0
    try:
        with (output / 'predictions.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for page in pages:
                slots = page['rows']
                try:
                    payload, elapsed = predict(page['path'])
                    timed += 1
                    elapsed_total += elapsed
                    elapsed_max = max(elapsed_max, elapsed)
                    run['pages_completed'] += 1
                    if subtask == 'B':
                        htmls, surplus = align_tables(payload, len(slots))
                        run['surplus_tables'] += surplus
                        rows = [{'id': slot['id'], 'status': 'ok', 'html': html,
                                 **({'elapsed_seconds': round(elapsed, 3)} if index == 0 else {})}
                                for index, (slot, html) in enumerate(zip(slots, htmls))]
                    else:
                        rows = [{'id': slots[0]['id'], 'status': 'ok', 'text': payload,
                                 'elapsed_seconds': round(elapsed, 3)}]
                except Exception as error:  # noqa: BLE001 - each page isolated
                    empty = {} if subtask == 'C' else ''
                    rows = [{'id': slot['id'], 'status': 'error', payload_key: empty,
                             'error_type': type(error).__name__} for slot in slots]
                    run['error_pages'] += 1
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
        run['state'] = 'completed'
    except BaseException:
        run['state'] = 'interrupted'
        raise
    finally:
        run['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if timed:
            run['cost'] = {'pages_timed': timed,
                           'mean_elapsed_seconds': round(elapsed_total / timed, 3),
                           'max_elapsed_seconds': round(elapsed_max, 3),
                           'total_elapsed_seconds': round(elapsed_total, 3)}
        save_state()
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subtask', choices=_SUBTASKS, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    pages = load_page_cases(args.manifest)
    plan = {'protocol_version': PROTOCOL_VERSION, 'engine': 'surya2',
            'manifest_sha256': digest(args.manifest), 'subtask': args.subtask,
            'pages': len(pages),
            'table_slots': sum(len(page['rows']) for page in pages) if args.subtask == 'B' else None}
    if not args.execute:
        print(json.dumps({key: value for key, value in
                          {**plan, 'state': 'preflight-only'}.items() if value is not None},
                         indent=2))
        return
    recognition, table_rec = create_predictors()

    def predict(path):
        return predict_page(recognition, table_rec, args.subtask, path)

    run = run_pages(args.subtask, pages, predict, args.output, plan)
    print(json.dumps({key: run[key] for key in
                      ('state', 'pages_completed', 'error_pages', 'surplus_tables', 'cost')
                      if key in run}, indent=2))


if __name__ == '__main__':
    main()
