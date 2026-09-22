"""Kaggle one-cell script: Qwen2.5-VL zero-shot baseline for PolOCRBench A/B/C.

Run on a Kaggle T4 GPU notebook with Internet access, as one cell (same pattern
as training/kaggle_ehri_control.py). Steps:

1. clone this repository at main and pip-install the scoring extras (jiwer);
2. generate the synthetic PolOCRBench pages deterministically (same seed as the
   local run reproduces the same images and manifests);
3. run Qwen2.5-VL-7B-Instruct (4-bit) zero-shot with the frozen organizer
   prompt (training.run_vision_baseline.load_templates, hash-pinned);
4. write predictions A/B/C + run.json with per-page latency, score them with
   training.transcription_eval / table_eval / kie_eval and composite_score;
5. zip everything for download.

No model training; the run belongs to the zero-shot/API track (open-weight
model, no fine-tuning, organizer prompt). Adjust COUNT/SEED/MODEL below to
reproduce or extend the baseline.
"""
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = 'https://github.com/PiotrStyla/OCR_engine.git'
MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
SEED = 20260922
COUNT = 8
WORKDIR = Path('/kaggle/working/polocrbench-qwen-bc')

# --- 1. environment ----------------------------------------------------------

if not (WORKDIR / 'repo' / 'training').exists():
    subprocess.run(['git', 'clone', '--depth', '1', REPO, str(WORKDIR / 'repo')], check=True)
sys.path.insert(0, str(WORKDIR / 'repo'))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U',
                'jiwer', 'bitsandbytes>=0.46.1'], check=True)
# image torchao 0.10 crashes new peft's availability checks
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import (AutoProcessor, BitsAndBytesConfig,  # noqa: E402
                          Qwen2_5_VLForConditionalGeneration)

from training.generate_documents import generate  # noqa: E402
from training.run_vision_baseline import PAYLOAD_KEYS, extract_payload, render_prompt  # noqa: E402
from training.run_vision_baseline import load_templates  # noqa: E402

# --- 2. deterministic synthetic pages ---------------------------------------

data = WORKDIR / 'data'
if not (data / 'manifest-A.jsonl').exists():
    report = generate(data, count=COUNT, seed=SEED, split='baseline-smoke')
else:
    report = json.loads((data / 'generation.json').read_text(encoding='utf-8'))
print(json.dumps({key: report[key] for key in ('seed', 'count', 'types', 'degradations')}))

# --- 3. model and frozen prompt ---------------------------------------------

if 'model' in globals():
    del model  # stale model from an earlier attempt keeps the GPU full
import gc
gc.collect()
torch.cuda.empty_cache()

try:
    import bitsandbytes  # noqa: F401
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, device_map='cuda', torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=['visual']))  # vision tower must stay floating point
except Exception as error:  # noqa: BLE001 - quantization is optional
    print('4-bit path unusable, falling back to fp16 3B:', repr(error)[:300])
    MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map='cuda')
processor = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28,
                                          max_pixels=1280 * 28 * 28)
templates = load_templates(data.parent / 'repo' / 'benchmarks' / 'polocrbench'
                           / 'prompts' / 'zero_shot_prompt_v1.md')


def ask(subtask, row, image_path):
    prompt = render_prompt(subtask, templates[subtask], row)
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': f'file://{image_path}'},
        {'type': 'text', 'text': prompt}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    image = Image.open(image_path).convert('RGB')
    inputs = processor(text=[rendered], images=[image], return_tensors='pt').to('cuda')
    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    elapsed = time.perf_counter() - start
    decoded = processor.batch_decode(output[:, inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True)[0]
    return extract_payload(subtask, decoded, slot=int(row.get('table_index', 0) or 0)), elapsed


# --- 4. predictions A/B/C ---------------------------------------------------

for subtask in 'ABC':
    manifest = data / f'manifest-{subtask}.jsonl'
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    run_dir = WORKDIR / 'runs' / f'qwen-{subtask}'
    run_dir.mkdir(parents=True, exist_ok=True)
    payload_key = PAYLOAD_KEYS[subtask]
    elapsed_total, completed, errors = 0.0, 0, 0
    with (run_dir / 'predictions.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            image_path = data / row['image']
            try:
                payload, elapsed = ask(subtask, row, image_path)
                record = {'id': row['id'], 'status': 'ok', payload_key: payload,
                          'elapsed_seconds': round(elapsed, 3)}
                elapsed_total += elapsed
                completed += 1
            except Exception as error:  # noqa: BLE001 - every page attempted
                record = {'id': row['id'], 'status': 'error',
                          payload_key: {} if subtask == 'C' else '',
                          'error_type': type(error).__name__}
                errors += 1
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    cost = {'pages_timed': completed,
            'mean_elapsed_seconds': round(elapsed_total / max(1, completed), 3),
            'total_elapsed_seconds': round(elapsed_total, 3)}
    (run_dir / 'run.json').write_text(json.dumps({
        'engine': 'qwen2.5-vl-7b-instruct-4bit', 'model': MODEL, 'subtask': subtask,
        'pages_expected': len(rows), 'pages_completed': completed,
        'error_pages': errors, 'state': 'completed', 'cost': cost,
        'device': torch.cuda.get_device_name(0)}, indent=2), encoding='utf-8', newline='\n')
    print(subtask, completed, '/', len(rows), cost)

# --- 5. score and zip -------------------------------------------------------

from training.composite_score import composite  # noqa: E402
from training.kie_eval import evaluate as kie_evaluate  # noqa: E402
from training.table_eval import evaluate as table_evaluate  # noqa: E402
from training.transcription_eval import evaluate as transcription_evaluate  # noqa: E402

reports = {}
for subtask, evaluate in (('A', transcription_evaluate), ('B', table_evaluate),
                          ('C', kie_evaluate)):
    score = evaluate(data / f'manifest-{subtask}.jsonl',
                     WORKDIR / 'runs' / f'qwen-{subtask}' / 'predictions.jsonl')
    (WORKDIR / 'runs' / f'score-{subtask}.json').write_text(
        json.dumps(score, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    reports[subtask] = score
summary = composite(reports)
(WORKDIR / 'runs' / 'composite.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
print(json.dumps({key: summary[key] for key in ('scores', 'composite')}, indent=2))

archive = Path('/kaggle/working/polocrbench-qwen-bc.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(WORKDIR.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.jsonl'):
            bundle.write(path, path.relative_to(WORKDIR))
print('archive:', archive)
