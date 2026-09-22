"""Kaggle one-cell script: constrained-track baseline (LoRA on organizer data).

Track ``constrained``: open-weight model, training exclusively on PolOCRBench
organizer data (``training.generate_documents``), no external corpora. The
fine-tune uses the frozen organizer prompt (``zero_shot_prompt_v1``) so the
trained system stays prompt-compatible with the zero-shot/API track.

Self-contained on a Kaggle T4 notebook (Internet on):

1. clone this repository and install ``peft`` (transformers/bitsandbytes ship
   with the Kaggle image);
2. generate organizer pages (pinned SEED/COUNT) plus a held-out evaluation
   slice (SEED+1, different degradations) — the evaluation slice stands in for
   test A until the annotated set lands, and is scored as such;
3. LoRA fine-tune Qwen2.5-VL on (prompt -> payload) pairs covering all three
   subtasks A/B/C at once (one adapter, instruction-following);
4. predict the evaluation slice and score it with the repository evaluators;
5. zip the adapter, predictions and scores for download.

This is a baseline recipe (small step budget), not a tuned result: adjust
MODEL/COUNT/STEPS for stronger runs.
"""
import importlib
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = 'https://github.com/PiotrStyla/OCR_engine.git'
MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
FALLBACK_MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'  # fp16 when bitsandbytes is unusable
SEED = 20260922
COUNT = 300
EVAL_COUNT = 40
TRAIN_DEGRADATIONS = 'clean,scan,print_scan,compress'
EVAL_DEGRADATIONS = 'photo,compress'
STEPS = 400
WORKDIR = Path('/kaggle/working/polocrbench-constrained')

# --- 1. environment ----------------------------------------------------------

if not (WORKDIR / 'repo' / 'training').exists():
    subprocess.run(['git', 'clone', '--depth', '1', REPO, str(WORKDIR / 'repo')], check=True)
sys.path.insert(0, str(WORKDIR / 'repo'))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U',
                'peft', 'bitsandbytes>=0.46.1'], check=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (AutoProcessor, BitsAndBytesConfig,  # noqa: E402
                          Qwen2_5_VLForConditionalGeneration, Trainer,
                          TrainingArguments)

from training.generate_documents import generate  # noqa: E402
from training.kie_eval import evaluate as kie_evaluate  # noqa: E402
from training.run_vision_baseline import (PAYLOAD_KEYS, extract_payload,  # noqa: E402
                                          load_templates, render_prompt)
from training.table_eval import evaluate as table_evaluate  # noqa: E402
from training.transcription_eval import evaluate as transcription_evaluate  # noqa: E402

# --- 2. organizer data (deterministic) --------------------------------------

train_dir = WORKDIR / 'train'
eval_dir = WORKDIR / 'eval'
if not (train_dir / 'manifest-A.jsonl').exists():
    generate(train_dir, count=COUNT, seed=SEED, split='constrained-train',
             degradations=tuple(TRAIN_DEGRADATIONS.split(',')))
if not (eval_dir / 'manifest-A.jsonl').exists():
    generate(eval_dir, count=EVAL_COUNT, seed=SEED + 1, split='constrained-eval',
             degradations=tuple(EVAL_DEGRADATIONS.split(',')))
templates = load_templates(WORKDIR / 'repo' / 'benchmarks' / 'polocrbench'
                           / 'prompts' / 'zero_shot_prompt_v1.md')


def examples(data_dir):
    """(subtask, manifest row, image path, prompt, target) for every record."""
    rows = []
    for subtask in 'ABC':
        manifest = data_dir / f'manifest-{subtask}.jsonl'
        for line in manifest.read_text(encoding='utf-8').splitlines():
            if not line:
                continue
            row = json.loads(line)
            payload = row[PAYLOAD_KEYS[subtask]]
            if subtask == 'C':
                target = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'))
            else:
                target = payload
            prompt = render_prompt(subtask, templates[subtask], row)
            rows.append((subtask, row, data_dir / row['image'], prompt, target))
    return rows


train_examples = examples(train_dir)
eval_examples = examples(eval_dir)
print('train examples:', len(train_examples), '| eval examples:', len(eval_examples))

# --- 3. model and LoRA (4-bit when bitsandbytes works, else fp16 3B) ---------


def load_backbone():
    """bitsandbytes can be present-but-broken on Kaggle images: probe for real."""
    importlib.invalidate_caches()
    try:
        import bitsandbytes  # noqa: F401
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, device_map='cuda',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                   bnb_4bit_compute_dtype=torch.bfloat16))
        return MODEL, '4bit-lora', prepare_model_for_kbit_training(model)
    except Exception as error:  # noqa: BLE001 - quantization is optional
        print('4-bit path unusable, falling back to fp16 3B:', repr(error)[:300])
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            FALLBACK_MODEL, torch_dtype=torch.float16, device_map='cuda')
        return FALLBACK_MODEL, 'fp16-lora', model


MODEL, QUANTIZATION, model = load_backbone()
print('backbone:', MODEL, '|', QUANTIZATION)
processor = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28,
                                          max_pixels=1280 * 28 * 28)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type='CAUSAL_LM',
    target_modules='all-linear'))
model.print_trainable_parameters()


def tokenize_example(item):
    _, _, image_path, prompt, target = item
    image = Image.open(image_path).convert('RGB')
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': f'file://{image_path}'}, {'type': 'text', 'text': prompt}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors='pt')
    answer = processor(text=[target], return_tensors='pt')
    input_ids = torch.cat([inputs['input_ids'], answer['input_ids']], dim=1)
    labels = input_ids.clone()
    labels[:, :inputs['input_ids'].shape[1]] = -100  # loss on the payload only
    return {'input_ids': input_ids[0], 'attention_mask':
            torch.cat([inputs['attention_mask'], answer['attention_mask']], dim=1)[0],
            'labels': labels[0], 'pixel_values': inputs['pixel_values'][0],
            'image_grid_thw': inputs['image_grid_thw'][0]}


class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = [tokenize_example(item) for item in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    return {key: torch.stack([row[key] for row in batch]) for key in batch[0]}


started = time.perf_counter()
Trainer(model=model,
        args=TrainingArguments(
            output_dir=str(WORKDIR / 'checkpoints'), max_steps=STEPS,
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=1e-4, lr_scheduler_type='cosine', warmup_ratio=0.03,
            logging_steps=10, save_strategy='no', report_to=[],
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported()),
        train_dataset=ExampleDataset(train_examples),
        data_collator=collate).train()
training_seconds = time.perf_counter() - started
model.save_pretrained(WORKDIR / 'adapter')

# --- 4. predict and score the held-out slice ---------------------------------


def ask(item):
    subtask, row, image_path, prompt, _ = item
    image = Image.open(image_path).convert('RGB')
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': f'file://{image_path}'}, {'type': 'text', 'text': prompt}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors='pt').to('cuda')
    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    elapsed = time.perf_counter() - start
    decoded = processor.batch_decode(output[:, inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True)[0]
    slot = int(row.get('table_index', 0) or 0)
    return extract_payload(subtask, decoded, slot=slot), elapsed


run_dir = WORKDIR / 'runs'
predictions = {'A': {}, 'B': {}, 'C': {}}
elapsed_total, completed, errors = 0.0, 0, 0
for item in eval_examples:
    subtask, row, _, _, _ = item
    try:
        payload, elapsed = ask(item)
        predictions[subtask][row['id']] = {'id': row['id'], 'status': 'ok',
                                           PAYLOAD_KEYS[subtask]: payload,
                                           'elapsed_seconds': round(elapsed, 3)}
        elapsed_total += elapsed
        completed += 1
    except Exception as error:  # noqa: BLE001 - every example attempted
        predictions[subtask][row['id']] = {'id': row['id'], 'status': 'error',
                                           PAYLOAD_KEYS[subtask]: {} if subtask == 'C' else '',
                                           'error_type': type(error).__name__}
        errors += 1
run_dir.mkdir(parents=True, exist_ok=True)
for subtask in 'ABC':
    rows = list(predictions[subtask].values())
    (run_dir / f'predictions-{subtask}.jsonl').write_text(
        '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n',
        encoding='utf-8', newline='\n')

scores = {}
for subtask, evaluate in (('A', transcription_evaluate), ('B', table_evaluate),
                          ('C', kie_evaluate)):
    manifest = eval_dir / f'manifest-{subtask}.jsonl'
    score = evaluate(manifest, run_dir / f'predictions-{subtask}.jsonl')
    (run_dir / f'score-{subtask}.json').write_text(
        json.dumps(score, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    scores[subtask] = score
from training.composite_score import composite  # noqa: E402
summary = composite(scores)
(run_dir / 'composite.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                        encoding='utf-8', newline='\n')
(run_dir / 'run.json').write_text(json.dumps({
    'track': 'constrained', 'model': MODEL, 'quantization': QUANTIZATION,
    'train_examples': len(train_examples),
    'eval_examples': len(eval_examples), 'steps': STEPS,
    'train_degradations': TRAIN_DEGRADATIONS, 'eval_degradations': EVAL_DEGRADATIONS,
    'seed': SEED, 'training_seconds': round(training_seconds, 1),
    'eval_completed': completed, 'eval_errors': errors,
    'mean_seconds_per_example': round(elapsed_total / max(1, completed), 3),
    'device': torch.cuda.get_device_name(0)}, indent=2), encoding='utf-8', newline='\n')
print(json.dumps({'scores': summary['scores'], 'composite': summary['composite']}, indent=2))

# --- 5. zip -----------------------------------------------------------------

archive = Path('/kaggle/working/polocrbench-constrained.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(WORKDIR.rglob('*')):
        if path.is_file() and (path.suffix in ('.json', '.jsonl', '.safetensors')
                               or path.name == 'adapter_config.json'):
            bundle.write(path, path.relative_to(WORKDIR))
print('archive:', archive)
