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
import gc
import importlib
import io
import json
import os
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
# image torchao 0.10 crashes new peft's LoRA dispatcher availability check
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)
subprocess.run([sys.executable, '-c',
                'import importlib.metadata as m, torch; '
                'print("versions: bnb", m.version("bitsandbytes"), '
                '"| peft", m.version("peft"), "| torch", torch.__version__)'])

os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (AutoProcessor, BitsAndBytesConfig,  # noqa: E402
                          Qwen2_5_VLForConditionalGeneration)

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


def _load_4bit():
    """Runs in its own frame so a failure frees its half-loaded model at once."""
    import bitsandbytes
    print('bitsandbytes', bitsandbytes.__version__, bitsandbytes.__file__)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, device_map='cuda', torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=['visual']))  # vision tower must stay floating point
    model = prepare_model_for_kbit_training(model)
    model.visual.to(torch.float16)  # prepare casts it to fp32; fp16 is enough
    return model


def load_backbone():
    """bitsandbytes can be present-but-broken on Kaggle images: probe for real."""
    importlib.invalidate_caches()
    try:
        return MODEL, '4bit-lora', _load_4bit()
    except torch.cuda.OutOfMemoryError:
        raise  # a full GPU needs a session restart, not a different model
    except Exception as error:  # noqa: BLE001 - quantization is optional
        print('4-bit path unusable, falling back to fp16 3B:', repr(error)[:300])
        try:
            import inspect
            from transformers.utils import import_utils
            print('is_bitsandbytes_available source:\n',
                  inspect.getsource(import_utils.is_bitsandbytes_available)[:600])
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
    gc.collect()
    torch.cuda.empty_cache()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        FALLBACK_MODEL, torch_dtype=torch.float16, device_map='cuda')
    return FALLBACK_MODEL, 'fp16-lora', model


if 'model' in globals():
    del model  # stale model from an earlier attempt keeps the GPU full
import sys
sys.last_traceback = sys.last_exc = None  # Jupyter keeps dead frames (and their VRAM)
gc.collect()
torch.cuda.empty_cache()

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
            'labels': labels[0],
            'pixel_values': inputs['pixel_values'],  # (patches, dim) - no batch axis!
            'image_grid_thw': inputs['image_grid_thw'][0]}


class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = [tokenize_example(item) for item in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    """Pad sequences; Qwen-VL pixel_values concatenate across the batch."""
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    longest = max(row['input_ids'].shape[0] for row in batch)
    input_ids, attention, labels = [], [], []
    for row in batch:
        pad = longest - row['input_ids'].shape[0]
        input_ids.append(torch.cat([row['input_ids'],
                                    torch.full((pad,), pad_id, dtype=row['input_ids'].dtype)]))
        attention.append(torch.cat([row['attention_mask'],
                                    torch.zeros(pad, dtype=row['attention_mask'].dtype)]))
        labels.append(torch.cat([row['labels'],
                                 torch.full((pad,), -100, dtype=row['labels'].dtype)]))
    return {'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention),
            'labels': torch.stack(labels),
            'pixel_values': torch.cat([row['pixel_values'] for row in batch], dim=0),
            'image_grid_thw': torch.stack([row['image_grid_thw'] for row in batch])}


def train(model, dataset, steps=STEPS, learning_rate=1e-4, accumulation=8):
    """Plain loop: Trainer/DataParallel replicates the whole model per device."""
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=learning_rate)
    use_fp16 = not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    losses, cursor = [], 0
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        for _ in range(accumulation):
            if cursor >= len(order):
                order = torch.randperm(len(dataset), generator=generator).tolist()
                cursor = 0
            batch = {key: value.to('cuda')
                     for key, value in collate([dataset[order[cursor]]]).items()}
            cursor += 1
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_fp16):
                loss = model(**batch).loss / accumulation
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(round(float(loss.detach()) * accumulation, 4))
        if step % 10 == 0:
            print(f'step {step}/{steps} loss {losses[-1]:.4f} '
                  f'| gpu {torch.cuda.memory_allocated() / 1e9:.1f} GB', flush=True)
    return losses


started = time.perf_counter()
losses = train(model, ExampleDataset(train_examples))
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
    'loss_first': losses[0], 'loss_final': losses[-1], 'loss_min': min(losses),
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
