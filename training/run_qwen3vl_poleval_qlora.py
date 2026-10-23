"""Pinned Qwen3-VL QLoRA experiment on the official PolOCRBench train split.

Sibling of runpod_qwen3vl_histfinetune (same model pin, hyperparameters and
protocol) with the corpus swapped to the pinned poleval-corpus-v1 built from
slayerlabs/PolEvalDS data/train. Use ``--prepare-only`` to verify all
train/dev images and image-expanded labels on CPU. Use ``--execute``
explicitly for training: one continuous three-epoch schedule, durable
per-epoch adapters and DEV CER/WER, baseline-protected DEV selection, then
one frozen test A measurement. No resource provisioning or automatic retry.
"""
import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_REPOSITORY = "slayerlabs/PolEvalDS data/train"
SOURCE_REVISION = "a503dc8588e17b486045d36f3c21b5a6c98c605a"
CORPUS_MANIFEST_SHA256 = "af14901cce55094826cc3caf7a0d2402c89df8e497cd2fedfef446752df74717"
EXPECTED_TRAIN, EXPECTED_DEV = 1820, 210
IMPACT_REPOSITORY = "PiotrSty/impact-print-v2"
IMPACT_REVISION = "a2480fde6f15284701458ff370b81cce50dc5c2d"
IMPACT_SHA256 = "0a9ffa126029703726fc5883a8279ddccf15763c4fdd483ed9b8cc034bdb42f0"
IMPACT_PATH = "impact-print-v2-test.tar.gz"
MODEL_SOURCE = "PiotrSty/slayer-ocr-qwen3vl-4b-frozen"
MODEL_REVISION = "2f5de2dc7bc70a683d53be1792d60e459a2d2c6d"
TEST_MANIFEST_SHA256 = "b7ed83788ae887565bb2e305d760a869d120fbaa8546949ad4b4dbfe6d0db211"
EPOCHS = 3
LORA_R = 16
LORA_ALPHA = 32
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
MAX_NEW_TOKENS = 4096
MAX_PIXELS = 4194304
MIN_PIXELS = 262144
EVIDENCE_REPOSITORY = "PiotrSty/slayer-ocr-experiment-evidence"
EVIDENCE_PATH = "experiments/2026-10-23/qwen3vl-poleval-qlora-v1"

WORK = Path("/workspace/qwen3vl-poleval-qlora-v1")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_finetune_rows(pages, test_image_hashes):
    """Split the corpus pages into train/dev, excluding the test split.

    Raises when a test-split row or an image shared with test A would enter
    training: the benchmark stays untouched.
    """
    train, dev = [], []
    for row in pages:
        split = row.get("split")
        if split == "test":
            continue
        if split not in {"train", "validation"}:
            raise ValueError(f"Unknown corpus split: {split}")
        if not row.get("image_sha256"):
            raise ValueError(f"Missing image hash: {row['id']}")
        if row["image_sha256"] in test_image_hashes:
            raise ValueError(f"Training page overlaps test A: {row['id']}")
        (train if split == "train" else dev).append(row)
    if not train or not dev:
        raise ValueError("Empty train or dev split")
    ids = [row["id"] for row in train + dev]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate page ids")
    if {row["image_sha256"] for row in train} & {row["image_sha256"] for row in dev}:
        raise ValueError("Image shared between train and dev")
    return train, dev




def efficient_sdpa_attention(module, query, key, value, attention_mask,
                             dropout=0.0, scaling=None, is_causal=None, **kwargs):
    """Expand GQA heads so CUDA can use efficient SDPA; never fall back to math."""
    from contextlib import nullcontext
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.integrations.sdpa_attention import repeat_kv

    groups = module.num_key_value_groups
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, :key.shape[-2]]
    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and module.is_causal
    # Native enable_gqa only supports CUDA Flash/math, not the T4 efficient kernel.
    # CPU is used only for explicit structural verification, never as a GPU fallback.
    backend = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]) if query.is_cuda else nullcontext()
    with backend:
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout,
            scale=scaling, is_causal=is_causal)
    return output.transpose(1, 2).contiguous(), None


def configure_attention(engine):
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, ALL_MASK_ATTENTION_FUNCTIONS

    name = "slayer_sdpa_fused"
    AttentionInterface.register(name, efficient_sdpa_attention)
    # Reuse the SDPA mask builder, including padding and cached decode semantics.
    AttentionMaskInterface.register(name, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    engine.model.language_model.set_attn_implementation(name)
    return engine


def prepare_lora_model(engine):
    """Freeze the base and recompute LM activations instead of retaining them."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    configure_attention(engine)

    engine = prepare_model_for_kbit_training(engine, use_gradient_checkpointing=False)
    # PEFT casts non-quantized weights to fp32; the frozen vision tower stays fp16.
    engine.model.visual.to(torch.float16)
    engine.config.use_cache = False
    engine.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [name for name, _ in engine.named_modules()
               if ".language_model." in name and name.rsplit(".", 1)[-1] in suffixes]
    if not targets:
        raise ValueError("No Qwen3-VL language-model LoRA targets")
    return get_peft_model(engine, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets))


def make_epoch_callback(engine, ledger, evaluate_dev, output):
    from transformers import TrainerCallback

    class EpochCheckpoint(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = round(state.epoch)
            adapter = Path(output) / f"adapter-epoch{epoch}"
            phase = f"epoch-{epoch}-checkpoint"
            try:
                engine.save_pretrained(adapter)
                phase = f"epoch-{epoch}-dev"
                metrics = evaluate_dev(epoch)
                ledger.record(epoch, metrics, adapter)
                print("DEV_EPOCH", epoch, metrics, flush=True)
            except Exception as error:
                ledger.mark_failure(phase, error)
                raise
            finally:
                engine.train()
            return control

    return EpochCheckpoint()


def training_arguments(output, *, smoke=False, use_cpu=False):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(Path(output) / "checkpoints"), num_train_epochs=EPOCHS,
        max_steps=1 if smoke else -1,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        # A one-step warmup starts at LR=0; smoke must exercise an actual update.
        learning_rate=LEARNING_RATE, lr_scheduler_type="cosine", warmup_ratio=0.0 if smoke else 0.05,
        logging_steps=1 if smoke else 5, save_strategy="no", fp16=not use_cpu,
        use_cpu=use_cpu, remove_unused_columns=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], seed=42)


def make_training_trainer(*args, **kwargs):
    """Preserve next-token CE while projecting only supervised hidden states."""
    import torch
    from transformers import Trainer

    class SupervisedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Label at t+1 is predicted by hidden state t; the final state has no target.
            shifted = inputs["labels"][:, 1:]
            positions = shifted.ne(-100).any(dim=0).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                raise ValueError("Training batch has no supervised next-token targets")
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            outputs = model(**model_inputs, logits_to_keep=positions)
            targets = shifted.index_select(1, positions).to(outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.reshape(-1, outputs.logits.shape[-1]).float(),
                targets.reshape(-1), ignore_index=-100)
            outputs.loss = loss
            return (loss, outputs) if return_outputs else loss

    trainer = SupervisedTrainer(*args, **kwargs)
    # CE is a per-microbatch mean, not a sum divided by num_items_in_batch.
    # Trainer must divide it by the current accumulation-window size before backward.
    trainer.model_accepts_loss_kwargs = False
    return trainer


def smoke_train_step(engine, features, output, pad_token_id, *, use_cpu=False):
    """One optimizer step with the real accumulation/AMP path; discard its weights."""
    import math
    import torch
    from transformers import TrainerCallback
    from training.qwen3vl_training_data import collate_pages

    if not use_cpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for full-model smoke")
        torch.cuda.reset_peak_memory_stats()
    observation = {}

    class CheckOptimizer(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                if not torch.isfinite(parameter.grad).all():
                    raise ValueError(f"Nonfinite smoke gradient: {name}")
                if "parameter" not in observation and torch.count_nonzero(parameter.grad).item():
                    observation.update(parameter=parameter, name=name,
                                       norm_before=parameter.detach().float().norm().item())
            if "parameter" not in observation:
                raise ValueError("Smoke has no nonzero adapter gradients")
            return control

    # Repeat the same train page for the four accumulation microbatches, not four steps.
    trainer = make_training_trainer(
        model=engine, args=training_arguments(output, smoke=True, use_cpu=use_cpu),
        train_dataset=[features] * GRADIENT_ACCUMULATION,
        data_collator=lambda batch: collate_pages(batch, pad_token_id),
        callbacks=[CheckOptimizer()])
    result = trainer.train()
    if trainer.state.global_step != 1 or not math.isfinite(result.training_loss):
        raise ValueError("Smoke did not complete one finite optimizer step")
    norm_after = observation["parameter"].detach().float().norm().item()
    if not math.isfinite(norm_after) or norm_after == observation["norm_before"]:
        raise ValueError("Optimizer did not change the observed adapter parameter")
    report = {"status": "ok", "optimizer_steps": trainer.state.global_step,
              "gradient_accumulation": GRADIENT_ACCUMULATION, "loss": result.training_loss,
              "gradient_parameter": observation["name"], "norm_before": observation["norm_before"],
              "norm_after": norm_after, "tokens": len(features["input_ids"]),
              "image_patches": len(features["pixel_values"]), "cpu_structural_smoke": use_cpu}
    if not use_cpu:
        torch.cuda.synchronize()
        report.update(gpu=torch.cuda.get_device_name(0),
                      gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true", help="CPU data/token preflight; no model weights loaded")
    mode.add_argument("--execute", action="store_true", help="Explicitly run CUDA training and final test A")
    mode.add_argument("--smoke-train", action="store_true", help="One CUDA optimizer step on the heaviest train page")
    parser.add_argument("--corpus", type=Path, help="Local pinned corpus; otherwise download train/dev only")
    parser.add_argument("--benchmark", type=Path, help="Locally staged frozen test A")
    parser.add_argument("--model-path", type=Path, help="Local frozen model copy; verify checksums before use")
    parser.add_argument("--output", type=Path, default=WORK)
    parser.add_argument("--upload", action="store_true", help="Upload completed evidence using HF_TOKEN")
    cli = parser.parse_args()
    if cli.execute or cli.smoke_train:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; use --prepare-only on CPU")
    if cli.smoke_train and (cli.output / "smoke-result.json").exists():
        raise FileExistsError("Existing smoke evidence: choose a new --output")
    if cli.execute and (cli.output / "dev-metrics.json").exists():
        raise FileExistsError("Existing experiment results: choose a new --output; do not restart over them.")
    cli.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO))
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoProcessor
    from training.qwen3vl_training_data import collate_pages, tokenize_page
    from training.run_vision_baseline import load_templates

    model_source = str(cli.model_path) if cli.model_path else MODEL_SOURCE
    model_kwargs = {"local_files_only": True} if cli.model_path else {"revision": MODEL_REVISION}
    if cli.model_path:
        checksums = json.loads((cli.model_path / "checksums.json").read_text(encoding="utf-8"))
        for filename, expected in checksums.items():
            if digest(cli.model_path / filename) != expected:
                raise ValueError(f"Frozen model checksum mismatch: {filename}")
        freeze = json.loads((cli.model_path / "freeze-manifest.json").read_text(encoding="utf-8"))
        if freeze["source_revision"] != "ebb281ec70b05090aa6165b016eac8ec08e71b17":
            raise ValueError("Local model is not the frozen Qwen3-VL source")
    if not cli.corpus:
        raise ValueError("--corpus is required: pinned poleval-corpus-v1 directory")
    corpus = cli.corpus
    if digest(corpus / "pages.jsonl") != CORPUS_MANIFEST_SHA256:
        raise ValueError("Poleval corpus manifest checksum mismatch")
    if cli.benchmark:
        staged = cli.benchmark
    else:
        archive = Path(hf_hub_download(IMPACT_REPOSITORY, IMPACT_PATH,
                                      repo_type="dataset", revision=IMPACT_REVISION))
        if digest(archive) != IMPACT_SHA256:
            raise ValueError("Frozen test A archive checksum mismatch")
        staged = cli.output / "benchmark"
        if not (staged / "manifest.jsonl").exists():
            subprocess.run([sys.executable, "-m", "training.stage_impact_benchmark",
                            "--archive", str(archive), "--output", str(staged)], cwd=REPO, check=True)
    if digest(staged / "manifest.jsonl") != TEST_MANIFEST_SHA256:
        raise ValueError("Frozen test A manifest checksum mismatch")
    test_rows = read_rows(staged / "manifest.jsonl")
    if len(test_rows) != 36:
        raise ValueError(f"Expected 36 frozen test A pages, got {len(test_rows)}")
    train_rows, dev_rows = build_finetune_rows(read_rows(corpus / "pages.jsonl"),
                                             {row["sha256"] for row in test_rows})
    if len(train_rows) != EXPECTED_TRAIN or len(dev_rows) != EXPECTED_DEV:
        raise ValueError(f"Pinned corpus must contain {EXPECTED_TRAIN} train / "
                         f"{EXPECTED_DEV} dev pages, got {len(train_rows)}/{len(dev_rows)}")
    for row in train_rows + dev_rows:
        image = corpus / row["image"]
        if digest(image) != row["image_sha256"]:
            raise ValueError(f"Corpus image checksum mismatch: {row['id']}")
        row["_image"] = str(image)
    processor = AutoProcessor.from_pretrained(
        model_source, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, **model_kwargs)
    prompt = load_templates()["A"]
    audit = []
    for row in train_rows + dev_rows:
        tokens = tokenize_page(row, processor, prompt)
        audit.append({"id": row["id"], "split": row["split"], "image_sha256": row["image_sha256"],
                      "tokens": len(tokens["input_ids"]),
                      "supervised_tokens": int((tokens["labels"] != -100).sum()),
                      "image_patches": len(tokens["pixel_values"]),
                      "image_grid_thw": tokens["image_grid_thw"].tolist()})
        del tokens
    metadata = {
        "protocol": "polocrbench-qwen3vl-poleval-qlora-v1",
        "model": MODEL_SOURCE, "model_revision": MODEL_REVISION,
        "source_dataset": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
        "corpus_manifest_sha256": digest(corpus / "pages.jsonl"),
        "benchmark_manifest_sha256": TEST_MANIFEST_SHA256,
        "train_pages": len(train_rows), "dev_pages": len(dev_rows), "test_split_excluded": True,
        "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS, "max_new_tokens": MAX_NEW_TOKENS,
        "learning_rate": LEARNING_RATE, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "planned_epochs": EPOCHS, "gradient_accumulation": GRADIENT_ACCUMULATION,
        "gradient_checkpointing": True, "use_cache_training": False,
        "attention_policy": "expanded-kv-fused-only-v1",
        "cuda_sdpa_backends": ["FLASH_ATTENTION", "EFFICIENT_ATTENTION"],
        "native_enable_gqa": False,
        "training_loss_policy": "supervised-next-token-ce-v1",
        "loss_reduction": "microbatch-mean-with-trainer-accumulation",
        "prompt_version": "polocrbench-zero-shot-prompt-v1",
        "prompt_sha256": digest(REPO / "benchmarks/polocrbench/prompts/zero_shot_prompt_v1.md"),
        "code_sha256": {name: digest(REPO / "training" / name) for name in
                        ("run_qwen3vl_poleval_qlora.py", "qwen3vl_training_data.py", "qwen3vl_epoch_state.py")},
    }
    (cli.output / "preflight.json").write_text(
        json.dumps(dict(metadata, token_audit=audit), ensure_ascii=False, indent=2), encoding="utf-8")
    print("PREFLIGHT_OK", len(train_rows), "train /", len(dev_rows), "dev; no test leakage", flush=True)
    if cli.prepare_only:
        return

    import torch
    from jiwer import cer, wer
    from PIL import Image
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from training.qwen3vl_epoch_state import EpochLedger
    from training.run_sota_benchmark import run as run_benchmark
    from training.transcription_eval import normalize

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; preflight completed without starting training")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    def load_base():
        engine = Qwen3VLForConditionalGeneration.from_pretrained(
            model_source, device_map={"": 0}, torch_dtype=torch.float16,
            attn_implementation="sdpa", quantization_config=quantization, **model_kwargs)
        return configure_attention(engine).eval()

    if cli.smoke_train:
        phase = "model-load"
        report = dict(metadata, status="error")
        try:
            heaviest = max((row for row in audit if row["split"] == "train"),
                          key=lambda row: (row["tokens"], row["image_patches"], row["id"]))
            report["page_id"] = heaviest["id"]
            row = next(row for row in train_rows if row["id"] == heaviest["id"])
            features = tokenize_page(row, processor, prompt)
            engine = load_base()
            phase = "lora-preparation"
            engine = prepare_lora_model(engine)
            phase = "optimizer-step"
            report.update(smoke_train_step(engine, features, cli.output, processor.tokenizer.pad_token_id))
        except Exception as error:
            report.update(status="error", phase=phase, error_type=type(error).__name__, error=str(error))
            raise
        finally:
            (cli.output / "smoke-result.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SMOKE_TRAIN_OK", report["page_id"], "peak_reserved_bytes",
              report["peak_reserved_bytes"], flush=True)
        return

    engine = load_base()
    ledger = EpochLedger(cli.output, metadata)

    def predict(path):
        with Image.open(path) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(engine.device)
        eos = engine.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        with torch.inference_mode():
            outputs = engine.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      use_cache=True, eos_token_id=eos,
                                      pad_token_id=processor.tokenizer.pad_token_id)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        if len(generated) >= MAX_NEW_TOKENS and int(generated[-1]) not in eos:
            raise ValueError("Incomplete response: generation hit max_new_tokens")
        return processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def evaluate_dev(epoch):
        engine.eval()
        references, hypotheses = [], []
        errors = 0
        with (cli.output / f"dev-epoch{epoch}-predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in dev_rows:
                started = time.monotonic()
                try:
                    text = predict(row["_image"])
                    result = {"id": row["id"], "status": "ok", "text": text}
                except torch.OutOfMemoryError:
                    raise
                except ValueError as error:
                    errors += 1
                    text = ""
                    result = {"id": row["id"], "status": "error", "text": "", "error": str(error)}
                result["elapsed_seconds"] = time.monotonic() - started
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                references.append(normalize(row["text"]))
                hypotheses.append(normalize(text))
        return {"cer_micro": cer(references, hypotheses), "wer_micro": wer(references, hypotheses),
                "pages": len(dev_rows), "errors_or_missing": errors}

    phase = "baseline-dev"
    try:
        baseline = evaluate_dev(0)
        ledger.record(0, baseline)
        print("DEV_BASELINE", baseline, flush=True)
        if ledger.selected is None:
            raise ValueError("Failed DEV baseline; training is not authorized by a valid comparison")
        phase = "training"
        engine = prepare_lora_model(engine)
        engine.print_trainable_parameters()

        class PagesDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(train_rows)

            def __getitem__(self, index):
                return tokenize_page(train_rows[index], processor, prompt)

        args = training_arguments(cli.output)
        trainer = make_training_trainer(
            model=engine, args=args, train_dataset=PagesDataset(),
            data_collator=lambda batch: collate_pages(batch, processor.tokenizer.pad_token_id),
            callbacks=[make_epoch_callback(engine, ledger, evaluate_dev, cli.output)])
        trainer.train()
        selected = ledger.selected
        print("SELECTED_EPOCH", selected, flush=True)
        del trainer
        if selected["adapter"] is None:
            # Reload the original baseline; disable_adapter alone retains training casts.
            engine = None
            gc.collect()
            torch.cuda.empty_cache()
            engine = load_base()
        else:
            engine.load_adapter(selected["adapter"], adapter_name="selected")
            engine.set_adapter("selected")
            engine.gradient_checkpointing_disable()
            engine.eval()
        phase = "test-A"
        out = cli.output / "testA"
        run_benchmark("qwen3vl", staged, out, predictor=predict)
        run_path = out / "run.json"
        run_metadata = json.loads(run_path.read_text(encoding="utf-8"))
        run_metadata.update(model_source=MODEL_SOURCE, model_revision=MODEL_REVISION,
                            selected_epoch=selected["epoch"], adapter=selected["adapter"],
                            training_performed=selected["epoch"] > 0)
        run_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
        score = json.loads((out / "score.json").read_text(encoding="utf-8"))
        print("TEST_A_RESULT", {key: score[key] for key in
                                ("cer_micro", "wer_micro", "structure_similarity", "errors_or_missing")}, flush=True)
        phase = "evidence"
        evidence = cli.output / "evidence"
        evidence.mkdir()
        for name in ("predictions.jsonl", "score.json", "run.json"):
            shutil.copyfile(out / name, evidence / f"qwen3vl-selected-{name}")
        for name in ("dev-metrics.json", "preflight.json"):
            shutil.copyfile(cli.output / name, evidence / name)
        for path in cli.output.glob("dev-epoch*-predictions.jsonl"):
            shutil.copyfile(path, evidence / path.name)
        shutil.copyfile(staged / "verification.json", evidence / "staged-verification.json")
        if selected["adapter"]:
            shutil.copytree(selected["adapter"], evidence / "adapter")
        (evidence / "receipt.json").write_text(json.dumps(
            dict(metadata, selected=selected, training_performed=True, benchmark_touched=False,
                 promotion=False, sota_claim=False), indent=2), encoding="utf-8")
        hashes = {path.relative_to(evidence).as_posix(): digest(path)
                  for path in evidence.rglob("*") if path.is_file()}
        (evidence / "checksums.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        zip_path = Path(shutil.make_archive(str(cli.output / "qwen3vl-poleval-qlora-v1-evidence"), "zip", evidence))
        print("GOTOWY_ZIP", zip_path, zip_path.stat().st_size, flush=True)
        if cli.upload:
            from huggingface_hub import HfApi
            if not os.environ.get("HF_TOKEN"):
                raise ValueError("HF_TOKEN required for --upload; local evidence is preserved")
            info = HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                repo_id=EVIDENCE_REPOSITORY, repo_type="dataset",
                path_or_fileobj=str(zip_path), path_in_repo=f"{EVIDENCE_PATH}/{zip_path.name}")
            print("HF_UPLOAD", EVIDENCE_REPOSITORY, info.oid, flush=True)
        print("DONE", flush=True)
    except Exception as error:
        if ledger.failure is None:
            ledger.mark_failure(phase, error)
        raise


if __name__ == "__main__":
    main()
