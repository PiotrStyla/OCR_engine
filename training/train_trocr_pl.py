"""Fine-tune TrOCR with decoder LoRA and best-validation-CER selection.

Use a new output directory and a pinned base revision. The Kaggle notebook
uses ordinary LoRA; 4-bit remains an optional experimental path.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import importlib.metadata
import subprocess

import torch
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    set_seed,
)

from .dataset import TrOCRLineDataset, load_pairs
from .protocol import (
    AlignedSeq2SeqTrainer,
    compute_ocr_metrics,
    configure_generation,
    pair_manifest,
    write_json,
)

logger = logging.getLogger(__name__)


def _collate(batch, processor):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def _validate_lora_targets(wrapped, target_modules: list[str]) -> None:
    for name in target_modules:
        if not hasattr(wrapped.base_model.model.get_submodule(name), "lora_A"):
            raise ValueError(f"Adapter was not attached: {name}")


def _inject_lora(model, lora_rank: int, lora_alpha: int, include_mlp=False) -> object:
    """Wstrzykuje adaptery LoRA do dekodera modelu (PEFT)."""
    from peft import LoraConfig, TaskType, get_peft_model

    targets = ["q_proj", "k_proj", "v_proj", "out_proj"]
    if include_mlp:
        targets += ["fc1", "fc2"]
    names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    matched = {
        target: [
            name
            for name in names
            if name.startswith("decoder.") and name.endswith("." + target)
        ]
        for target in targets
    }
    if any(not values for values in matched.values()):
        raise ValueError(f"Missing expected decoder LoRA modules: {matched}")
    target_modules = [name for values in matched.values() for name in values]
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    wrapped = get_peft_model(model, lora_config)
    _validate_lora_targets(wrapped, target_modules)
    return wrapped


def train(
    train_dir: str,
    val_dir: str | None,
    base_model: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 2e-4,  # QLoRA używa wyższego LR niż full-FT
    lora_rank: int = 16,
    lora_alpha: int = 32,
    use_4bit: bool = True,
    neftune_noise: float = 0.0,
    revision: str | None = None,
    include_mlp: bool = False,
    seed: int = 42,
) -> None:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory; preserve previous run')
    if not val_dir:
        raise ValueError('Validation data is required for best-CER checkpoint selection')
    train_records, val_records = pair_manifest(train_dir), pair_manifest(val_dir)
    if {r['image_sha256'] for r in train_records} & {r['image_sha256'] for r in val_records}:
        raise ValueError('Exact image overlap between train and validation')
    output.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and use_4bit:
        logger.warning(
            "Brak CUDA — 4-bit bitsandbytes wymaga GPU. "
            "Przełączam na zwykłe LoRA (wolne na CPU, tylko do smoke-testu)."
        )
        use_4bit = False

    processor = TrOCRProcessor.from_pretrained(base_model, revision=revision)

    # QLoRA: 4-bit quantization przy ładowaniu bazowego modelu.
    # bnb 4-bit + VisionEncoderDecoderModel potrafi crashować w transformers
    # (missing-keys init) — wtedy fallback na pełne LoRA fp16; przy 336M
    # parametrach TrOCR-base to i tak wystarczające na 16 GB VRAM.
    model = None
    if use_4bit:
        try:
            model = VisionEncoderDecoderModel.from_pretrained(
                base_model,
                revision=revision,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
                    bnb_4bit_use_double_quant=True,
                ),
            )
            logger.info("QLoRA: 4-bit NF4 quantization włączona")
        except (AttributeError, RuntimeError, ImportError) as exc:
            logger.warning(
                "4-bit load nie powiódł się (%s) — przełączam na pełne LoRA fp16",
                exc,
            )
            use_4bit = False
    if model is None:
        model = VisionEncoderDecoderModel.from_pretrained(base_model, revision=revision)
    if not use_4bit:
        model.to(device)

    # konfiguracja dekodera
    generation = configure_generation(model, processor.tokenizer)

    # LoRA: adaptery zawsze (metoda treningu); przy 4-bit najpierw przygotowanie kbit
    if use_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    model = _inject_lora(model, lora_rank, lora_alpha, include_mlp)
    model.print_trainable_parameters()

    train_samples = load_pairs(train_dir)
    if not train_samples:
        raise SystemExit(f"Brak danych treningowych w {train_dir}")
    train_ds = TrOCRLineDataset(train_samples, processor)

    val_ds = None
    if val_dir:
        val_samples = load_pairs(val_dir)
        if val_samples:
            val_ds = TrOCRLineDataset(val_samples, processor)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model='cer',
        greater_is_better=False,
        save_total_limit=2,
        seed=seed,
        data_seed=seed,
        # bf16 dopasowane do bnb_4bit_compute_dtype; fp16 tylko bez 4-bit
        bf16=(device == "cuda" and use_4bit and torch.cuda.is_bf16_supported()),
        fp16=(device == "cuda" and not use_4bit),
        logging_steps=50,
        report_to="none",
        # NEFTune is disabled by default; treat enabling it as a separate experiment.
        neftune_noise_alpha=neftune_noise if neftune_noise > 0 else None,
    )
    try:
        source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = None
    write_json(output/'run.json', dict(base_model=base_model, requested_revision=revision,
        resolved_revision=getattr(model.config, '_commit_hash', None), source_commit=source_commit,
        train=train_records, validation=val_records, seed=seed, generation=generation,
        use_4bit=use_4bit, include_mlp=include_mlp, neftune_noise=neftune_noise,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        packages={name:importlib.metadata.version(name) for name in ['torch','transformers','peft','accelerate','jiwer']},
        training_arguments=args.to_dict()))

    trainer = AlignedSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda b: _collate(b, processor),
        compute_metrics=compute_ocr_metrics(processor.tokenizer),
    )
    trainer.train()
    trainer.save_state()
    write_json(output/'best_metrics.json', trainer.evaluate())
    write_json(output/'selection.json', dict(best_checkpoint=trainer.state.best_model_checkpoint,
                                           best_cer=trainer.state.best_metric))

    # Zapis: adaptery w output_dir/adapter + scalony pełny model w output_dir
    # (Recognizer ładuje pełny model przez VisionEncoderDecoderModel.from_pretrained)
    model.save_pretrained(str(Path(output_dir) / "adapter"))
    merged = model.merge_and_unload()
    configure_generation(merged, processor.tokenizer, generation['decoder_start_token_id'])
    merged.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Model zapisany w %s (adaptery: %s)", output_dir, Path(output_dir) / "adapter")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune TrOCR (PL) — QLoRA")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", default=None)
    p.add_argument("--base", default="microsoft/trocr-base-printed")
    p.add_argument("--output", default="./ocr/trocr-pl-base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--no-4bit", action="store_true", help="Wyłącz 4-bit (full LoRA)")
    p.add_argument("--no-neftune", action="store_true", help="Wyłącz NEFTune")
    p.add_argument('--revision', default=None)
    p.add_argument('--include-mlp', action='store_true', help='Separate capacity experiment: add fc1/fc2')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = _parse_args()
    train(
        a.train_dir, a.val_dir, a.base, a.output,
        a.epochs, a.batch_size, a.lr,
        a.lora_rank, a.lora_alpha,
        use_4bit=not a.no_4bit,
        neftune_noise=0.0,
        revision=a.revision, include_mlp=a.include_mlp, seed=a.seed,
    )


if __name__ == "__main__":
    main()
