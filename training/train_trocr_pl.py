"""Fine-tuning TrOCR na polskich liniach tekstu — QLoRA (wg wskazówek Slayer).

QLoRA = 4-bit quantization bazowego modelu + trening tylko adapterów LoRA.
Pozwala trenować 11B modeli na jednym GPU (8–16 GB VRAM) zamiast full-FT.

Wskazówki Slayer (https://slayer.fabryka.ai/trening):
  - QLoRA / LoRA: 4-bit + adaptery → 11–14B na jednym GPU
  - Unsloth / Liger: 2–4× szybciej, mniej VRAM
  - NEFTune: darmowy zysk jakości

Uruchomienie (wymaga CUDA + bitsandbytes + peft):

    python -m training.train_trocr_pl \
        --train-dir ./data/pl_lines_train \
        --val-dir   ./data/pl_lines_val \
        --base microsoft/trocr-base-printed \
        --output ./ocr/trocr-pl-base

Dane: katalog z parami <id>.png + <id>.txt (pojedyncza linia tekstu).
Patrz training/README.md.
"""

from __future__ import annotations

import argparse
import logging

import torch
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from .dataset import TrOCRLineDataset, load_pairs

logger = logging.getLogger(__name__)


def _collate(batch, processor):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def _inject_lora(model, lora_rank: int, lora_alpha: int) -> object:
    """Wstrzykuje adaptery LoRA do dekodera modelu (PEFT)."""
    from peft import LoraConfig, TaskType, get_peft_model

    # TrOCR: dekoder to TrOCRForCausalLM — targetujemy projekcje attention
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    return get_peft_model(model, lora_config)


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
    neftune_noise: float = 5.0,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning(
            "Brak CUDA — QLoRA wymaga GPU (bitsandbytes). "
            "Uruchom na maszynie z CUDA."
        )

    processor = TrOCRProcessor.from_pretrained(base_model)

    # QLoRA: 4-bit quantization przy ładowaniu bazowego modelu
    quant_kwargs = {}
    if use_4bit and device == "cuda":
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("QLoRA: 4-bit NF4 quantization włączona")

    model = VisionEncoderDecoderModel.from_pretrained(base_model, **quant_kwargs)
    model.to(device)

    # konfiguracja dekodera
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # QLoRA: wstrzyknięcie adapterów (tylko one są trenowane)
    if use_4bit and device == "cuda":
        model = _inject_lora(model, lora_rank, lora_alpha)
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
        # bf16 dopasowane do bnb_4bit_compute_dtype; fp16 tylko bez 4-bit
        bf16=(device == "cuda" and use_4bit),
        fp16=(device == "cuda" and not use_4bit),
        logging_steps=50,
        report_to="none",
        # NEFTune: dodaje szum do embeddingów — darmowy zysk jakości (Slayer)
        neftune_noise_alpha=neftune_noise if neftune_noise > 0 else None,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda b: _collate(b, processor),
    )
    trainer.train()

    # Zapis: adaptery LoRA + processor. Pełny model można scalić później.
    if use_4bit and device == "cuda":
        model.save_pretrained(output_dir)  # zapisuje tylko adaptery
    else:
        model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Model zapisany w %s", output_dir)


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
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = _parse_args()
    train(
        a.train_dir, a.val_dir, a.base, a.output,
        a.epochs, a.batch_size, a.lr,
        a.lora_rank, a.lora_alpha,
        use_4bit=not a.no_4bit,
        neftune_noise=0.0 if a.no_neftune else 5.0,
    )


if __name__ == "__main__":
    main()
