# -*- coding: utf-8 -*-
"""QLoRA fine-tune SmolVLM na polskich stronach dokumentow (pool polocrbench-train-pool-v1).

Adaptacja przepisu `smolvlm.py` (merveenoyan/smol-vision, Apache-2.0) na polskie
dane dokumentowe: zamiast VQAv2 trenujemy transkrypcje stron (zadanie A, Markdown).
Konfiguracja QLoRA (4-bit nf4, LoRA r=8/alpha=8, target_modules jak w oryginale)
i sposob skladania batcha (maskowanie -100 na padach i tokenach obrazu) zostaja
wierne przepisowi.

Dane (publiczny pool HF, manifest-A.jsonl + images/):
  huggingface-cli download PiotrSty/polocrbench-train-pool-v1 manifest.jsonl images --repo-type dataset --local-dir data/pool

Trening (Colab/Kaggle - GPU wymagane; lokalnie tylko --dry-run):
  python training/train_smolvlm_pl.py --pool data/pool --dry-run
  python training/train_smolvlm_pl.py --pool data/pool --epochs 2 --output out/smolvLM-pl

Wynik: adapter LoRA + logi; ewaluacja modelem w training.transcription_eval osobno.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_rows(manifest: Path, limit: int):
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != "train":  # development trzymamy na ewaluacje
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


PROMPT = ("Przepisz cala strone dokumentu w formacie Markdown, zachowujac kolejnosc "
          "czytania, naglowki, listy i akapity. Tabele jako HTML, wzory jako LaTeX w $. "
          "Nie dodawaj komentarzy - sam tekst dokumentu.")


def build_messages(image, text: str):
    return [
        {"role": "user", "content": [
            {"type": "text", "text": "Odczytaj dokument."},
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": text}]},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="katalog z manifest.jsonl + images/")
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Base")
    ap.add_argument("--output", default="out/smolvLM-pl")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0, help="0 = wszystkie wiersze train")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--dry-run", action="store_true", help="tylko przygotowanie batcha, bez treningu")
    args = ap.parse_args()

    pool = Path(args.pool)
    rows = load_rows(pool / "manifest.jsonl", args.limit)
    n_val = max(1, int(len(rows) * args.val_fraction))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"wiersze: train {len(train_rows)} | val {len(val_rows)} (z manifestu: manifest.jsonl, split=train)")

    from PIL import Image
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    image_token_id = tok.convert_tokens_to_ids("<image>")
    if image_token_id in (None, getattr(tok, "unk_token_id", None)):
        ids = getattr(tok, "additional_special_tokens_ids", None)
        if ids:
            image_token_id = ids[tok.additional_special_tokens.index("<image>")]

    def collate_fn(examples):
        texts, images = [], []
        for row, img_path in examples:
            image = Image.open(img_path).convert("RGB")
            messages = build_messages(image, row.get("text", ""))
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    def examples_from(rows_):
        out = []
        for r in rows_:
            img = pool / r["image"]
            if img.exists():
                out.append((r, img))
        return out

    train_ex, val_ex = examples_from(train_rows), examples_from(val_rows)
    print(f"obrazy znalezione: train {len(train_ex)} | val {len(val_ex)}")

    if args.dry_run:
        b = collate_fn(train_ex[:2])
        print("dry-run OK | batch keys:", sorted(b.keys()))
        print("input_ids:", tuple(b["input_ids"].shape), "| images:", len(b["pixel_values"]) if "pixel_values" in b else "?")
        print("labels z -100:", int((b["labels"] == -100).sum()), "/", b["labels"].numel())
        return

    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from transformers import BitsAndBytesConfig, TrainingArguments, Trainer

    lora_config = LoraConfig(
        r=8, lora_alpha=8, lora_dropout=0.1,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        use_dora=False, init_lora_weights="gaussian")
    lora_config.inference_mode = False
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    model = Idefics3ForConditionalGeneration.from_pretrained(
        args.model, quantization_config=bnb_config,
        _attn_implementation="flash_attention_2", device_map="auto")
    model.add_adapter(lora_config)
    model.enable_adapters()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print(model.get_nb_trainable_parameters())

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, ex): self.ex = ex
        def __len__(self): return len(self.ex)
        def __getitem__(self, i): return self.ex[i]

    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=50,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ListDataset(train_ex),
        data_collator=collate_fn,
    )
    trainer.train()
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    print("ZAPISANO:", args.output)


if __name__ == "__main__":
    main()
