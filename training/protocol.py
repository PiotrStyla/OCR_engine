"""Shared training, decoding and dataset provenance for TrOCR experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer


def configure_generation(model, tokenizer, start_token=None):
    if start_token is None:
        start_token = model.generation_config.decoder_start_token_id
    if start_token is None:
        raise ValueError("Base model must declare decoder_start_token_id")
    tokens = dict(
        decoder_start_token_id=start_token,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    for key, value in tokens.items():
        setattr(model.config, key, value)
        setattr(model.generation_config, key, value)
    for key, value in dict(
        max_length=128,
        max_new_tokens=None,
        num_beams=4,
        do_sample=False,
        early_stopping=True,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        use_cache=True,
    ).items():
        setattr(model.generation_config, key, value)
    return model.generation_config.to_dict()


def pair_manifest(directory):
    records = []
    for image in sorted(Path(directory).glob("*.png")):
        label = image.with_suffix(".txt")
        if not label.exists() or not label.read_text(encoding="utf-8").strip():
            raise ValueError(f"Missing or empty label: {image.name}")
        records.append(
            dict(
                id=image.stem,
                image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                text_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
            )
        )
    if not records:
        raise ValueError("Empty dataset")
    return records


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def compute_ocr_metrics(tokenizer):
    def compute(prediction):
        import numpy as np
        from jiwer import cer, wer

        pred = prediction.predictions
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = np.where(pred == -100, tokenizer.pad_token_id, pred)
        labels = np.where(
            prediction.label_ids == -100,
            tokenizer.pad_token_id,
            prediction.label_ids,
        )
        hypotheses = tokenizer.batch_decode(pred, skip_special_tokens=True)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"cer": cer(references, hypotheses), "wer": wer(references, hypotheses)}

    return compute


def aligned_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: torch.Tensor | None = None,
) -> torch.Tensor:
    targets = labels.reshape(-1).to(logits.device)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets,
        ignore_index=-100,
        reduction="none",
    )
    valid = targets.ne(-100)
    total = losses[valid].sum()
    denominator = num_items_in_batch if num_items_in_batch is not None else valid.sum()
    return total / denominator.to(device=logits.device, dtype=total.dtype).clamp_min(1)


class AlignedSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        labels = inputs["labels"]
        prepare = getattr(model, "prepare_decoder_input_ids_from_labels", None)
        if prepare is None and hasattr(model, "get_base_model"):
            prepare = model.get_base_model().prepare_decoder_input_ids_from_labels
        if prepare is None:
            raise TypeError("Model does not expose prepare_decoder_input_ids_from_labels")
        decoder_input_ids = prepare(labels=labels)
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs, decoder_input_ids=decoder_input_ids, use_cache=False)
        loss = aligned_token_loss(outputs.logits, labels, num_items_in_batch)
        return (loss, outputs) if return_outputs else loss
