"""Shared, explicit decoding and dataset provenance for TrOCR experiments."""
import hashlib
import json
from pathlib import Path


def configure_generation(model, tokenizer, start_token=None):
    if start_token is None:
        start_token = model.generation_config.decoder_start_token_id
    if start_token is None:
        raise ValueError('Base model must declare decoder_start_token_id')
    tokens = dict(decoder_start_token_id=start_token,
                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.sep_token_id)
    for key, value in tokens.items():
        setattr(model.config, key, value)
        setattr(model.generation_config, key, value)
    # Shared baseline/candidate policy. Do not suppress legitimate repeated OCR text.
    for key, value in dict(max_length=128, max_new_tokens=None, num_beams=4,
                           do_sample=False, early_stopping=True, length_penalty=1.0,
                           no_repeat_ngram_size=0, use_cache=True).items():
        setattr(model.generation_config, key, value)
    return model.generation_config.to_dict()


def pair_manifest(directory):
    records = []
    for image in sorted(Path(directory).glob('*.png')):
        label = image.with_suffix('.txt')
        if not label.exists() or not label.read_text(encoding='utf-8').strip():
            raise ValueError(f'Missing or empty label: {image.name}')
        records.append(dict(id=image.stem, image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                            text_sha256=hashlib.sha256(label.read_bytes()).hexdigest()))
    if not records:
        raise ValueError('Empty dataset')
    return records


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def compute_ocr_metrics(tokenizer):
    def compute(prediction):
        import numpy as np
        from jiwer import cer, wer
        pred = prediction.predictions
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = np.where(pred == -100, tokenizer.pad_token_id, pred)
        labels = np.where(prediction.label_ids == -100, tokenizer.pad_token_id, prediction.label_ids)
        hypotheses = tokenizer.batch_decode(pred, skip_special_tokens=True)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {'cer': cer(references, hypotheses), 'wer': wer(references, hypotheses)}
    return compute
