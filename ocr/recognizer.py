"""Rozpoznawanie tekstu z linii przy użyciu TrOCR.

Ładuje dwa modele (EN i PL) leniwie. PL to lokalny fine-tune; jeśli nie
istnieje, używany jest fallback (EN) zgodnie z konfiguracją.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .preprocess import crop_to_bbox, to_pil_rgb
from .result import Language

if TYPE_CHECKING:
    from .config import OcrConfig

logger = logging.getLogger(__name__)


class _TrOCRBackend:
    """Pojedynczy model TrOCR (processor + model)."""

    def __init__(self, model_name: str, device: str) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def recognize(self, images: Sequence, batch_size: int) -> list[tuple[str, float]]:
        """Rozpoznaje listę PIL.Image. Zwraca (text, confidence)."""
        import torch

        results: list[tuple[str, float]] = []
        for start in range(0, len(images), batch_size):
            batch = list(images[start : start + batch_size])
            pixel_values = self.processor(batch, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    pixel_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=128,
                )
            sequences = generated.sequences
            decoded = self.processor.batch_decode(sequences, skip_special_tokens=True)
            results.extend((text.strip(), _mean_token_conf(generated)) for text in decoded)
        return results


def _mean_token_conf(generated) -> float:
    """Szacuje średnie prawdopodobieństwo wygenerowanych tokenów."""
    try:
        import torch
        scores = generated.scores  # tuple[tensor] per krok
        if not scores:
            return 0.0
        probs = []
        for step, score in enumerate(scores):
            tok = generated.sequences[:, step + 1]  # token na tym kroku
            p = torch.softmax(score, dim=-1).gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            probs.append(p.mean().item())
        return float(sum(probs) / len(probs)) if probs else 0.0
    except Exception:
        return 0.0


def _model_exists(name: str) -> bool:
    """True jeśli `name` to istniejący katalog lokalny lub nazwa repo HF."""
    if os.path.isdir(name):
        return True
    # nazwa HF repo (zawiera '/') — zakładamy dostępność, HF rzuci błędem przy ładowaniu
    return "/" in name


class Recognizer:
    """Zarządza modelami TrOCR dla języków PL i EN."""

    def __init__(self, config: "OcrConfig") -> None:
        self.config = config
        self.device = config.resolved_device()
        self._backends: dict[Language, _TrOCRBackend] = {}

    def _backend_for(self, language: Language) -> _TrOCRBackend:
        if language in self._backends:
            return self._backends[language]

        if language == "pl":
            name = self.config.recognizer_pl
            if not _model_exists(name):
                if self.config.use_fallback_if_pl_missing:
                    logger.warning(
                        "Model PL (%s) niedostępny — używam fallback EN. "
                        "Wytrenuj polski TrOCR (patrz training/).",
                        name,
                    )
                    language = "en"
                    name = self.config.recognizer_fallback
                else:
                    raise FileNotFoundError(f"Model PL niedostępny: {name}")
        else:
            name = self.config.recognizer_en

        if language not in self._backends:
            logger.info("Ładowanie TrOCR (%s): %s", language, name)
            self._backends[language] = _TrOCRBackend(name, self.device)
        return self._backends[language]

    def recognize_lines(
        self,
        image: np.ndarray,
        bboxes: Sequence,
        languages: Sequence[Language],
    ) -> list[tuple[str, float]]:
        """Rozpoznaje linie. `languages` to język per linia (z routingu)."""
        # Pogrupuj po języku, żeby ładować tylko potrzebne modele i batchować
        img_h, img_w = image.shape[:2]
        pad = self.config.line_padding
        crops_by_lang: dict[Language, list[tuple[int, object]]] = {}
        for idx, (bbox, lang) in enumerate(zip(bboxes, languages)):
            if hasattr(bbox, "expand"):
                bbox = bbox.expand(pad, img_w, img_h)
            crop = crop_to_bbox(image, bbox)
            if crop.size == 0:
                continue
            crops_by_lang.setdefault(lang, []).append((idx, to_pil_rgb(crop)))

        out: list[tuple[str, float]] = [("", 0.0)] * len(bboxes)
        for lang, items in crops_by_lang.items():
            backend = self._backend_for(lang)
            idxs = [i for i, _ in items]
            imgs = [im for _, im in items]
            decoded = backend.recognize(imgs, self.config.batch_size)
            for i, (text, conf) in zip(idxs, decoded):
                out[i] = (text, conf)
        return out

    def close(self) -> None:
        self._backends.clear()
