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
            confidence = _token_confidences(self.model, generated)
            results.extend((text.strip(), conf) for text, conf in zip(decoded, confidence))
        return results


def _token_confidences(model, generated) -> list[float]:
    """Mean selected-token probability per sequence; not calibrated accuracy.

    Transformers resolves beam ancestry. EOS, PAD and tokens after EOS are
    excluded. Empty output has unknown confidence, never a fabricated zero.
    """
    import torch
    if not generated.scores:
        return [float("nan")] * len(generated.sequences)
    scores = model.compute_transition_scores(
        generated.sequences, generated.scores,
        beam_indices=getattr(generated, "beam_indices", None), normalize_logits=True,
    )
    tokens = generated.sequences[:, -scores.shape[1]:]
    mask = torch.ones_like(tokens, dtype=torch.bool)
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
    end = torch.zeros_like(mask)
    for token in eos_ids:
        if token is not None:
            end |= tokens == token
    mask &= end.cumsum(dim=1) == 0
    pad = model.generation_config.pad_token_id
    if pad is not None:
        mask &= tokens != pad
    return [float(row[valid].exp().mean()) if valid.any() else float("nan")
            for row, valid in zip(scores, mask)]


def _paddlevl_pipeline_version(model_name: str) -> str:
    """Mapuje nazwę repo HF na pipeline_version pakietu paddleocr."""
    versions = {"PaddlePaddle/PaddleOCR-VL": "v1",
                "PaddlePaddle/PaddleOCR-VL-1.5": "v1.5",
                "PaddlePaddle/PaddleOCR-VL-1.6": "v1.6"}
    if model_name not in versions:
        raise ValueError("Unsupported PaddleOCR-VL model. Use an explicit supported "
                         "PaddlePaddle model ID; custom weights/adapters are not supported.")
    return versions[model_name]


class _PaddleVLBackend:
    """PaddleOCR-VL (0.9B VLM) przez pakiet `paddleocr` — rozpoznawanie linii.

    Model wielojęzyczny (109 języków, w tym polski) — bez fine-tuningu.
    Wymaga extras `[vlm]` (paddlepaddle + paddleocr[doc-parser]).
    VLM nie raportuje confidence → zwracany jest NaN (linie nie są
    flagowane jako niskopewne).
    """

    def __init__(self, model_name: str, device: str) -> None:
        from paddleocr import PaddleOCRVL

        self._pipe = PaddleOCRVL(
            pipeline_version=_paddlevl_pipeline_version(model_name),
            # dla pojedynczych linii: bez preprocessingu i detekcji layoutu
            use_layout_detection=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            device="gpu" if device == "cuda" else "cpu",
        )

    def recognize(self, images: Sequence, batch_size: int) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for img in images:
            arr = np.asarray(img.convert("RGB"))
            text = ""
            for res in self._pipe.predict(arr):
                blocks = res.json["res"].get("parsing_res_list", [])
                blocks = sorted(blocks, key=lambda b: b.get("block_order", 0))
                text = "\n".join(
                    b.get("block_content", "")
                    for b in blocks if b.get("block_label") != "image"
                ).strip()
            results.append((text, float("nan")))
        return results


def _model_exists(name: str) -> bool:
    """True jeśli `name` to istniejący katalog lokalny lub realne repo HF."""
    if os.path.isdir(name):
        return True
    if "/" not in name:
        return False
    # wygląda jak repo HF — zweryfikuj (krótki timeout; brak sieci → False)
    try:
        from huggingface_hub import model_info
        model_info(name, timeout=5)
        return True
    except Exception:
        return False


class Recognizer:
    """Zarządza modelami rozpoznawania (TrOCR per język lub VLM wielojęzyczny)."""

    def __init__(self, config: "OcrConfig") -> None:
        self.config = config
        self.device = config.resolved_device()
        self._backends: dict[str, object] = {}

    def has_model_for(self, language: Language) -> bool:
        """True jeśli model dla języka istnieje (bez ładowania)."""
        if self.config.recognizer_backend == "paddlevl":
            # PaddleOCR-VL obsługuje 109 języków natywnie — wystarczy pakiet
            try:
                import paddleocr  # noqa: F401
                return True
            except ImportError:
                return False
        name = self.config.recognizer_pl if language == "pl" else self.config.recognizer_en
        return _model_exists(name)

    def _backend_for(self, language: Language):
        if self.config.recognizer_backend == "paddlevl":
            if "paddlevl" not in self._backends:
                logger.info("Ładowanie PaddleOCR-VL: %s", self.config.paddlevl_model)
                self._backends["paddlevl"] = _PaddleVLBackend(
                    self.config.paddlevl_model, self.device
                )
            return self._backends["paddlevl"]

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
