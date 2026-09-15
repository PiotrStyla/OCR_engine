"""Backend OCR oparty na Kraken — end-to-end segmentacja + rozpoznawanie.

Kraken to system OCR/HTR zoptymalizowany pod dokumenty historyczne i
niewydrukowane skrypty. Używa trainable baseline segmentation (sieć neuronowa
wykrywająca linie-bazy) oraz modeli rozpoznawania .mlmodel (CTC).

Dobre dla: maszynopisów, dokumentów historycznych, wyblakłych skanów.
Wymaga extras `[kraken]` (`pip install -e .[kraken]`).

Model domyślny: `polish_nfd_9313.mlmodel` z EHRI (93,1% accuracy na polskim
maszynopisie) — znacznie lepszy niż TrOCR na tej domenie.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from .result import BBox, Language, OcrResult, TextLine

if TYPE_CHECKING:
    from .config import OcrConfig

logger = logging.getLogger(__name__)


def _resolve_kraken_model(model_spec: str) -> str:
    """Rozwiązuje spec modelu na ścieżkę lokalnego pliku .mlmodel.

    Formaty:
      - ścieżka lokalna: /path/to/model.mlmodel → zwraca jak jest
      - HF repo+plik: PiotrSty/ehri-dataset::models/polish_nfd_9313.mlmodel
        → pobiera z HF i zwraca ścieżkę cache
    """
    if os.path.isfile(model_spec):
        return model_spec
    if "::" in model_spec:
        repo_id, filename = model_spec.split("::", 1)
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, filename, repo_type="dataset")
        return str(path)
    if "/" in model_spec and model_spec.endswith(".mlmodel"):
        # Może być repo/model.mlmodel bez :: — spróbuj jako HF repo
        parts = model_spec.rsplit("/", 1)
        if len(parts) == 2:
            from huggingface_hub import hf_hub_download
            try:
                path = hf_hub_download(parts[0], parts[1], repo_type="dataset")
                return str(path)
            except Exception:
                try:
                    path = hf_hub_download(parts[0], parts[1], repo_type="model")
                    return str(path)
                except Exception as e:
                    raise FileNotFoundError(f"Nie można pobrać modelu Kraken: {model_spec}") from e
    raise FileNotFoundError(f"Model Kraken nie znaleziony: {model_spec}")


class KrakenBackend:
    """End-to-end backend Kraken: segmentacja baseline + rozpoznawanie .mlmodel.

    Bypassuje TextDetector i Recognizer — robi wszystko w jednym przebiegu.
    Zwraca OcrResult z liniami, bboxami i confidence.
    """

    def __init__(self, config: "OcrConfig") -> None:
        self.config = config
        self.device = config.resolved_device()
        self._model_path: str | None = None
        self._recognizer = None  # kraken.lib.models.TorchSeqRecognizer

    def _ensure_loaded(self) -> None:
        """Leniwe ładowanie modelu rozpoznawania Kraken."""
        if self._recognizer is not None:
            return
        import kraken.lib.models as models

        self._model_path = _resolve_kraken_model(self.config.kraken_model)
        logger.info("Ładowanie Kraken: %s (device=%s)", self._model_path, self.device)
        self._recognizer = models.load_any(self._model_path, device=self.device)
        logger.info("Kraken załadowany")

    def recognize(self, image) -> OcrResult:
        """End-to-end OCR: segmentacja + rozpoznawanie. Zwraca OcrResult."""
        self._ensure_loaded()
        from kraken.blla import segment
        from kraken.rpred import rpred

        # Konwertuj na PIL Image (Kraken wymaga PIL)
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        elif isinstance(image, (str, Path)):
            pil_img = Image.open(image)
        else:
            pil_img = image  # już PIL

        if pil_img.mode != "L":
            pil_img = pil_img.convert("L")

        # Segmentacja baseline (Kraken używa wbudowanego modelu segmentacji)
        seg = segment(pil_img)
        if not seg.lines:
            return OcrResult(lines=[], image_size=pil_img.size)

        # Rozpoznawanie linia po linii
        pred = rpred(self._recognizer, pil_img, seg)
        lines: list[TextLine] = []
        for record in pred:
            text = record.prediction.strip()
            if not text:
                continue
            # BBox z polygonu baselinu — użyj boundary (z BaselineLine)
            polygon = record.boundary or record.baseline
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            bbox = BBox(
                int(min(xs)), int(min(ys)),
                int(max(xs)), int(max(ys)),
            )
            # Confidence: średnia z confidence per znak (jeśli dostępne)
            confs = record.confidences or []
            confidence = float(sum(confs) / len(confs)) if confs else float("nan")
            lines.append(TextLine(
                text=text,
                bbox=bbox,
                language="pl",  # model jest polski
                confidence=confidence,
            ))

        return OcrResult(lines=lines, image_size=pil_img.size)

    def close(self) -> None:
        self._recognizer = None
