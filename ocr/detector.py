"""Detekcja linii tekstu: CRAFT (główny) lub OpenCV (fallback).

Zwraca listę BBox otaczających wykryte linie tekstu, posortowanych
w porządku czytania (top→bottom, left→right).

Auto-detekcja: jeśli craft-text-detector jest zainstalowany, używany jest CRAFT.
W przeciwnym razie fallback do detektora OpenCV (morfologia + kontury).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .preprocess import load_image
from .result import BBox

if TYPE_CHECKING:
    from .config import OcrConfig

logger = logging.getLogger(__name__)


def _craft_available() -> bool:
    try:
        import craft_text_detector  # noqa: F401
        return True
    except ImportError:
        return False


class TextDetector:
    """Opakowanie na detektor tekstu: CRAFT lub OpenCV fallback."""

    def __init__(self, config: "OcrConfig") -> None:
        self.config = config
        self._craft = None
        self._opencv_detector = None
        self._use_craft: bool | None = None  # leniwe

    def _resolve_backend(self) -> bool:
        """Zwraca True jeśli używamy CRAFT, False jeśli OpenCV fallback."""
        if self._use_craft is not None:
            return self._use_craft
        self._use_craft = _craft_available()
        if self._use_craft:
            logger.info("Detektor: CRAFT (craft-text-detector dostępny)")
        else:
            logger.info("Detektor: OpenCV fallback (craft-text-detector niedostępny)")
        return self._use_craft

    def _ensure_craft_loaded(self) -> None:
        if self._craft is not None:
            return
        from craft_text_detector import Craft  # leniwy import (ciężka zależność)

        device = "cuda" if self.config.resolved_device() == "cuda" else "cpu"
        self._craft = Craft(
            output_dir=None,
            crop_type="box",
            cuda=(device == "cuda"),
            text_threshold=self.config.text_threshold,
            link_threshold=self.config.link_threshold,
            low_text=self.config.low_text_threshold,
        )
        logger.info("CRAFT załadowany (device=%s)", device)

    def _ensure_opencv_loaded(self):
        if self._opencv_detector is not None:
            return
        from .opencv_detector import OpenCVDetector
        self._opencv_detector = OpenCVDetector(self.config)

    def detect(self, image) -> list[BBox]:
        """Wykrywa linie tekstu. `image` może być ścieżką lub array RGB."""
        if self._resolve_backend():
            return self._detect_craft(image)
        return self._detect_opencv(image)

    def _detect_craft(self, image) -> list[BBox]:
        self._ensure_craft_loaded()
        arr = load_image(image) if not isinstance(image, np.ndarray) else image
        import cv2
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        result = self._craft.detect_text(bgr)
        boxes = result.get("boxes")
        if boxes is None:
            boxes = []

        bboxes: list[BBox] = []
        h, _ = arr.shape[:2]
        for box in boxes:
            xs = box[:, 0]
            ys = box[:, 1]
            bbox = BBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            if bbox.area < self.config.min_line_area:
                continue
            bboxes.append(bbox)

        bboxes = sort_reading_order(bboxes, h)
        logger.debug("CRAFT: wykryto %d linii", len(bboxes))
        return bboxes

    def _detect_opencv(self, image) -> list[BBox]:
        self._ensure_opencv_loaded()
        return self._opencv_detector.detect(image)

    def close(self) -> None:
        if self._craft is not None:
            try:
                self._craft.unload_models()
            except Exception:  # pragma: no cover - zależne od wersji
                pass
            self._craft = None
        self._opencv_detector = None


def sort_reading_order(bboxes: list[BBox], image_height: int) -> list[BBox]:
    """Sortuje bboxy w porządku czytania.

    Grupuje linie w poziome pasy (kolejność top→bottom), a wewnątrz pasa
    sortuje left→right. Pas określamy przez środek pionowy bboxa.
    """
    if not bboxes:
        return []
    # tolerancja grupowania: 50% mediany wysokości linii
    heights = [b.height for b in bboxes]
    median_h = sorted(heights)[len(heights) // 2] or 1
    tol = median_h * 0.5

    by_top = sorted(bboxes, key=lambda b: (b.y1, b.x1))
    rows: list[list[BBox]] = []
    for bbox in by_top:
        center_y = (bbox.y1 + bbox.y2) / 2
        placed = False
        for row in rows:
            row_center = sum((b.y1 + b.y2) / 2 for b in row) / len(row)
            if abs(center_y - row_center) <= tol:
                row.append(bbox)
                placed = True
                break
        if not placed:
            rows.append([bbox])

    ordered: list[BBox] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b.x1))
    return ordered
