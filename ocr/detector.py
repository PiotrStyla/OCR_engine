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


def sort_reading_order_columns(
    bboxes: list[BBox], image_width: int, image_height: int
) -> list[BBox]:
    """Sort detected lines by columns while retaining wide separators.

    The heuristic is deliberately conservative. Narrow boxes are connected
    into horizontal-overlap components; only layouts with at least two
    components containing multiple lines are treated as multi-column. Boxes
    spanning at least 70% of the page width retain row-major position and split
    the page into vertical sections.
    """
    row_major = sort_reading_order(bboxes, image_height)
    if len(bboxes) < 4 or image_width <= 0:
        return row_major

    indexed = list(enumerate(bboxes))
    spanning = [index for index, box in indexed if box.width >= image_width * 0.70]
    body = [index for index, box in indexed if index not in set(spanning)]
    if len(body) < 4:
        return row_major

    neighbours = {index: set() for index in body}
    for position, left_index in enumerate(body):
        left = bboxes[left_index]
        for right_index in body[position + 1:]:
            right = bboxes[right_index]
            overlap = min(left.x2, right.x2) - max(left.x1, right.x1)
            if overlap <= 0:
                continue
            if overlap / max(1, min(left.width, right.width)) >= 0.25:
                neighbours[left_index].add(right_index)
                neighbours[right_index].add(left_index)

    components: list[list[int]] = []
    unseen = set(body)
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component = []
        unseen.remove(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in neighbours[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(component)

    major = [component for component in components if len(component) >= 2]
    if len(major) < 2:
        return row_major

    def component_center(component: list[int]) -> float:
        left = min(bboxes[index].x1 for index in component)
        right = max(bboxes[index].x2 for index in component)
        return (left + right) / 2

    major.sort(key=component_center)
    column_for_index: dict[int, int] = {}
    for column, component in enumerate(major):
        for index in component:
            column_for_index[index] = column

    major_members = {index for component in major for index in component}
    for component in components:
        if any(index in major_members for index in component):
            continue
        target = min(
            range(len(major)),
            key=lambda column: abs(component_center(component) - component_center(major[column])),
        )
        for index in component:
            column_for_index[index] = target

    def order_section(indices: list[int]) -> list[int]:
        return sorted(
            indices,
            key=lambda index: (
                column_for_index[index],
                bboxes[index].y1,
                bboxes[index].x1,
                bboxes[index].y2,
            ),
        )

    pending = set(body)
    ordered_indices: list[int] = []
    for span_index in sorted(
        spanning,
        key=lambda index: (bboxes[index].y1, bboxes[index].x1),
    ):
        span_center = (bboxes[span_index].y1 + bboxes[span_index].y2) / 2
        before = [
            index
            for index in pending
            if (bboxes[index].y1 + bboxes[index].y2) / 2 < span_center
        ]
        ordered_indices.extend(order_section(before))
        pending.difference_update(before)
        ordered_indices.append(span_index)
    ordered_indices.extend(order_section(list(pending)))

    if len(ordered_indices) != len(bboxes) or len(set(ordered_indices)) != len(bboxes):
        raise RuntimeError("Column ordering did not preserve the detected boxes")
    return [bboxes[index] for index in ordered_indices]
