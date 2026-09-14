"""Detektor tekstu oparty na OpenCV — fallback gdy CRAFT niedostępny.

Wykorzystuje morfologię: dylatacja pozioma łączy litery w linie tekstu,
potem kontury wyznaczają bboxy linii. Prostsze niż CRAFT ale działa
bez ciężkich zależności ML (tylko opencv + numpy).

Dobre dla: dokumentów, skanów, tekstu drukowanego na białym tle.
Słabsze dla: tekstu w naturze (scene text), złożonych tła.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .preprocess import load_image, to_grayscale
from .result import BBox

if TYPE_CHECKING:
    from .config import OcrConfig

logger = logging.getLogger(__name__)


def _character_height(gray: np.ndarray) -> float:
    background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    )
    normalized = cv2.divide(gray, background, scale=255)
    binary = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    heights = [
        int(ch)
        for _, _, cw, ch, area in stats[1:]
        if area >= 10 and 12 <= ch < 80 and cw < 150
    ]
    return float(np.median(heights)) if heights else 20.0


def _line_bands(binary: np.ndarray, character_height: float) -> list[tuple[int, int]]:
    """Znajdź pasma linii metodą projekcji poziomej (ink profile per wiersz)."""
    rows = binary.sum(axis=1) / 255.0
    if rows.size == 0 or rows.max() <= 0:
        return []
    # Otsu na profilu projekcji: automatyczny podział wiersz-linia vs dolina
    scaled = np.clip(rows / rows.max() * 255.0, 0, 255).astype(np.uint8)
    otsu = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    threshold = max(1.0, otsu / 255.0 * rows.max())
    is_ink = rows > threshold
    # scal tylko bardzo drobne przerwy (1-2 px szumu), nie doliny między liniami
    min_gap = max(1, int(character_height * 0.15))
    bands: list[list[int]] = []
    row = 0
    n = len(is_ink)
    while row < n:
        if is_ink[row]:
            start = row
            while row < n and is_ink[row]:
                row += 1
            bands.append([start, row])
        else:
            row += 1
    merged: list[list[int]] = []
    for band in bands:
        if merged and band[0] - merged[-1][1] <= min_gap:
            merged[-1][1] = band[1]
        else:
            merged.append(band)
    min_height = max(1, int(character_height * 0.5))
    return [(a, b) for a, b in merged if b - a >= min_height]


def _split_block(gray: np.ndarray, bbox: BBox, character_height: float) -> list[BBox]:
    if bbox.height <= character_height * 2.0:
        return [bbox]
    crop = gray[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    background = cv2.morphologyEx(
        crop, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    )
    normalized = cv2.divide(crop, background, scale=255)
    binary = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    bands = _line_bands(binary, character_height)
    if len(bands) < 2:
        return [bbox]

    padding = max(1, int(character_height * 0.15))
    lines = []
    for top, bottom in bands:
        y1 = max(0, top - padding)
        y2 = min(crop.shape[0], bottom + padding)
        cols = np.where(binary[y1:y2].sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        x1 = max(0, int(cols[0]) - padding)
        x2 = min(crop.shape[1], int(cols[-1]) + 1 + padding)
        lines.append(BBox(bbox.x1 + x1, bbox.y1 + y1, bbox.x1 + x2, bbox.y1 + y2))
    return lines or [bbox]


def _dedup(boxes: list[BBox]) -> list[BBox]:
    """Usuń zduplikowane linie (ramka strony generuje te same linie co kontury).

    Zachowuje węższy (ciaśniejszy) box spośród silnie nakładających się.
    """
    kept: list[BBox] = []
    for box in sorted(boxes, key=lambda b: (b.y1, b.x1)):
        duplicate = False
        for index, other in enumerate(kept):
            oy = min(box.y2, other.y2) - max(box.y1, other.y1)
            ox = min(box.x2, other.x2) - max(box.x1, other.x1)
            if oy <= 0 or ox <= 0:
                continue
            vertical = oy / min(box.height, other.height)
            horizontal = ox / min(box.width, other.width)
            if vertical > 0.6 and horizontal > 0.4:
                if box.width * box.height < other.width * other.height:
                    kept[index] = box
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


class OpenCVDetector:
    """Detektor linii tekstu oparty na morfologii OpenCV."""

    def __init__(self, config: "OcrConfig") -> None:
        self.config = config

    def detect(self, image) -> list[BBox]:
        arr = load_image(image) if not isinstance(image, np.ndarray) else image
        gray = to_grayscale(arr)
        h, w = gray.shape

        # 1. Binaryzacja (tekst ciemny na jasnym tle)
        binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        # 2. Dylatacja pozioma — łączy znaki w linie
        # szerokść jądra proporcjonalna do szerokości obrazu
        kernel_w = max(30, w // 20)
        kernel_h = max(3, h // 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # 3. Kontury = linie tekstu
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bboxes: list[BBox] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # filtruj za małe regiony (szum)
            if cw * ch < self.config.min_line_area:
                continue
            # filtruj za wąskie (pojedyncze znaki, nie linie)
            if cw < 20:
                continue
            bboxes.append(BBox(int(x), int(y), int(x + cw), int(y + ch)))

        character_height = _character_height(gray)
        bboxes = [
            line
            for bbox in bboxes
            for line in _split_block(gray, bbox, character_height)
        ]
        bboxes = [
            bbox
            for bbox in bboxes
            if bbox.height >= character_height * 0.75
            and bbox.height <= h * 0.5
            and not (
                bbox.width < w * 0.15 and bbox.height < character_height * 1.5
            )
        ]
        bboxes = _dedup(bboxes)

        # sortuj w porządku czytania
        from .detector import sort_reading_order
        bboxes = sort_reading_order(bboxes, h)
        logger.debug("OpenCV detektor: znaleziono %d linii", len(bboxes))
        return bboxes

    def close(self) -> None:
        pass
