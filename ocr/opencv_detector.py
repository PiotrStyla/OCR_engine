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
            # filtruj za wysokie (prawdopodobnie cały obraz lub margines)
            if ch > h * 0.5:
                continue
            bboxes.append(BBox(int(x), int(y), int(x + cw), int(y + ch)))

        # sortuj w porządku czytania
        from .detector import sort_reading_order
        bboxes = sort_reading_order(bboxes, h)
        logger.debug("OpenCV detektor: znaleziono %d linii", len(bboxes))
        return bboxes

    def close(self) -> None:
        pass
