"""Główny pipeline OCR: detekcja → routing języka → rozpoznawanie → porządek czytania.

Punkt wejścia wysokopoziomowy: `OcrEngine.recognize(image) -> OcrResult`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from .config import OcrConfig
from .detector import TextDetector
from .lang import detect_language
from .postprocess import TextCorrector
from .preprocess import deskew, load_image
from .recognizer import Recognizer
from .result import BBox, Language, OcrResult, TextLine

logger = logging.getLogger(__name__)

ImageLike = Union[str, Path, np.ndarray]


class OcrEngine:
    """Wysokopoziomowy silnik OCR."""

    def __init__(self, config: OcrConfig | None = None) -> None:
        self.config = config or OcrConfig.from_env()
        self.detector = TextDetector(self.config)
        self.recognizer = Recognizer(self.config)
        self.corrector = TextCorrector(self.config)

    def recognize(self, image: ImageLike) -> OcrResult:
        """Rozpoznaje tekst z obrazu (ścieżka lub array RGB)."""
        arr = load_image(image)
        if self.config.deskew:
            arr = deskew(arr)

        bboxes = self.detector.detect(arr)
        if not bboxes:
            return OcrResult(lines=[], image_size=(arr.shape[1], arr.shape[0]))

        languages = self._route_languages(bboxes)
        decoded = self.recognizer.recognize_lines(arr, bboxes, languages)

        lines: list[TextLine] = []
        for bbox, lang, (text, conf) in zip(bboxes, languages, decoded):
            if not text:
                continue
            lines.append(TextLine(text=text, bbox=bbox, language=lang, confidence=conf))

        result = OcrResult(lines=lines, image_size=(arr.shape[1], arr.shape[0]))
        if self.config.correct_text:
            result = self._correct_result(result)
        return result

    def _correct_result(self, result: OcrResult) -> OcrResult:
        """Stosuje korektę tekstu przez Fabryka API (jeśli włączona i dostępna)."""
        if not result.lines or not self.corrector.enabled:
            return result
        full_text = result.text
        if not full_text.strip():
            return result
        corrected = self.corrector.correct(full_text)
        if not corrected or corrected == full_text:
            return result
        # Rozdziel poprawiony tekst z powrotem na linie (zachowaj liczbę linii).
        corrected_lines = corrected.split("\n")
        # Jeśli liczba linii się zgadza — nadpisz teksty; w przeciwnym razie
        # zostaw jeden "poprawiony" blok z oryginalnymi bboxami pierwszej linii.
        if len(corrected_lines) == len(result.lines):
            new_lines = [
                TextLine(text=cl, bbox=ln.bbox, language=ln.language, confidence=ln.confidence)
                for ln, cl in zip(result.lines, corrected_lines)
            ]
        else:
            logger.warning(
                "Korekta zmieniła liczbę linii (%d → %d) — zachowuję oryginalne bboxy.",
                len(result.lines), len(corrected_lines),
            )
            new_lines = [
                TextLine(text=cl, bbox=result.lines[min(i, len(result.lines) - 1)].bbox,
                         language=result.lines[0].language, confidence=result.lines[0].confidence)
                for i, cl in enumerate(corrected_lines)
            ]
        return OcrResult(lines=new_lines, image_size=result.image_size)

    def _route_languages(self, bboxes: list[BBox]) -> list[Language]:
        """Bezpoznawaniowy routing: jeśli wymuszono język → ten język,
        w przeciwnym razie 'en' domyślnie (detekcja języka następuje po rozpoznaniu
        w drugim przebiegu — tu uproszczenie: używamy force lub 'en')."""
        force = self.config.force_language
        if force in ("pl", "en"):
            return [force] * len(bboxes)  # type: ignore[list-item]
        # Bez wymuszenia domyślnie EN; po rozpoznaniu można by przeanalizować
        # i powtórzyć dla PL — zostawiamy jako 'en' (patrz refine_languages).
        return ["en"] * len(bboxes)

    def refine_languages(self, result: OcrResult) -> OcrResult:
        """Drugie przejście: detekcja języka na rozpoznanym tekście i ponowne
        rozpoznanie linii, których język różni się od użytego."""
        if self.config.force_language is not None:
            return result
        if not result.lines:
            return result

        new_lines: list[TextLine] = []
        changed = False
        for line in result.lines:
            detected = detect_language(line.text)
            if detected != line.language and detected in ("pl", "en"):
                changed = True
            new_lines.append(
                TextLine(text=line.text, bbox=line.bbox, language=detected, confidence=line.confidence)
            )
        if not changed:
            return result

        # Ponowne rozpoznanie linii, których język się zmienił
        arr = None  # leniwe
        re_idx = [
            i for i, ln in enumerate(new_lines)
            if ln.language != result.lines[i].language
        ]
        if not re_idx:
            return OcrResult(lines=new_lines, image_size=result.image_size)

        logger.debug("Refine: ponowne rozpoznanie %d linii", len(re_idx))
        # Wymaga oryginalnego obrazu — tu uproszczenie: zostawiamy tekst,
        # tylko aktualizujemy etykiety języka. Pełna ponowna inferencja
        # wymagałaby zachowania obrazu (patrz recognize(keep_image=True)).
        return OcrResult(lines=new_lines, image_size=result.image_size)

    def close(self) -> None:
        self.detector.close()
        self.recognizer.close()
        self.corrector.close()

    def __enter__(self) -> "OcrEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def recognize(image: ImageLike, config: OcrConfig | None = None) -> OcrResult:
    """Funkcja wygodnego użycia: jednorazowe rozpoznanie."""
    with OcrEngine(config) as engine:
        return engine.recognize(image)
