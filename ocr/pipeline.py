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
        return self._recognize_array(arr)

    def recognize_pdf(
        self, pdf: ImageLike, pages: str | None = None, dpi: int = 300
    ) -> list[OcrResult]:
        """Rozpoznaje strony PDF. `pages`: zakres '1-3,5' (None = wszystkie)."""
        from .preprocess import iter_pdf_pages

        results: list[OcrResult] = []
        for _page_no, arr in iter_pdf_pages(pdf, dpi=dpi, pages=pages):
            if self.config.deskew:
                arr = deskew(arr)
            results.append(self._recognize_array(arr))
        return results

    def _recognize_array(self, arr: np.ndarray) -> OcrResult:
        """Właściwy pipeline na array RGB (po preprocessingu)."""
        bboxes = self.detector.detect(arr)
        if not bboxes:
            return OcrResult(lines=[], image_size=(arr.shape[1], arr.shape[0]))

        languages = self._route_languages(bboxes)
        decoded = self.recognizer.recognize_lines(arr, bboxes, languages)

        lines: list[TextLine] = []
        redo: list[tuple[int, BBox, Language]] = []  # (idx w lines, bbox, wykryty język)
        for bbox, routed_lang, (text, conf) in zip(bboxes, languages, decoded):
            if not text:
                continue
            lang = routed_lang
            # Bez wymuszonego języka aktualizujemy etykietę z rozpoznanego tekstu
            if self.config.force_language is None:
                detected = detect_language(text)
                if detected != "unknown":
                    lang = detected
            lines.append(TextLine(text=text, bbox=bbox, language=lang, confidence=conf))
            if lang != routed_lang and lang in ("pl", "en"):
                redo.append((len(lines) - 1, bbox, lang))

        # Drugi przebieg: linie rozpoznane niewłaściwym modelem → re-rozpoznanie
        # właściwym. Tylko gdy model docelowego języka istnieje (fallback EN
        # dałby ten sam wynik — strata czasu).
        if redo:
            redo = [r for r in redo if self.recognizer.has_model_for(r[2])]
        if redo:
            logger.debug("Drugi przebieg: %d linii do re-rozpoznania", len(redo))
            re_decoded = self.recognizer.recognize_lines(
                arr, [b for _, b, _ in redo], [lang for _, _, lang in redo]
            )
            for (li, _, _), (text, conf) in zip(redo, re_decoded):
                if text:
                    lines[li] = TextLine(
                        text=text, bbox=lines[li].bbox,
                        language=lines[li].language, confidence=conf,
                    )

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
            n = len(result.lines)
            new_lines = [
                TextLine(
                    text=cl,
                    bbox=result.lines[min(i, n - 1)].bbox,
                    language=result.lines[min(i, n - 1)].language,
                    confidence=result.lines[min(i, n - 1)].confidence,
                )
                for i, cl in enumerate(corrected_lines)
            ]
        return OcrResult(lines=new_lines, image_size=result.image_size)

    def _route_languages(self, bboxes: list[BBox]) -> list[Language]:
        """Routing języka: jeśli wymuszono język → ten język,
        w przeciwnym razie 'en' domyślnie. Detekcja języka z tekstu
        następuje po rozpoznaniu (etykieta w wyniku), bez ponownej inferencji."""
        force = self.config.force_language
        if force in ("pl", "en"):
            return [force] * len(bboxes)  # type: ignore[list-item]
        return ["en"] * len(bboxes)

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
