"""Główny pipeline OCR: detekcja → routing języka → rozpoznawanie → porządek czytania.

Punkt wejścia wysokopoziomowy: `OcrEngine.recognize(image) -> OcrResult`.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Union

import numpy as np

from .config import OcrConfig
from .detector import TextDetector
from .jev import JevClient
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
        self.jev = JevClient(
            api_key=self.config.jev_api_key or "",
            model=self.config.jev_model,
            timeout=self.config.jev_timeout,
        )

    def recognize(self, image: ImageLike) -> OcrResult:
        """Rozpoznaje tekst z obrazu (ścieżka lub array RGB)."""
        # Kraken backend: end-to-end (segmentacja + rozpoznawanie), bypass TrOCR.
        if self.config.recognizer_backend == "kraken":
            return self._recognize_kraken(image)
        # Auto-routing: heurystyka OpenCV detection → Kraken dla maszynopisów.
        if self.config.recognizer_backend == "auto":
            return self._recognize_auto(image)
        arr = load_image(image)
        if self.config.deskew:
            arr, transform = deskew(arr, return_transform=True)
            return self._source_coordinates(self._recognize_array(arr), transform)
        return self._recognize_array(arr)

    def _recognize_auto(self, image: ImageLike) -> OcrResult:
        """Auto-routing: Jev (jeśli włączony) lub heurystyka OpenCV."""
        arr = load_image(image)
        bboxes = self.detector.detect(arr)
        h = arr.shape[0]

        # Jev routing (opcjonalny, wymaga TYPESAFE_API_KEY)
        if self.config.jev_route_backend and self.jev.enabled:
            backend = self._jev_route_backend(arr, bboxes)
            if backend == "kraken":
                logger.info("Jev routing → Kraken")
                return self._recognize_kraken(image)
            elif backend == "paddlevl":
                logger.info("Jev routing → PaddleOCR-VL (niezaimplementowany, używam TrOCR)")
                # PaddleVL nie jest jeszcze w pełni zintegrowany — fallback TrOCR
                return self._recognize_array(arr, bboxes)
            else:
                logger.info("Jev routing → TrOCR")
                return self._recognize_array(arr, bboxes)

        # Heurystyka: mało linii na dużej stronie → maszynopis → Kraken
        if (h >= self.config.auto_kraken_min_height
                and len(bboxes) < self.config.auto_kraken_min_lines):
            logger.info(
                "Auto-routing → Kraken (OpenCV wykrył %d linii na obrazie %dpx)",
                len(bboxes), h,
            )
            return self._recognize_kraken(image)
        logger.info(
            "Auto-routing → TrOCR (OpenCV wykrył %d linii na obrazie %dpx)",
            len(bboxes), h,
        )
        # Kontynuuj normalny pipeline TrOCR z już wykrytymi bboxami
        if self.config.deskew:
            arr, transform = deskew(arr, return_transform=True)
            # Re-detekcja po deskew (współrzędne się zmieniają)
            bboxes = self.detector.detect(arr)
            result = self._recognize_array(arr, bboxes)
            return self._source_coordinates(result, transform)
        return self._recognize_array(arr, bboxes)

    def _jev_route_backend(self, arr: np.ndarray, bboxes: list[BBox]) -> str:
        """Używa Jev do wyboru backendu OCR na podstawie statystyk + preview.

        Strategia: OCR pierwszych 3 linii przez TrOCR (szybki preview), potem
        Jev ocenia jakość + statystyki dokumentu → wybiera backend.
        """
        # Statystyki dokumentu jako tekst dla Jev
        w, h = arr.shape[1], arr.shape[0]
        stats_parts = [
            f"Document: {w}x{h}px, {len(bboxes)} text regions",
            f"avg region height: {np.mean([b.height for b in bboxes]):.0f}px" if bboxes else "no regions",
        ]
        # Preview OCR: pierwsze 3 linie przez TrOCR (tanio)
        if bboxes:
            preview = self.recognizer.recognize_lines(arr, bboxes[:3])
            sample = " | ".join(t for t, _ in preview if t)[:300]
            stats_parts.append(f"OCR preview: {sample}")
        stats_text = ". ".join(stats_parts)

        backend = self.jev.route_backend(stats_text)
        if backend is None:
            logger.warning("Jev routing failed — fallback to TrOCR")
            return "trocr"
        logger.info("Jev backend routing: %s (stats: %s)", backend, stats_text[:100])
        return backend

    def _recognize_kraken(self, image: ImageLike) -> OcrResult:
        """End-to-end OCR przez Kraken (segmentacja baseline + .mlmodel)."""
        from .kraken_backend import KrakenBackend
        if not hasattr(self, '_kraken_backend'):
            self._kraken_backend = KrakenBackend(self.config)
        result = self._kraken_backend.recognize(image)
        if self.config.correct_text:
            result = self._correct_result(result)
        return result

    @staticmethod
    def _source_coordinates(result, transform):
        import cv2
        inverse = cv2.invertAffineTransform(transform)
        lines = []
        w, h = result.image_size
        for line in result.lines:
            b = line.bbox
            corners = np.array([[b.x1,b.y1,1], [b.x2,b.y1,1],
                                [b.x2,b.y2,1], [b.x1,b.y2,1]], dtype=float)
            points = corners @ inverse.T
            points[:, 0] = np.clip(points[:, 0], 0, w)
            points[:, 1] = np.clip(points[:, 1], 0, h)
            box = BBox(int(np.floor(points[:,0].min())), int(np.floor(points[:,1].min())),
                       int(np.ceil(points[:,0].max())), int(np.ceil(points[:,1].max())))
            lines.append(replace(line, bbox=box, source_polygon=tuple(map(tuple, points.tolist()))))
        return replace(result, lines=lines, source_to_processed=transform.tolist())

    def recognize_pdf(
        self, pdf: ImageLike, pages: str | None = None, dpi: int = 300
    ) -> list[OcrResult]:
        """Rozpoznaje strony PDF. `pages`: zakres '1-3,5' (None = wszystkie)."""
        from .preprocess import iter_pdf_pages

        results: list[OcrResult] = []
        for page_no, arr in iter_pdf_pages(pdf, dpi=dpi, pages=pages):
            results.append(replace(self.recognize(arr), page_number=page_no))
        return results

    def _recognize_array(self, arr: np.ndarray, bboxes: list[BBox] | None = None) -> OcrResult:
        """Właściwy pipeline na array RGB (po preprocessingu)."""
        if bboxes is None:
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
            if (self.config.recognizer_backend == "trocr"
                    and lang != routed_lang and lang in ("pl", "en")):
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
        # Jev: scoring jakości linii OCR (opcjonalny)
        if self.config.jev_score_lines and self.jev.enabled and result.lines:
            result = self._jev_score_lines(result)
        if self.config.correct_text:
            result = self._correct_result(result)
        return result

    def _jev_score_lines(self, result: OcrResult) -> OcrResult:
        """Ocenia jakość każdej linii OCR przez Jev (Score 0-2).
        Linie ze score < jev_quality_threshold dostają jev_score i są
        flagowane do korekty. Jedno zapytanie Jev na całą stronę (oszczędność)."""
        state = "\n".join(l.text for l in result.lines)
        score = self.jev.score_quality(state)
        if score is None:
            return result
        val, conf = score
        logger.info("Jev page quality: %.2f (confidence: %.2f)", val, conf)
        new_lines = []
        for line in result.lines:
            # Przypisz page-level score do każdej linii (Jev nie robi per-line)
            new_lines.append(replace(line, jev_score=val))
        return replace(result, lines=new_lines)

    def _correct_result(self, result: OcrResult) -> OcrResult:
        """Stosuje korektę tekstu przez Fabryka API (jeśli włączona i dostępna).

        Gdy `correct_low_confidence_only` i `confidence_threshold` > 0, do API
        trafiają wyłącznie linie z confidence < próg — oszczędność tokenów.
        """
        if not result.lines or not self.corrector.enabled:
            return result
        selective = (
            self.config.correct_low_confidence_only
            and self.config.confidence_threshold > 0
        )
        if selective:
            idx = [i for i, ln in enumerate(result.lines)
                   if ln.confidence < self.config.confidence_threshold]
            if not idx:
                return result
        else:
            idx = list(range(len(result.lines)))

        joined = "\n".join(result.lines[i].text for i in idx)
        if not joined.strip():
            return result
        corrected = self.corrector.correct(joined)
        if not corrected or corrected == joined:
            return result

        # Jev: walidacja korekty — czy poprawiony tekst jest lepszy?
        if self.config.jev_validate_correction and self.jev.enabled:
            better = self.jev.validate_correction(joined, corrected)
            if better is False:
                logger.info("Jev walidacja: poprawiony tekst nie jest lepszy — zachowuję oryginał")
                return result
            elif better is True:
                logger.info("Jev walidacja: poprawiony tekst jest lepszy — zastosowuję korektę")
            # better is None → Jev niedostępny, zachowaj oryginalną logikę (zastosuj korektę)

        corrected_lines = corrected.split("\n")

        if len(corrected_lines) == len(idx):
            new_lines = list(result.lines)
            for i, cl in zip(idx, corrected_lines):
                ln = result.lines[i]
                new_lines[i] = replace(
                    ln, text=cl, raw_text=ln.raw_text if ln.raw_text is not None else ln.text,
                    raw_confidence=ln.raw_confidence if ln.raw_confidence is not None else ln.confidence,
                    confidence=ln.confidence if cl == ln.text else float("nan"),
                )
        else:
            # Podzbiór: przy zmienionej liczbie linii nie da się bezpiecznie
            # dopasować poprawek do oryginałów — zachowujemy oryginał.
            logger.warning(
                "Korekta zmieniła liczbę linii (%d → %d) — pomijam korektę.",
                len(idx), len(corrected_lines),
            )
            return result
        return replace(result, lines=new_lines)

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
        self.jev.close()
        if hasattr(self, '_kraken_backend'):
            self._kraken_backend.close()

    def __enter__(self) -> "OcrEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def recognize(image: ImageLike, config: OcrConfig | None = None) -> OcrResult:
    """Funkcja wygodnego użycia: jednorazowe rozpoznanie."""
    with OcrEngine(config) as engine:
        return engine.recognize(image)
