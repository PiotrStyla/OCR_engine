"""Testy pipeline'u z zamockowanym detektorem i rozpoznawaczem (bez modeli ML)."""

from __future__ import annotations

import numpy as np
import pytest

from ocr.config import OcrConfig
from ocr.pipeline import OcrEngine
from ocr.result import BBox, OcrResult, TextLine


class _FakeDetector:
    def __init__(self, config):
        self.config = config

    def detect(self, image):
        return [BBox(0, 0, 100, 20), BBox(0, 30, 100, 50)]

    def close(self):
        pass


class _FakeRecognizer:
    def __init__(self, config):
        self.config = config
        self.pl_available = False
        self.calls: list[list[str]] = []
        self.responses: list[list[tuple[str, float]]] = []

    def has_model_for(self, language):
        return self.pl_available if language == "pl" else True

    def recognize_lines(self, image, bboxes, languages):
        self.calls.append(list(languages))
        if self.responses:
            return self.responses.pop(0)
        return [(f"linia {i + 1}", 0.9) for i in range(len(bboxes))]

    def close(self):
        pass


class _FakeCorrector:
    """Mock korektora — zwraca tekst z dodanym prefixem 'POPRAWIONO:'."""

    def __init__(self, config):
        self.config = config
        self.enabled = True
        self.calls = []

    def correct(self, text):
        self.calls.append(text)
        return "POPRAWIONO:" + text

    def close(self):
        pass


def _make_engine(monkeypatch, config=None):
    engine = OcrEngine(config or OcrConfig())
    engine.detector = _FakeDetector(engine.config)
    engine.recognizer = _FakeRecognizer(engine.config)
    engine.corrector = _FakeCorrector(engine.config)
    return engine


def test_pipeline_recognize_returns_lines(monkeypatch):
    engine = _make_engine(monkeypatch)
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:] = 255
    result = engine.recognize(arr)
    assert isinstance(result, OcrResult)
    assert len(result.lines) == 2
    assert result.lines[0].text == "linia 1"
    assert result.lines[1].text == "linia 2"
    assert result.text == "linia 1\nlinia 2"


def test_pipeline_force_language_applied(monkeypatch):
    engine = _make_engine(monkeypatch, OcrConfig(force_language="pl", deskew=False))
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:] = 255
    result = engine.recognize(arr)
    assert all(line.language == "pl" for line in result.lines)


def test_pipeline_empty_detection(monkeypatch):
    engine = _make_engine(monkeypatch)
    engine.detector = _FakeDetector(engine.config)
    engine.detector.detect = lambda image: []  # type: ignore[assignment]
    arr = np.zeros((50, 50, 3), dtype=np.uint8)
    arr[:] = 255
    result = engine.recognize(arr)
    assert result.lines == []
    assert result.image_size == (50, 50)


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("OCR_FORCE_LANGUAGE", "pl")
    monkeypatch.setenv("OCR_RECOGNIZER_EN", "custom/en")
    monkeypatch.setenv("OCR_CORRECT_TEXT", "true")
    monkeypatch.setenv("FABRYKA_API_KEY", "sk-test")
    monkeypatch.setenv("FABRYKA_MODEL", "bielik-11b-v3")
    cfg = OcrConfig.from_env()
    assert cfg.force_language == "pl"
    assert cfg.recognizer_en == "custom/en"
    assert cfg.correct_text is True
    assert cfg.fabryka_api_key == "sk-test"
    assert cfg.fabryka_model == "bielik-11b-v3"


def test_pipeline_correction_applied(monkeypatch):
    config = OcrConfig(correct_text=True, fabryka_api_key="test", deskew=False)
    engine = _make_engine(monkeypatch, config)
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:] = 255
    result = engine.recognize(arr)
    # korektor mock dodaje "POPRAWIONO:" przed pełnym tekstem (jedna linia)
    assert result.text.startswith("POPRAWIONO:")
    assert "linia 1" in result.text


def test_pipeline_correction_skipped_when_disabled(monkeypatch):
    config = OcrConfig(correct_text=False, deskew=False)
    engine = _make_engine(monkeypatch, config)
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:] = 255
    result = engine.recognize(arr)
    # bez korekty tekst oryginalny
    assert result.text == "linia 1\nlinia 2"
    assert engine.corrector.calls == []


def test_pipeline_second_pass_pl(monkeypatch):
    """Linie wykryte jako PL po pierwszym przebiegu są re-rozpoznawane modelem PL."""
    engine = _make_engine(monkeypatch, OcrConfig(deskew=False))
    engine.recognizer.pl_available = True
    # 1. wywołanie (EN routing): zwraca polski tekst → detect_language → "pl"
    # 2. wywołanie (PL re-rozpoznanie): lepszy tekst
    engine.recognizer.responses = [
        [("Zażółć gęślą jaźń", 0.7), ("hello world", 0.8)],
        [("Zażółć gęślą jaźń — PL", 0.95)],
    ]
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:] = 255
    result = engine.recognize(arr)

    assert len(engine.recognizer.calls) == 2
    assert engine.recognizer.calls[1] == ["pl"]  # tylko linia PL re-rozpoznana
    assert result.lines[0].text == "Zażółć gęślą jaźń — PL"
    assert result.lines[0].language == "pl"
    assert result.lines[0].confidence == 0.95
    assert result.lines[1].text == "hello world"


def test_pipeline_second_pass_skipped_without_pl_model(monkeypatch):
    """Brak modelu PL → drugi przebieg pomijany (fallback EN dałby to samo)."""
    engine = _make_engine(monkeypatch, OcrConfig(deskew=False))
    engine.recognizer.pl_available = False
    engine.recognizer.responses = [
        [("Zażółć gęślą jaźń", 0.7), ("hello world", 0.8)],
    ]
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:] = 255
    result = engine.recognize(arr)

    assert len(engine.recognizer.calls) == 1  # tylko pierwszy przebieg
    assert result.lines[0].language == "pl"   # etykieta zaktualizowana
    assert result.lines[0].text == "Zażółć gęślą jaźń"  # tekst z EN


def test_pipeline_second_pass_skipped_with_force_language(monkeypatch):
    """force_language → brak detekcji języka, brak drugiego przebiegu."""
    engine = _make_engine(monkeypatch, OcrConfig(force_language="en", deskew=False))
    engine.recognizer.pl_available = True
    engine.recognizer.responses = [[("Zażółć gęślą jaźń", 0.7), ("x", 0.8)]]
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:] = 255
    result = engine.recognize(arr)

    assert len(engine.recognizer.calls) == 1
    assert all(ln.language == "en" for ln in result.lines)


class _EchoCorrector:
    """Mock korektora — zwraca wejście z dopiskiem '!' na końcu każdej linii."""

    def __init__(self, config):
        self.config = config
        self.enabled = True
        self.calls: list[str] = []

    def correct(self, text):
        self.calls.append(text)
        return "\n".join(line + "!" for line in text.split("\n"))

    def close(self):
        pass


def test_correction_only_low_confidence_lines(monkeypatch):
    """correct_low_confidence_only: do API trafiają tylko linie poniżej progu."""
    config = OcrConfig(
        correct_text=True, deskew=False,
        confidence_threshold=0.8, correct_low_confidence_only=True,
    )
    engine = _make_engine(monkeypatch, config)
    engine.corrector = _EchoCorrector(config)
    engine.recognizer.responses = [[("pewna linia", 0.95), ("niepewna", 0.3)]]

    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    result = engine.recognize(arr)

    assert engine.corrector.calls == ["niepewna"]  # tylko niska pewność
    assert result.lines[0].text == "pewna linia"   # nietknięta
    assert result.lines[1].text == "niepewna!"     # poprawiona


def test_correction_skipped_when_all_confident(monkeypatch):
    """Wszystkie linie powyżej progu → zero wywołań API."""
    config = OcrConfig(
        correct_text=True, deskew=False,
        confidence_threshold=0.8, correct_low_confidence_only=True,
    )
    engine = _make_engine(monkeypatch, config)
    engine.recognizer.responses = [[("dobry tekst", 0.9), ("tez dobry", 0.85)]]

    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    result = engine.recognize(arr)

    assert engine.corrector.calls == []
    assert result.text == "dobry tekst\ntez dobry"


def test_correction_selective_mismatch_keeps_original(monkeypatch):
    """Selektywna korekta ze zmienioną liczbą linii → oryginał bez zmian."""

    class _SplittingCorrector(_EchoCorrector):
        def correct(self, text):
            self.calls.append(text)
            return "a\nb\nc"  # 3 linie zamiast 1

    config = OcrConfig(
        correct_text=True, deskew=False,
        confidence_threshold=0.8, correct_low_confidence_only=True,
    )
    engine = _make_engine(monkeypatch, config)
    engine.corrector = _SplittingCorrector(config)
    engine.recognizer.responses = [[("pewna", 0.9), ("niepewna", 0.3)]]

    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    result = engine.recognize(arr)

    assert result.text == "pewna\nniepewna"  # oryginał zachowany


def test_low_confidence_flag_in_json(monkeypatch):
    """to_dict z progiem oznacza linie niskopewne flagą low_confidence."""
    engine = _make_engine(monkeypatch, OcrConfig(deskew=False))
    engine.recognizer.responses = [[("pewna", 0.9), ("niepewna", 0.3)]]

    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    result = engine.recognize(arr)
    d = result.to_dict(confidence_threshold=0.5)

    assert "low_confidence" not in d["lines"][0]
    assert d["lines"][1]["low_confidence"] is True
