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

    def recognize_lines(self, image, bboxes, languages):
        return [("linia 1", 0.9), ("linia 2", 0.8)]

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
