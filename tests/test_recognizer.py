"""Testy wyboru backendu rozpoznawania (trocr / paddlevl) — bez ładowania modeli."""

from __future__ import annotations

import numpy as np

from ocr.config import OcrConfig
from ocr.recognizer import Recognizer, _PaddleVLBackend, _TrOCRBackend
from ocr.result import BBox


class _FakeVL:
    def __init__(self, model_name, device):
        self.model_name = model_name

    def recognize(self, images, batch_size):
        return [("tekst vlm", 0.9) for _ in images]


def test_paddlevl_backend_selected(monkeypatch):
    """recognizer_backend='paddlevl' → jeden backend VLM dla wszystkich języków."""
    monkeypatch.setattr("ocr.recognizer._PaddleVLBackend", _FakeVL)
    rec = Recognizer(OcrConfig(recognizer_backend="paddlevl", device="cpu"))

    arr = np.zeros((60, 200, 3), dtype=np.uint8)
    out = rec.recognize_lines(arr, [BBox(0, 0, 100, 20), BBox(0, 30, 100, 50)],
                              ["pl", "en"])
    assert [t for t, _ in out] == ["tekst vlm", "tekst vlm"]
    # jeden wspólny backend, załadowany raz
    assert list(rec._backends) == ["paddlevl"]
    assert isinstance(rec._backends["paddlevl"], _FakeVL)


def test_paddlevl_has_model_for_any_language(monkeypatch):
    """PaddleOCR-VL obsługuje 109 języków — wystarczy zainstalowany pakiet."""
    import sys
    import types
    monkeypatch.setitem(sys.modules, "paddleocr", types.ModuleType("paddleocr"))
    rec = Recognizer(OcrConfig(recognizer_backend="paddlevl", device="cpu"))
    assert rec.has_model_for("pl")
    assert rec.has_model_for("en")
    assert rec.has_model_for("unknown")


def test_trocr_has_model_for_pl_missing():
    # ścieżka lokalna bez '/' → od razu False, bez sieci
    rec = Recognizer(OcrConfig(device="cpu", recognizer_pl="nieistniejacy_katalog"))
    assert not rec.has_model_for("pl")


def test_model_exists_hf_check(monkeypatch):
    """Repo HF weryfikowane przez model_info; 404/brak sieci → False."""
    from ocr import recognizer as r

    # symulacja braku sieci: model_info rzuca
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "model_info",
        lambda *a, **k: (_ for _ in ()).throw(Exception("offline")),
    )
    assert r._model_exists("org/nieistniejacy-model") is False
    assert r._model_exists("lokalna_sciezka_bez_slasha") is False


def test_config_backend_from_env(monkeypatch):
    monkeypatch.setenv("OCR_RECOGNIZER_BACKEND", "paddlevl")
    monkeypatch.setenv("OCR_PADDLEVL_MODEL", "custom/vlm")
    cfg = OcrConfig.from_env()
    assert cfg.recognizer_backend == "paddlevl"
    assert cfg.paddlevl_model == "custom/vlm"


def test_trocr_remains_default():
    cfg = OcrConfig()
    assert cfg.recognizer_backend == "trocr"
    rec = Recognizer(cfg)
    assert rec._backends == {}


def test_paddlevl_backend_class_exists():
    assert hasattr(_PaddleVLBackend, "recognize")
    assert hasattr(_TrOCRBackend, "recognize")
