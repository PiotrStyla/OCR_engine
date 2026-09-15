"""Testy backendu Kraken — z zamockowanym kraken (bez ciężkich zależności ML).

Testy weryfikują:
- Rozwiązanie spec modelu (lokalny plik vs HF repo+plik)
- Routing: recognizer_backend='kraken' bypassuje TrOCR pipeline
- Konwersja wyniku Kraken na OcrResult
- Puste linie → pusty wynik
- close() zwalnia zasoby
"""

from __future__ import annotations

import sys
import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock

from ocr.config import OcrConfig
from ocr.result import BBox, OcrResult, TextLine


def test_kraken_model_spec_local_file(tmp_path):
    """Lokalny plik .mlmodel jest zwracany bezpośrednio."""
    from ocr.kraken_backend import _resolve_kraken_model
    model = tmp_path / "model.mlmodel"
    model.write_text("dummy")
    assert _resolve_kraken_model(str(model)) == str(model)


def test_kraken_model_spec_missing_raises():
    """Nieistniejący plik bez :: i bez / rzuca FileNotFoundError."""
    from ocr.kraken_backend import _resolve_kraken_model
    with pytest.raises(FileNotFoundError):
        _resolve_kraken_model("nonexistent.mlmodel")


def test_kraken_config_defaults():
    """Domyślna konfiguracja ma recognizer_backend='trocr' i kraken_model ustawiony."""
    cfg = OcrConfig()
    assert cfg.recognizer_backend == "trocr"
    assert "polish_nfd_9313" in cfg.kraken_model


def test_kraken_config_from_env():
    """OCR_KRAKEN_MODEL nadpisuje domyślny model."""
    import os
    old = os.environ.get("OCR_KRAKEN_MODEL")
    os.environ["OCR_KRAKEN_MODEL"] = "/custom/model.mlmodel"
    try:
        cfg = OcrConfig.from_env()
        assert cfg.kraken_model == "/custom/model.mlmodel"
    finally:
        if old is not None:
            os.environ["OCR_KRAKEN_MODEL"] = old
        else:
            os.environ.pop("OCR_KRAKEN_MODEL", None)


def _inject_mock_kraken():
    """Wstrzykuje mock moduły kraken do sys.modules (kraken nie jest zainstalowany)."""
    mock_kraken = MagicMock()
    mock_blla = MagicMock()
    mock_rpred = MagicMock()
    mock_models = MagicMock()
    mock_binarization = MagicMock()
    mock_lib = MagicMock()
    mock_lib.models = mock_models
    mock_kraken.blla = mock_blla
    mock_kraken.rpred = mock_rpred
    mock_kraken.binarization = mock_binarization
    mock_kraken.lib = mock_lib
    modules = {
        "kraken": mock_kraken,
        "kraken.blla": mock_blla,
        "kraken.rpred": mock_rpred,
        "kraken.binarization": mock_binarization,
        "kraken.lib": mock_lib,
        "kraken.lib.models": mock_models,
    }
    return patch.dict(sys.modules, modules)


def test_kraken_backend_recognize_with_mock(tmp_path):
    """KrakenBackend.recognize z zamockowanym kraken zwraca OcrResult."""
    from ocr.kraken_backend import KrakenBackend

    cfg = OcrConfig()
    cfg.kraken_model = str(tmp_path / "fake.mlmodel")
    backend = KrakenBackend(cfg)

    mock_seg = MagicMock()
    mock_seg.lines = [MagicMock()]
    mock_pred_record = MagicMock()
    mock_pred_record.prediction = "Test linia"
    mock_pred_record.boundary = [(10, 20), (100, 20), (100, 40), (10, 40)]
    mock_pred_record.baseline = [(10, 30), (100, 30)]
    mock_pred_record.confidences = [0.9, 0.8]
    mock_pred = MagicMock()
    mock_pred.__iter__ = MagicMock(return_value=iter([mock_pred_record]))

    with _inject_mock_kraken():
        import kraken.blla as blla_mod
        import kraken.rpred as rpred_mod
        import kraken.binarization as bin_mod
        blla_mod.segment = MagicMock(return_value=mock_seg)
        rpred_mod.rpred = MagicMock(return_value=mock_pred)
        bin_mod.nlbin = MagicMock(side_effect=lambda im: im)  # passthrough

        # Pre-set _recognizer so _ensure_loaded is a no-op (skip model file load).
        backend._recognizer = MagicMock()
        img = Image.new("RGB", (200, 100), "white")
        result = backend.recognize(img)

    assert isinstance(result, OcrResult)
    assert len(result.lines) == 1
    assert result.lines[0].text == "Test linia"
    assert result.lines[0].language == "pl"
    assert result.lines[0].bbox.x1 == 10
    assert result.lines[0].bbox.y1 == 20
    assert result.lines[0].bbox.x2 == 100
    assert result.lines[0].bbox.y2 == 40


def test_kraken_backend_empty_segmentation(tmp_path):
    """Pusta segmentacja → pusty OcrResult."""
    from ocr.kraken_backend import KrakenBackend

    cfg = OcrConfig()
    cfg.kraken_model = str(tmp_path / "fake.mlmodel")
    backend = KrakenBackend(cfg)

    mock_seg = MagicMock()
    mock_seg.lines = []

    with _inject_mock_kraken():
        import kraken.blla as blla_mod
        import kraken.rpred as rpred_mod
        import kraken.binarization as bin_mod
        blla_mod.segment = MagicMock(return_value=mock_seg)
        rpred_mod.rpred = MagicMock()
        bin_mod.nlbin = MagicMock(side_effect=lambda im: im)

        # Pre-set _recognizer so _ensure_loaded is a no-op (skip model file load).
        backend._recognizer = MagicMock()
        img = Image.new("RGB", (200, 100), "white")
        result = backend.recognize(img)

    assert isinstance(result, OcrResult)
    assert len(result.lines) == 0


def test_kraken_backend_close(tmp_path):
    """close() zwalnia model."""
    from ocr.kraken_backend import KrakenBackend

    cfg = OcrConfig()
    cfg.kraken_model = str(tmp_path / "fake.mlmodel")
    backend = KrakenBackend(cfg)
    backend._recognizer = MagicMock()
    backend.close()
    assert backend._recognizer is None


def test_pipeline_routes_to_kraken(tmp_path):
    """Pipeline z recognizer_backend='kraken' używa KrakenBackend, nie TrOCR."""
    from ocr.pipeline import OcrEngine
    from ocr.kraken_backend import KrakenBackend

    cfg = OcrConfig()
    cfg.recognizer_backend = "kraken"
    cfg.kraken_model = str(tmp_path / "fake.mlmodel")

    mock_result = OcrResult(
        lines=[TextLine(text="kraken linia", bbox=BBox(0, 0, 100, 20),
                       language="pl", confidence=float("nan"))],
        image_size=(200, 100),
    )

    with patch.object(KrakenBackend, "recognize", return_value=mock_result):
        engine = OcrEngine(cfg)
        img = Image.new("RGB", (200, 100), "white")
        result = engine.recognize(img)

    assert isinstance(result, OcrResult)
    assert len(result.lines) == 1
    assert result.lines[0].text == "kraken linia"
