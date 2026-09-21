"""Testy integracji Jev (TypeSafe AI) — mock API."""
from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

import numpy as np

from ocr.config import OcrConfig
from ocr.jev import JevClient, TYPESAFE_API_URL
from ocr.result import BBox, TextLine


def _mock_urlopen_response(mock_urlopen, payload: dict):
    """Helper: ustawia mock urlopen zwracający JSON payload."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_resp


class TestJevClient:
    """Testy klienta Jev API."""

    def test_disabled_without_key(self):
        client = JevClient(api_key="")
        assert not client.enabled
        assert client.ask("test", {}) is None
        assert client.score_quality("test") is None
        assert client.route_backend("test") is None
        assert client.validate_correction("a", "b") is None

    @patch("urllib.request.urlopen")
    def test_ask_success(self, mock_urlopen):
        _mock_urlopen_response(mock_urlopen, {
            "model": "jev-1.0",
            "answers": {"q": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        client = JevClient(api_key="test-key")
        result = client.ask("test state", {"q": {"type": "noul", "instructions": "test"}})
        assert result is not None
        assert result["answers"]["q"]["noul"] == 0.9
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_ask_http_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url=TYPESAFE_API_URL, code=401, msg="Unauthorized",
            hdrs={}, fp=None
        )
        client = JevClient(api_key="bad-key")
        assert client.ask("test", {}) is None

    @patch("urllib.request.urlopen")
    def test_score_quality(self, mock_urlopen):
        _mock_urlopen_response(mock_urlopen, {
            "model": "jev-1.0",
            "answers": {"quality": {"type": "score", "score": 1.8, "confidence": 0.95}},
        })
        client = JevClient(api_key="test-key")
        result = client.score_quality("good text")
        assert result == (1.8, 0.95)

    @patch("urllib.request.urlopen")
    def test_score_quality_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("network error")
        client = JevClient(api_key="test-key")
        assert client.score_quality("text") is None

    @patch("urllib.request.urlopen")
    def test_route_backend(self, mock_urlopen):
        _mock_urlopen_response(mock_urlopen, {
            "model": "jev-1.0",
            "answers": {"backend": {"type": "choice", "choice": "kraken", "confidence": 0.8}},
        })
        client = JevClient(api_key="test-key")
        assert client.route_backend("stats") == "kraken"

    @patch("urllib.request.urlopen")
    def test_route_backend_invalid(self, mock_urlopen):
        _mock_urlopen_response(mock_urlopen, {
            "model": "jev-1.0",
            "answers": {"backend": {"type": "choice", "choice": "invalid", "confidence": 0.8}},
        })
        client = JevClient(api_key="test-key")
        assert client.route_backend("stats") is None

    @patch("urllib.request.urlopen")
    def test_validate_correction_better(self, mock_urlopen):
        _mock_urlopen_response(mock_urlopen, {
            "model": "jev-1.0",
            "answers": {"better": {"type": "noul", "noul": 0.9}},
        })
        client = JevClient(api_key="test-key")
        assert client.validate_correction("raw", "corrected") is True

    @patch("urllib.request.urlopen")
    def test_validate_correction_worse(self, mock_urlopen):
        _mock_urlopen_response(mock_urlopen, {
            "model": "jev-1.0",
            "answers": {"better": {"type": "noul", "noul": 0.3}},
        })
        client = JevClient(api_key="test-key")
        assert client.validate_correction("raw", "corrected") is False


class TestJevIntegration:
    """Testy integracji Jev z pipeline."""

    def test_config_env(self):
        with patch.dict("os.environ", {
            "TYPESAFE_API_KEY": "key123",
            "OCR_JEV_ROUTE_BACKEND": "true",
            "OCR_JEV_SCORE_LINES": "false",
            "OCR_JEV_VALIDATE": "true",
            "OCR_JEV_QUALITY_THRESHOLD": "1.5",
        }):
            config = OcrConfig.from_env()
            assert config.jev_api_key == "key123"
            assert config.jev_route_backend is True
            assert config.jev_score_lines is False
            assert config.jev_validate_correction is True
            assert config.jev_quality_threshold == 1.5

    def test_config_defaults(self):
        config = OcrConfig()
        assert config.jev_api_key is None
        assert config.jev_model == "jev-latest"
        assert config.jev_route_backend is False
        assert config.jev_score_lines is False
        assert config.jev_validate_correction is False
        assert config.jev_quality_threshold == 1.0

    def test_textline_jev_score(self):
        line = TextLine(
            text="test", bbox=BBox(0, 0, 10, 10),
            language="en", confidence=0.9, jev_score=1.5,
        )
        assert line.jev_score == 1.5
        d = line.to_dict() if hasattr(line, 'to_dict') else None
        # jev_score in to_dict via result
        from ocr.result import OcrResult
        result = OcrResult(lines=[line])
        d = result.to_dict()
        assert d["lines"][0]["jev_score"] == 1.5

    def test_textline_jev_score_none(self):
        line = TextLine(
            text="test", bbox=BBox(0, 0, 10, 10),
            language="en", confidence=0.9,
        )
        assert line.jev_score is None
        from ocr.result import OcrResult
        result = OcrResult(lines=[line])
        d = result.to_dict()
        assert "jev_score" not in d["lines"][0]


class TestJevPipeline:
    """Testy pipeline z Jev — mock JevClient."""

    @patch("ocr.pipeline.load_image")
    def test_jev_route_backend_kraken(self, mock_load):
        from ocr.pipeline import OcrEngine
        mock_load.return_value = np.zeros((500, 300, 3), dtype=np.uint8)
        config = OcrConfig(
            recognizer_backend="auto",
            jev_route_backend=True,
            jev_api_key="test",
        )
        engine = OcrEngine(config)
        engine.jev = MagicMock()
        engine.jev.enabled = True
        engine.jev.route_backend.return_value = "kraken"
        engine.detector.detect = MagicMock(return_value=[])
        engine.recognizer.recognize_lines = MagicMock(return_value=[])
        engine._recognize_kraken = MagicMock(return_value="kraken_result")

        result = engine._recognize_auto("test.tif")
        assert result == "kraken_result"
        engine.jev.route_backend.assert_called_once()

    @patch("ocr.pipeline.load_image")
    def test_jev_route_backend_fallback(self, mock_load):
        from ocr.pipeline import OcrEngine
        mock_load.return_value = np.zeros((500, 300, 3), dtype=np.uint8)
        config = OcrConfig(
            recognizer_backend="auto",
            jev_route_backend=True,
            jev_api_key="test",
        )
        engine = OcrEngine(config)
        engine.jev = MagicMock()
        engine.jev.enabled = True
        engine.jev.route_backend.return_value = None
        engine.detector.detect = MagicMock(return_value=[])
        engine.recognizer.recognize_lines = MagicMock(return_value=[])
        engine._recognize_array = MagicMock(return_value="trocr_result")

        result = engine._recognize_auto("test.tif")
        assert result == "trocr_result"

    def test_jev_score_lines(self):
        from ocr.pipeline import OcrEngine
        config = OcrConfig(
            jev_score_lines=True,
            jev_api_key="test",
        )
        engine = OcrEngine(config)
        engine.jev = MagicMock()
        engine.jev.enabled = True
        engine.jev.score_quality.return_value = (1.5, 0.9)

        lines = [
            TextLine(text="line1", bbox=BBox(0, 0, 10, 10), language="en", confidence=0.9),
            TextLine(text="line2", bbox=BBox(0, 10, 10, 20), language="en", confidence=0.8),
        ]
        from ocr.result import OcrResult
        result = OcrResult(lines=lines)
        result = engine._jev_score_lines(result)
        assert result.lines[0].jev_score == 1.5
        assert result.lines[1].jev_score == 1.5

    def test_jev_validate_correction_revert(self):
        from ocr.pipeline import OcrEngine
        config = OcrConfig(
            correct_text=True,
            jev_validate_correction=True,
            jev_api_key="test",
            fabryka_api_key="test",
        )
        engine = OcrEngine(config)
        engine.jev = MagicMock()
        engine.jev.enabled = True
        engine.jev.validate_correction.return_value = False  # corrected is worse
        engine.corrector = MagicMock()
        engine.corrector.enabled = True
        engine.corrector.correct.return_value = "corrected text"

        lines = [TextLine(text="raw", bbox=BBox(0, 0, 10, 10), language="en", confidence=0.9)]
        from ocr.result import OcrResult
        result = OcrResult(lines=lines)
        result = engine._correct_result(result)
        # Jev said corrected is worse → revert to original
        assert result.lines[0].text == "raw"

    def test_jev_validate_correction_keep(self):
        from ocr.pipeline import OcrEngine
        config = OcrConfig(
            correct_text=True,
            jev_validate_correction=True,
            jev_api_key="test",
            fabryka_api_key="test",
        )
        engine = OcrEngine(config)
        engine.jev = MagicMock()
        engine.jev.enabled = True
        engine.jev.validate_correction.return_value = True  # corrected is better
        engine.corrector = MagicMock()
        engine.corrector.enabled = True
        engine.corrector.correct.return_value = "corrected text"

        lines = [TextLine(text="raw", bbox=BBox(0, 0, 10, 10), language="en", confidence=0.9)]
        from ocr.result import OcrResult
        result = OcrResult(lines=lines)
        result = engine._correct_result(result)
        # Jev said corrected is better → keep corrected
        assert result.lines[0].text == "corrected text"
        assert result.lines[0].raw_text == "raw"
