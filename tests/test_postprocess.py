"""Testy postprocessingu (korekta tekstu przez Fabryka API) z mockami."""

from __future__ import annotations

import pytest

from ocr.config import OcrConfig
from ocr.postprocess import TextCorrector, _detect_base_url


class _FakeCompletions:
    def __init__(self, response_text: str = "ok", model_ids: list[str] | None = None):
        self.response_text = response_text
        self.model_ids = model_ids or ["bielik-11b-v3", "auto"]
        self.calls = []
        self.finish_reason = "stop"

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Msg:
            content = self.response_text

        class _Choice:
            message = _Msg()
            finish_reason = self.finish_reason

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeModels:
    def __init__(self, model_ids: list[str]):
        self.model_ids = model_ids

        class _M:
            def __init__(self, mid):
                self.id = mid

        self.data = [_M(mid) for mid in model_ids]

    def list(self):
        return self


class _FakeClient:
    def __init__(self, response_text: str = "ok", model_ids: list[str] | None = None):
        self.chat = type("C", (), {"completions": _FakeCompletions(response_text, model_ids)})()
        self.models = _FakeModels(model_ids or ["bielik-11b-v3", "auto"])


def _make_corrector(
    response_text: str = "ok",
    api_key: str = "test-key",
    correct: bool = True,
    model: str = "bielik-11b-v3",
    model_ids: list[str] | None = None,
) -> TextCorrector:
    config = OcrConfig(
        correct_text=correct,
        fabryka_api_key=api_key,
        fabryka_model=model,
    )
    corrector = TextCorrector(config)
    corrector._client = _FakeClient(response_text, model_ids)
    return corrector


# --- auto-detekcja endpointu ---


def test_detect_base_url_sk_fab_goes_to_router():
    assert _detect_base_url("sk-fab-123456", "auto") == "https://router.fabryka.ai/v1"


def test_detect_base_url_fab_live_goes_to_lab():
    assert _detect_base_url("fab_live_xyz", "auto") == "https://fabryka.ai/v1"


def test_detect_base_url_explicit_url_respected():
    assert _detect_base_url("sk-fab-123", "https://custom.example.com/v1") == "https://custom.example.com/v1"


def test_corrector_base_url_auto_detects_from_key():
    c = TextCorrector(OcrConfig(fabryka_api_key="sk-fab-test"))
    assert c.base_url == "https://router.fabryka.ai/v1"
    c2 = TextCorrector(OcrConfig(fabryka_api_key="fab_live_test"))
    assert c2.base_url == "https://fabryka.ai/v1"


# --- korekta ---


def test_correct_disabled_returns_unchanged():
    corrector = _make_corrector(response_text="POPRAWIONO", correct=False)
    assert corrector.correct("oryginalny tekst") == "oryginalny tekst"


def test_correct_no_api_key_returns_unchanged():
    config = OcrConfig(correct_text=True, fabryka_api_key=None)
    corrector = TextCorrector(config)
    assert corrector.correct("tekst") == "tekst"
    assert not corrector.enabled


def test_correct_empty_text():
    corrector = _make_corrector(response_text="X")
    assert corrector.correct("") == ""
    assert corrector.correct("   ") == "   "


def test_correct_returns_corrected_text():
    corrector = _make_corrector(response_text="Żółw chodzi po łące")
    result = corrector.correct("Zolw chodzi po lace")
    assert result == "Żółw chodzi po łące"


def test_correct_strips_whitespace():
    corrector = _make_corrector(response_text="  poprawiony  ")
    result = corrector.correct("tekst")
    assert result == "poprawiony"


def test_correct_rejects_truncated_response():
    corrector = _make_corrector(response_text="ucięty")
    corrector._client.chat.completions.finish_reason = "length"
    assert corrector.correct("pełna oryginalna linia") == "pełna oryginalna linia"


def test_correct_api_failure_returns_original():
    corrector = _make_corrector(response_text="X")

    def _raise(**kwargs):
        raise RuntimeError("API error")

    corrector._client.chat.completions.create = _raise  # type: ignore[assignment]
    result = corrector.correct("oryginalny tekst")
    assert result == "oryginalny tekst"


def test_correct_uses_bielik_model():
    corrector = _make_corrector(response_text="ok")
    corrector.correct("tekst")
    fake = corrector._client.chat.completions  # type: ignore[attr-defined]
    assert fake.calls[0]["model"] == "bielik-11b-v3"
    assert fake.calls[0]["temperature"] == 0.0


def test_correct_preserves_system_prompt():
    corrector = _make_corrector(response_text="ok")
    corrector.correct("tekst")
    fake = corrector._client.chat.completions  # type: ignore[attr-defined]
    messages = fake.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "OCR" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "tekst"


# --- check_connection ---


def test_check_connection_no_key():
    corrector = TextCorrector(OcrConfig(fabryka_api_key=None))
    ok, msg = corrector.check_connection()
    assert ok is False
    assert "Brak klucza" in msg


def test_check_connection_success():
    corrector = _make_corrector(response_text="test-reply", model_ids=["bielik-11b-v3", "auto"])
    ok, msg = corrector.check_connection()
    assert ok is True
    assert "OK" in msg
    assert "test-reply" in msg


def test_check_connection_model_not_available():
    corrector = _make_corrector(model="nonexistent-model", model_ids=["bielik-11b-v3", "auto"])
    ok, msg = corrector.check_connection()
    assert ok is False
    assert "nonexistent-model" in msg


def test_check_connection_chat_fails():
    corrector = _make_corrector(response_text="ok")

    # /models działa, ale /chat/completions nie
    def _raise(**kwargs):
        raise RuntimeError("401")

    corrector._client.chat.completions.create = _raise  # type: ignore[assignment]
    ok, msg = corrector.check_connection()
    assert ok is False
    assert "POST /chat/completions" in msg
