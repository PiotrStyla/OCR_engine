"""Postprocessing tekstu po OCR: korekta błędów rozpoznawania przez Fabryka API.

Fabryka (https://fabryka.ai) udostępnia polskie modele LLM (Bielik) przez API
kompatybilne z OpenAI. Model językowy dobrze nadaje się do naprawy typowych
błędów OCR: zamiana diakrytyków (ą→a, ł→l, ę→e), łączenie pociętych słów,
normalizacja interpunkcji — przy zachowaniu oryginalnego sensu.

Obsługiwane endpointy (auto-detekcja):
  - https://fabryka.ai/v1        (lab, klucze fab_live_...)
  - https://router.fabryka.ai/v1 (router, klucze sk-fab-...)

Jeśli base_url nie jest ustawione, wybieramy na podstawie prefixu klucza.
Brak klucza → korekta pominięta z ostrzeżeniem (graceful).
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import OcrConfig

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Jesteś korektorem tekstu rozpoznanego przez OCR. Otrzymasz tekst z błędami "
    "rozpoznawania (często brak polskich diakrytyków: ą→a, ę→e, ł→l, ó→o, ś→s, "
    "ż→z, ź→z, ć→c, ń→n; czasami pocięte słowa, błędna interpunkcja). "
    "Zwróć WYŁĄCZNIE poprawiony tekst, bez komentarzy, bez wyjaśnień, "
    "bez markdown. Zachowaj oryginalny układ linii i sens. "
    "Nie dodawaj informacji, której nie ma w tekście. "
    "Jeśli tekst jest poprawny, zwróć go bez zmian."
)

# Endpointy Fabryki
_FABRYKA_LAB_URL = "https://fabryka.ai/v1"
_FABRYKA_ROUTER_URL = "https://router.fabryka.ai/v1"

# Kody błędów do retry z backoffem (wg wskazówek Hermes z fabryka.ai)
_RETRYABLE_STATUS = {429, 502, 503, 504}


def _detect_base_url(api_key: str, configured_url: str) -> str:
    """Wybiera endpoint na podstawie prefixu klucza, jeśli URL jest 'auto'."""
    if configured_url != "auto":
        return configured_url
    if api_key.startswith("sk-fab"):
        return _FABRYKA_ROUTER_URL
    return _FABRYKA_LAB_URL


class TextCorrector:
    """Korekta tekstu OCR przez Fabryka API (Bielik)."""

    def __init__(self, config: "OcrConfig") -> None:
        self.config = config
        self._client = None
        self._base_url: str | None = None

    @property
    def api_key(self) -> str | None:
        return self.config.fabryka_api_key or os.environ.get("FABRYKA_API_KEY")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            key = self.api_key or ""
            self._base_url = _detect_base_url(key, self.config.fabryka_base_url)
        return self._base_url

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Pakiet 'openai' jest wymagany do korekty tekstu. "
                "Zainstaluj: pip install ocr-engine[correct]"
            ) from e
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.config.fabryka_timeout,
        )
        return self._client

    def correct(self, text: str) -> str:
        """Poprawia tekst po OCR. Zwraca tekst bez zmian jeśli korekta wyłączona/niedostępna."""
        if not text or not text.strip():
            return text
        if not self.config.correct_text:
            return text
        if not self.api_key:
            logger.warning(
                "Korekta tekstu włączona, ale brak FABRYKA_API_KEY — pomijam."
            )
            return text

        client = self._ensure_client()
        if client is None:
            return text

        try:
            return self._call_with_retry(client, text)
        except Exception as e:
            logger.warning("Korekta tekstu przez Fabryka API nie powiodła się: %s", e)
            return text

    def _call_with_retry(self, client, text: str) -> str:
        """Wywołuje API z retry z capped exponential backoff dla 429/502/503/504."""
        import openai

        max_retries = 3
        base_delay = 1.0  # sekundy

        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.config.fabryka_model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.0,
                    max_tokens=max(256, len(text) * 4),
                )
                corrected = resp.choices[0].message.content
                if corrected is None:
                    return text
                return corrected.strip()
            except openai.APIStatusError as e:
                status = getattr(e, "status_code", None)
                if status in _RETRYABLE_STATUS and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # 1, 2, 4s
                    logger.debug(
                        "Fabryka API %d, retry %d/%d za %.1fs", status, attempt + 1, max_retries, delay
                    )
                    time.sleep(delay)
                    continue
                # 401/403 lub wyczerpane retry → przekaż dalej
                raise
            except openai.APIConnectionError as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.debug("Fabryka API conn error, retry %d/%d", attempt + 1, max_retries)
                    time.sleep(delay)
                    continue
                raise
        return text  # fallback (nie powinno tu dojść)

    def check_connection(self) -> tuple[bool, str]:
        """Sprawdza połączenie z API i ważność klucza. Zwraca (ok, komunikat)."""
        if not self.api_key:
            return False, "Brak klucza API (FABRYKA_API_KEY)"
        try:
            client = self._ensure_client()
            if client is None:
                return False, "Nie udało się utworzyć klienta"
        except ImportError as e:
            return False, str(e)

        # 1. GET /models — szybki test autoryzacji
        try:
            models = client.models.list()
            model_ids = [m.id for m in models.data]
            if self.config.fabryka_model not in model_ids:
                return False, (
                    f"Model '{self.config.fabryka_model}' niedostępny. "
                    f"Dostępne: {', '.join(model_ids)}"
                )
        except Exception as e:
            return False, f"GET /models nie powiodło się: {e}"

        # 2. POST /chat/completions — minimalne zapytanie testowe
        try:
            resp = client.chat.completions.create(
                model=self.config.fabryka_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            return True, f"OK (endpoint: {self.base_url}, model: {self.config.fabryka_model}, odpowiedź: {content!r})"
        except Exception as e:
            return False, (
                f"GET /models działa, ale POST /chat/completions nie: {e}. "
                f"Sprawdź saldo konta / uprawnienia klucza na dashboardzie."
            )

    def close(self) -> None:
        self._client = None
