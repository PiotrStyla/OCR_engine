"""Klient Jev (TypeSafe AI) — model decyzyjny do scoringu, routingu i walidacji.

Jev nie generuje tekstu — odpowiada na pytania typu Choice/Score/Noul
i zwraca strukturalne odpowiedzi z prawdopodobieństwami i confidence.

API: POST https://api.typesafe.ai/v1/systemone
Auth: Bearer token (env TYPESAFE_API_KEY)
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

TYPESAFE_API_URL = "https://api.typesafe.ai/v1/systemone"

# Próbkowanie tekstu do scoringu — nie wysyłaj całej strony (koszt tokenów)
_MAX_STATE_CHARS = 4000


class JevClient:
    """Klient Jev (TypeSafe System One API)."""

    def __init__(self, api_key: str, model: str = "jev-latest",
                 timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def ask(self, state: str, questions: dict) -> dict | None:
        """Wysyła zapytanie do Jev API. Zwraca dict odpowiedzi lub None przy błędzie.

        Args:
            state: Tekst kontekstu (np. OCR output, statystyki obrazu).
            questions: {name: {"type": "choice"|"score"|"noul", "instructions": str, "criteria": ...}}
        """
        if not self.enabled:
            return None
        # Ogranicz state do 4KB (koszt tokenów)
        state = state[:_MAX_STATE_CHARS] if state else ""
        payload = {
            "state": state,
            "model": self.model,
            "questions": questions,
        }
        req = urllib.request.Request(
            TYPESAFE_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("Jev API nie powiodło się: %s", e)
            return None

    def score_quality(self, text: str) -> tuple[float, float] | None:
        """Ocenia jakość tekstu OCR (0-2). Zwraca (score, confidence) lub None."""
        answer = self.ask(
            state=f"OCR output: {text[:2000]}",
            questions={
                "quality": {
                    "type": "score",
                    "instructions": "Rate OCR quality of this text",
                    "criteria": [
                        "Unreadable or heavily garbled text",
                        "Partially readable, some errors",
                        "Readable, clean text",
                    ],
                }
            },
        )
        if answer is None:
            return None
        try:
            a = answer["answers"]["quality"]
            return (a["score"], a.get("confidence", 0.0))
        except (KeyError, TypeError):
            return None

    def route_backend(self, stats_text: str) -> str | None:
        """Wybiera backend OCR na podstawie statystyk dokumentu.
        Zwraca 'trocr', 'kraken', 'paddlevl' lub None."""
        answer = self.ask(
            state=stats_text[:_MAX_STATE_CHARS],
            questions={
                "backend": {
                    "type": "choice",
                    "instructions": (
                        "Which OCR backend is best for this document? "
                        "Consider image quality, text type, and layout."
                    ),
                    "criteria": {
                        "trocr": "Clean printed text, standard fonts, good quality scans",
                        "kraken": "Typewritten, historical, or degraded documents with baseline text",
                        "paddlevl": "Complex layout, mixed content, tables, figures",
                    },
                }
            },
        )
        if answer is None:
            return None
        try:
            a = answer["answers"]["backend"]
            backend = a.get("choice")
            if backend in ("trocr", "kraken", "paddlevl"):
                return backend
            return None
        except (KeyError, TypeError):
            return None

    def validate_correction(self, raw: str, corrected: str) -> bool | None:
        """Czy poprawiony tekst jest lepszy niż surowy? Zwraca True/False/None."""
        answer = self.ask(
            state=f"Original OCR: {raw[:1000]}\nCorrected: {corrected[:1000]}",
            questions={
                "better": {
                    "type": "noul",
                    "instructions": (
                        "The corrected text is better than the original OCR "
                        "(fewer errors, better Polish diacritics, same meaning)"
                    ),
                }
            },
        )
        if answer is None:
            return None
        try:
            a = answer["answers"]["better"]
            noul = a.get("noul", 0.0)
            return noul > 0.5
        except (KeyError, TypeError):
            return None

    def close(self) -> None:
        """Nic do zamknięcia — klient jest stateless."""
