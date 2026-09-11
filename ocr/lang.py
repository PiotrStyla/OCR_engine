"""Detekcja języka linii tekstu (PL/EN) do routingu modeli TrOCR.

Heurystyka: najpierw licznik polskich znaków diakrytycznych, potem
langdetect jako potwierdzenie. Jeśli wymuszono język w konfiguracji,
zwracany jest on bez analizy.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .result import Language

# Polskie znaki diakrytyczne (małe i wielkie)
_PL_DIACRITICS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


@lru_cache(maxsize=1)
def _langdetect_available() -> bool:
    try:
        import langdetect  # noqa: F401
        return True
    except ImportError:
        return False


def detect_language(text: str, force: str | None = None) -> Language:
    """Rozpoznaje język tekstu linii."""
    if force in ("pl", "en"):
        return force  # type: ignore[return-value]

    if not text or not text.strip():
        return "unknown"

    # 1. Szybka heurystyka: polskie diakrytyki
    chars = [c for c in text if c.isalpha()]
    if chars:
        diacritic_count = sum(1 for c in chars if c in _PL_DIACRITICS)
        ratio = diacritic_count / len(chars)
        # nawet kilka % diakrytyków to silny sygnał polskiego
        if ratio >= 0.02:
            return "pl"

    # 2. langdetect dla dłuższych tekstów
    clean = re.sub(r"[^A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż ]+", " ", text).strip()
    if len(clean) >= 8 and _langdetect_available():
        try:
            from langdetect import detect
            lang = detect(clean)
            if lang == "pl":
                return "pl"
            if lang == "en":
                return "en"
        except Exception:
            pass

    # 3. Fallback: jeśli są polskie diakrytyki (nawet <2%) → pl, inaczej en
    if any(c in _PL_DIACRITICS for c in text):
        return "pl"
    return "en"
