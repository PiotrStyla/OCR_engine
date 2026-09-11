"""Pobieranie losowych zdań z polskiej Wikipedii do korpusu treningowego.

Używa MediaWiki API (generator=random + extracts). Wyniki można zapisać
do pliku (jedno zdanie na linię) i używać jako rozszerzenia korpusu —
patrz flaga --wiki-sentences w generate_synthetic.

Wymaga dostępu do sieci. Zdania są filtrowane: długość, litery, diakrytyki.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_API = "https://pl.wikipedia.org/w/api.php"
_PL_DIACRITICS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9„(])")
# wzorce do wycięcia: przypisy [1], nawiasowe infoboksy itp.
_CLEAN_PATTERNS = [
    (re.compile(r"\[\d+\]"), ""),
    (re.compile(r"\s+"), " "),
]


def clean_text(text: str) -> str:
    """Czyści wyciąg wiki: przypisy, nadmiarowe spacje."""
    for pat, repl in _CLEAN_PATTERNS:
        text = pat.sub(repl, text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Dzieli tekst na zdania."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def is_good_sentence(s: str, min_len: int = 20, max_len: int = 160) -> bool:
    """Filtr jakości: długość, litery, nie za dużo cyfr/znaków specjalnych."""
    if not min_len <= len(s) <= max_len:
        return False
    letters = sum(1 for c in s if c.isalpha())
    if letters < len(s) * 0.55:
        return False
    # zbyt wiele nawiasów/cudzysłowów → często metatekst (przypisy, szablony)
    if s.count("(") + s.count(")") > 2 or s.count('"') > 4:
        return False
    return True


def has_diacritics(s: str) -> bool:
    return any(c in _PL_DIACRITICS for c in s)


def fetch_random_articles(n: int = 50, timeout: float = 15.0) -> list[str]:
    """Pobiera n losowych artykułów (extracts) z pl.wikipedia.org."""
    import requests

    params = {
        "action": "query",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": min(n, 50),  # limit API na request
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "format": "json",
        "formatversion": 2,
    }
    resp = requests.get(_API, params=params, timeout=timeout,
                        headers={"User-Agent": "ocr-engine-corpus/0.1"})
    resp.raise_for_status()
    data = resp.json()
    return [
        p.get("extract", "")
        for p in data.get("query", {}).get("pages", [])
        if p.get("extract")
    ]


def fetch_wiki_sentences(
    target: int = 500,
    batch: int = 50,
    prefer_diacritics: bool = True,
    timeout: float = 15.0,
) -> list[str]:
    """Zbiera `target` losowych zdań z Wikipedii.

    `prefer_diacritics`: zdania z diakrytykami trafiają na początek puli.
    """
    out_with_dia: list[str] = []
    out_no_dia: list[str] = []
    seen: set[str] = set()

    while len(out_with_dia) + len(out_no_dia) < target:
        try:
            articles = fetch_random_articles(batch, timeout=timeout)
        except Exception as e:
            logger.warning("Pobieranie wiki nie powiodło się: %s", e)
            break
        if not articles:
            break
        for art in articles:
            for sent in split_sentences(clean_text(art)):
                if not is_good_sentence(sent) or sent in seen:
                    continue
                seen.add(sent)
                (out_with_dia if has_diacritics(sent) else out_no_dia).append(sent)
                if len(out_with_dia) + len(out_no_dia) >= target:
                    break

    if prefer_diacritics:
        return (out_with_dia + out_no_dia)[:target]
    return (out_no_dia + out_with_dia)[:target]


def save_sentences(sentences: list[str], path: str | Path) -> None:
    Path(path).write_text("\n".join(sentences) + "\n", encoding="utf-8")


def load_sentences(path: str | Path) -> list[str]:
    return [
        s.strip() for s in Path(path).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
