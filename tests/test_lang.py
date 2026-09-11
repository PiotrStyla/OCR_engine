"""Testy detekcji języka (PL/EN)."""

from __future__ import annotations

from ocr.lang import detect_language


def test_force_language_overrides():
    assert detect_language("hello world", force="pl") == "pl"
    assert detect_language("żółw", force="en") == "en"


def test_empty_text_unknown():
    assert detect_language("") == "unknown"
    assert detect_language("   ") == "unknown"


def test_polish_diacritics_detected():
    assert detect_language("Żółw chodzi po łące") == "pl"
    assert detect_language("Cześć, jak się masz?") == "pl"


def test_english_default():
    # tekst bez diakrytyków i na tyle krótki by langdetect nie zadziałał
    assert detect_language("hello") == "en"


def test_mixed_with_some_diacritics_polish():
    # nawet pojedynczy diakrytyk sugeruje polski
    assert detect_language("Zażółć gęśl") == "pl"
