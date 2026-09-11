"""Testy generatora syntetycznych danych treningowych."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image, ImageFont

from training.generate_synthetic import (
    _font_supports_pl,
    _random_text,
    _render_line,
    generate,
)


def test_random_text_returns_nonempty(tmp_path):
    rng = random.Random(42)
    for _ in range(20):
        text = _random_text(rng)
        assert isinstance(text, str)
        assert len(text.strip()) > 0


def test_random_text_has_polish_chars(tmp_path):
    """Korpus zawiera polskie diakrytyki w części próbek."""
    rng = random.Random(7)
    texts = [_random_text(rng) for _ in range(50)]
    pl_chars = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
    assert any(any(c in pl_chars for c in t) for t in texts)


def test_font_supports_pl_arial():
    """Arial obsługuje polskie znaki (sanity check filtra)."""
    arial = Path(r"C:\Windows\Fonts\arial.ttf")
    if not arial.exists():
        pytest.skip("arial.ttf niedostępny")
    assert _font_supports_pl(arial) is True


def test_render_line_produces_image():
    """Render linii zwraca obraz PIL o rozsądnych wymiarach."""
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    if not font_path.exists():
        pytest.skip("arial.ttf niedostępny")
    rng = random.Random(0)
    img = _render_line("Zażółć gęślą jaźń", font_path, 24, rng)
    assert isinstance(img, Image.Image)
    assert img.width > 100
    assert img.height > 20
    assert img.mode == "RGB"


def test_generate_creates_pairs(tmp_path):
    """generate() tworzy pary .png/.txt."""
    fonts_dir = Path(r"C:\Windows\Fonts")
    if not fonts_dir.exists():
        pytest.skip("Brak katalogu czcionek Windows")
    out = tmp_path / "out"
    written = generate(out, count=5, fonts_dir=fonts_dir, seed=1)
    assert written == 5
    pngs = sorted(out.glob("*.png"))
    txts = sorted(out.glob("*.txt"))
    assert len(pngs) == 5
    assert len(txts) == 5
    # każdy .png ma odpowiadający .txt z tekstem
    for p in pngs:
        t = p.with_suffix(".txt")
        assert t.exists()
        assert len(t.read_text(encoding="utf-8").strip()) > 0


def test_generate_deterministic_seed(tmp_path):
    """To samo ziarno → identyczne etykiety."""
    fonts_dir = Path(r"C:\Windows\Fonts")
    if not fonts_dir.exists():
        pytest.skip("Brak katalogu czcionek Windows")
    d1, d2 = tmp_path / "a", tmp_path / "b"
    generate(d1, count=3, fonts_dir=fonts_dir, seed=99)
    generate(d2, count=3, fonts_dir=fonts_dir, seed=99)
    labels1 = sorted(p.read_text(encoding="utf-8") for p in d1.glob("*.txt"))
    labels2 = sorted(p.read_text(encoding="utf-8") for p in d2.glob("*.txt"))
    assert labels1 == labels2
