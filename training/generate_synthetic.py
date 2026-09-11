"""Generator syntetycznych danych treningowych dla polskiego TrOCR.

Generuje pary <id>.png + <id>.txt — obrazy pojedynczych linii tekstu
z polskimi diakrytykami, różnymi czcionkami i augmentacjami (szum, blur,
pochylenie, odcienie tła). Każdy obraz to jedna linia tekstu.

Uruchomienie:

    python -m training.generate_synthetic --output ./data/pl_lines_train --count 5000

Czcionki: domyślnie C:\\Windows\\Fonts (systemowe, obsługują polskie znaki).
Można wskazać własny katalog z .ttf/.otf przez --fonts-dir.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .corpus_pl import (
    DOMAIN_GENERATORS,
    FORMULAS,
    INVOICE,
    LEGAL,
    MEDICAL,
    SENTENCES,
    WORDS,
)

logger = logging.getLogger(__name__)

_DEFAULT_FONTS_DIR = Path(r"C:\Windows\Fonts")
_FONT_GLOBS = ("*.ttf", "*.otf")

# Subtelny przesiew kandydatów — preferujemy czytelne kroje tekstowe
_SKIP_FONT_SUBSTR = ("icon", "emoji", "ding", "wing", "symbol", "webd")


def _find_fonts(fonts_dir: Path) -> list[Path]:
    fonts: list[Path] = []
    for glob in _FONT_GLOBS:
        fonts.extend(fonts_dir.glob(glob))
    fonts = [
        f for f in fonts
        if not any(s in f.name.lower() for s in _SKIP_FONT_SUBSTR)
    ]
    return sorted(fonts)


def _font_supports_pl(font_path: Path) -> bool:
    """True jeśli czcionka ma glify polskich diakrytyków (sprawdza cmap).

    Bez tego filtra generowalibyśmy obrazy z pustymi kwadratami (.notdef)
    podpisane etykietą z diakrytykami — uszkodzone dane treningowe.
    """
    try:
        from fontTools.ttLib import TTFont
        cmap = TTFont(str(font_path), fontNumber=0, lazy=True).getBestCmap()
    except Exception:
        return False
    if cmap is None:
        return False
    return all(ord(ch) in cmap for ch in "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def _random_text(rng: random.Random, wiki_pool: list[str] | None = None) -> str:
    """Losowy tekst: zdanie, formula, domena specjalistyczna, wiki lub wyrazy."""
    kind = rng.random()
    if wiki_pool and kind < 0.30:
        return rng.choice(wiki_pool)
    if kind < 0.50:
        return rng.choice(SENTENCES)
    if kind < 0.60:
        return rng.choice(FORMULAS)
    if kind < 0.75:
        # domeny specjalistyczne (faktury/prawo/medycyna) lub wzorce liczbowe
        if rng.random() < 0.5:
            return rng.choice(rng.choice([INVOICE, LEGAL, MEDICAL]))
        return rng.choice(DOMAIN_GENERATORS)(rng)
    # kompozycja losowych wyrazów (2–8 słów)
    n = rng.randint(2, 8)
    words = rng.sample(WORDS, min(n, len(WORDS)))
    text = " ".join(words)
    if rng.random() < 0.5:
        text = text.capitalize()
    if rng.random() < 0.3:
        text += rng.choice([".", ",", ":", "!", "?"])
    return text


def _render_line(
    text: str,
    font_path: Path,
    font_size: int,
    rng: random.Random,
) -> Image.Image:
    """Renderuje jedną linię tekstu na jasnym tle z paddingiem."""
    font = ImageFont.truetype(str(font_path), font_size)
    # pomiar tekstu
    probe = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(probe)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad_x = rng.randint(8, 24)
    pad_y = rng.randint(6, 16)
    w = tw + 2 * pad_x
    h = th + 2 * pad_y

    # jasne tło z lekką losową odcieniem
    bg = rng.randint(225, 255)
    img = Image.new("RGB", (w, h), (bg, bg, bg - rng.randint(0, 8)))
    d = ImageDraw.Draw(img)
    # ciemny tekst
    fg = rng.randint(0, 60)
    d.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=(fg, fg, fg))
    return img


def _augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Losowe augmentacje: pochylenie, blur, szum, kontrast."""
    # pochylenie ±3°
    if rng.random() < 0.6:
        angle = rng.uniform(-3.0, 3.0)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=(250, 250, 250))
    # lekki blur
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))
    # szum gaussowski
    if rng.random() < 0.5:
        arr = np.asarray(img).astype(np.float32)
        noise = rng.random() * 12.0 + 2.0
        arr = arr + np.random.normal(0.0, noise, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # losowa zmiana jasności/kontrastu (prosta gamma)
    if rng.random() < 0.4:
        arr = np.asarray(img).astype(np.float32) / 255.0
        gamma = rng.uniform(0.8, 1.3)
        arr = np.power(arr, gamma)
        img = Image.fromarray((arr * 255).astype(np.uint8))
    return img


def generate(
    output_dir: Path,
    count: int,
    fonts_dir: Path = _DEFAULT_FONTS_DIR,
    seed: int | None = None,
    font_size_range: tuple[int, int] = (18, 42),
    wiki_sentences: int = 0,
    wiki_cache: Path | None = None,
) -> int:
    """Generuje `count` par .png/.txt w `output_dir`. Zwraca liczbę wygenerowanych.

    `wiki_sentences`: jeśli >0, pobiera zdania z pl.wikipedia.org i dodaje je
    do puli tekstów (~30% próbek). `wiki_cache` — plik .txt do zapisu/odczytu
    pobranych zdań (powtarzalność bez ponownego pobierania).
    """
    rng = random.Random(seed)
    fonts = _find_fonts(fonts_dir)
    if not fonts:
        raise SystemExit(f"Brak czcionek .ttf/.otf w {fonts_dir}")
    logger.info("Znaleziono %d czcionek w %s", len(fonts), fonts_dir)

    # filtr: tylko czcionki z polskimi glifami (inaczej dane są uszkodzone)
    fonts = [f for f in fonts if _font_supports_pl(f)]
    if not fonts:
        raise SystemExit(f"Żadna czcionka w {fonts_dir} nie obsługuje polskich znaków")
    logger.info("Czcionek z polskimi glifami: %d", len(fonts))

    wiki_pool: list[str] = []
    if wiki_sentences > 0:
        from .wiki_corpus import fetch_wiki_sentences, load_sentences, save_sentences

        if wiki_cache and wiki_cache.exists():
            wiki_pool = load_sentences(wiki_cache)
            logger.info("Wikipedia: wczytano %d zdań z cache %s", len(wiki_pool), wiki_cache)
        else:
            wiki_pool = fetch_wiki_sentences(wiki_sentences)
            if wiki_cache:
                wiki_cache.parent.mkdir(parents=True, exist_ok=True)
                save_sentences(wiki_pool, wiki_cache)
            logger.info("Wikipedia: pobrano %d zdań", len(wiki_pool))

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i in range(count):
        text = _random_text(rng, wiki_pool or None)
        font = rng.choice(fonts)
        size = rng.randint(*font_size_range)
        try:
            img = _render_line(text, font, size, rng)
            img = _augment(img, rng)
        except Exception as e:
            logger.debug("Pominięto próbkę %d (%s, %s): %s", i, font.name, text[:30], e)
            continue
        stem = f"{i:05d}"
        img.save(output_dir / f"{stem}.png")
        (output_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        written += 1
    logger.info("Wygenerowano %d par w %s", written, output_dir)
    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generator syntetycznych danych PL dla TrOCR")
    p.add_argument("--output", type=Path, required=True, help="Katalog wyjściowy (pary .png/.txt)")
    p.add_argument("--count", type=int, default=1000, help="Liczba próbek")
    p.add_argument("--fonts-dir", type=Path, default=_DEFAULT_FONTS_DIR)
    p.add_argument("--seed", type=int, default=None, help="Ziarno RNG (powtarzalność)")
    p.add_argument("--min-font", type=int, default=18)
    p.add_argument("--max-font", type=int, default=42)
    p.add_argument("--wiki-sentences", type=int, default=0,
                   help="pobierz N zdań z pl.wikipedia.org do korpusu")
    p.add_argument("--wiki-cache", type=Path, default=None,
                   help="plik .txt z cache'em zdań wiki (odczyt/zapis)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = _parse_args()
    generate(
        output_dir=a.output,
        count=a.count,
        fonts_dir=a.fonts_dir,
        seed=a.seed,
        font_size_range=(a.min_font, a.max_font),
        wiki_sentences=a.wiki_sentences,
        wiki_cache=a.wiki_cache,
    )


if __name__ == "__main__":
    main()
