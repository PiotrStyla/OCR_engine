"""Augmentacja realnych cropów linii tekstu (np. EHRI) dla treningu TrOCR.

Generuje N augmentowanych kopii każdej pary .png/.txt z katalogu wejściowego.
Augmentacje symulują degradację maszynopisu: blur, szum gaussowski, zmiana
kontrastu/jasności, kompresja JPEG, lekkie pochylenie. Styl zgodny z
`generate_synthetic._augment`, ale dostosowany do realnych cropów.

Użycie:
    python -m training.augment_lines --input data/ehri-pl-lines/train \
        --output data/ehri-pl-lines-aug/train --copies 5 --seed 42

Nie augmentuje dev/test — tylko train. Zachowuje oryginalne pary i dodaje
augmentowane z sufiksem _aug<k>.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


# Presety intensywności augmentacji. Light = tylko blur+szum, niskie proby,
# bez JPEG/gamma — minimalne ryzyko regresji druku. Medium = run6 (kompromis).
# Heavy = pełna degradacja (dla eksperymentów ekstremalnych).
_STRENGTH_PRESETS = {
    "light": {
        "blur_prob": 0.25, "blur_range": (0.3, 0.7),
        "noise_prob": 0.35, "noise_range": (2.0, 6.0),
        "gamma_prob": 0.0, "gamma_range": (0.85, 1.15),
        "contrast_prob": 0.0, "contrast_range": (0.85, 1.0),
        "rotate_prob": 0.15, "rotate_range": (-1.5, 1.5),
        "jpeg_prob": 0.0, "jpeg_range": (70, 90),
    },
    "medium": {
        "blur_prob": 0.4, "blur_range": (0.3, 1.0),
        "noise_prob": 0.6, "noise_range": (3.0, 15.0),
        "gamma_prob": 0.5, "gamma_range": (0.7, 1.4),
        "contrast_prob": 0.3, "contrast_range": (0.6, 0.9),
        "rotate_prob": 0.3, "rotate_range": (-2.0, 2.0),
        "jpeg_prob": 0.3, "jpeg_range": (40, 85),
    },
    "heavy": {
        "blur_prob": 0.6, "blur_range": (0.5, 1.8),
        "noise_prob": 0.8, "noise_range": (5.0, 25.0),
        "gamma_prob": 0.7, "gamma_range": (0.5, 1.6),
        "contrast_prob": 0.5, "contrast_range": (0.4, 0.8),
        "rotate_prob": 0.4, "rotate_range": (-3.0, 3.0),
        "jpeg_prob": 0.5, "jpeg_range": (20, 70),
    },
}


def _augment_real(img: Image.Image, rng: random.Random, strength: str = "medium") -> Image.Image:
    """Augmentacje dostosowane do realnych cropów maszynopisu.

    `strength`: "light" (blur+szum tylko), "medium" (run6, pełny kompromis),
    "heavy" (ekstremalna degradacja).
    """
    s = _STRENGTH_PRESETS[strength]
    arr = np.asarray(img).astype(np.float32)

    # 1. Lekki blur (rozmycie atramentu)
    if rng.random() < s["blur_prob"]:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*s["blur_range"])))
        arr = np.asarray(img).astype(np.float32)

    # 2. Szum gaussowski (ziarno papieru)
    if rng.random() < s["noise_prob"]:
        noise_std = rng.uniform(*s["noise_range"])
        noise = np.random.default_rng(rng.getrandbits(64)).normal(0.0, noise_std, arr.shape)
        arr = np.clip(arr + noise, 0, 255)

    # 3. Zmiana jasności/kontrastu (gamma — symuluje blaknięcie/nierówną ekspozycję)
    if rng.random() < s["gamma_prob"]:
        arr_norm = arr / 255.0
        gamma = rng.uniform(*s["gamma_range"])
        arr_norm = np.power(arr_norm, gamma)
        arr = arr_norm * 255.0

    # 4. Redukcja kontrastu (wyblakły maszynopis)
    if rng.random() < s["contrast_prob"]:
        mean = arr.mean()
        contrast_factor = rng.uniform(*s["contrast_range"])
        arr = mean + (arr - mean) * contrast_factor

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 5. Lekkie pochylenie (skanowanie nieidealnie prosto)
    if rng.random() < s["rotate_prob"]:
        angle = rng.uniform(*s["rotate_range"])
        bg = int(arr.mean())
        img = img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=(bg, bg, bg))

    # 6. Kompresja JPEG (artefakty skanowania)
    if rng.random() < s["jpeg_prob"]:
        import io
        buf = io.BytesIO()
        quality = rng.randint(*s["jpeg_range"])
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


def augment_dir(
    input_dir: Path,
    output_dir: Path,
    copies: int,
    seed: int = 42,
    strength: str = "medium",
) -> int:
    """Augmentuje wszystkie pary .png/.txt z input_dir do output_dir.

    Kopiuje oryginały i dodaje `copies` augmentowanych wersji każdej.
    `strength`: "light", "medium", lub "heavy" (patrz _STRENGTH_PRESETS).
    Zwraca łączną liczbę par w output_dir.
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = sorted(p for p in input_dir.glob("*.png") if p.with_suffix(".txt").exists())
    if not pairs:
        raise SystemExit(f"Brak par .png/.txt w {input_dir}")

    total = 0
    for img_path in pairs:
        text = img_path.with_suffix(".txt").read_text(encoding="utf-8").strip()
        if not text:
            continue
        stem = img_path.stem

        # Kopiuj oryginał
        orig_img = Image.open(img_path).convert("RGB")
        orig_img.save(output_dir / f"{stem}.png")
        (output_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        total += 1

        # Generuj augmentacje
        for k in range(copies):
            aug = _augment_real(orig_img, rng, strength=strength)
            aug_name = f"{stem}_aug{k}"
            aug.save(output_dir / f"{aug_name}.png")
            (output_dir / f"{aug_name}.txt").write_text(text, encoding="utf-8")
            total += 1

    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Augmentuj realne cropy linii tekstu")
    p.add_argument("--input", required=True, help="Katalog z parami .png/.txt (train)")
    p.add_argument("--output", required=True, help="Katalog wyjściowy z augmentacjami")
    p.add_argument("--copies", type=int, default=5, help="Liczba augmentowanych kopii każdej linii")
    p.add_argument("--seed", type=int, default=42, help="Seed RNG dla reprodukowalności")
    p.add_argument("--strength", choices=list(_STRENGTH_PRESETS), default="medium",
                   help="Intensywność: light (blur+szum), medium (run6), heavy (ekstremalna)")
    args = p.parse_args()

    total = augment_dir(Path(args.input), Path(args.output), args.copies, args.seed, args.strength)
    print(f"Augmentowane: {total} par w {args.output} (oryginały + {args.copies}× {args.strength})")


if __name__ == "__main__":
    main()
