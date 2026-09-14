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


def _augment_real(img: Image.Image, rng: random.Random) -> Image.Image:
    """Augmentacje dostosowane do realnych cropów maszynopisu."""
    arr = np.asarray(img).astype(np.float32)

    # 1. Lekki blur (rozmycie atramentu)
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))
        arr = np.asarray(img).astype(np.float32)

    # 2. Szum gaussowski (ziarno papieru)
    if rng.random() < 0.6:
        noise_std = rng.uniform(3.0, 15.0)
        noise = np.random.default_rng(rng.getrandbits(64)).normal(0.0, noise_std, arr.shape)
        arr = np.clip(arr + noise, 0, 255)

    # 3. Zmiana jasności/kontrastu (gamma — symuluje blaknięcie/nierówną ekspozycję)
    if rng.random() < 0.5:
        arr_norm = arr / 255.0
        gamma = rng.uniform(0.7, 1.4)
        arr_norm = np.power(arr_norm, gamma)
        arr = arr_norm * 255.0

    # 4. Redukcja kontrastu (wyblakły maszynopis)
    if rng.random() < 0.3:
        mean = arr.mean()
        contrast_factor = rng.uniform(0.6, 0.9)
        arr = mean + (arr - mean) * contrast_factor

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 5. Lekkie pochylenie ±2° (skanowanie nieidealnie prosto)
    if rng.random() < 0.3:
        angle = rng.uniform(-2.0, 2.0)
        bg = int(arr.mean())
        img = img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=(bg, bg, bg))

    # 6. Kompresja JPEG (artefakty skanowania)
    if rng.random() < 0.3:
        import io
        buf = io.BytesIO()
        quality = rng.randint(40, 85)
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


def augment_dir(input_dir: Path, output_dir: Path, copies: int, seed: int = 42) -> int:
    """Augmentuje wszystkie pary .png/.txt z input_dir do output_dir.

    Kopiuje oryginały i dodaje `copies` augmentowanych wersji każdej.
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
            aug = _augment_real(orig_img, rng)
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
    args = p.parse_args()

    total = augment_dir(Path(args.input), Path(args.output), args.copies, args.seed)
    print(f"Augmentowane: {total} par w {args.output} (oryginały + {args.copies}× augmentacje)")


if __name__ == "__main__":
    main()
