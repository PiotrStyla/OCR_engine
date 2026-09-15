"""Testy dla training.augment_lines."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from training.augment_lines import _augment_real, augment_dir


def _make_pair(dir: Path, name: str, text: str, size=(100, 30)) -> None:
    img = Image.new("RGB", size, (250, 250, 250))
    img.save(dir / f"{name}.png")
    (dir / f"{name}.txt").write_text(text, encoding="utf-8")


def test_augment_real_returns_image(tmp_path: Path) -> None:
    import random
    img = Image.new("RGB", (120, 40), (200, 200, 200))
    rng = random.Random(42)
    out = _augment_real(img, rng)
    assert isinstance(out, Image.Image)
    assert out.size[0] > 0 and out.size[1] > 0


def test_augment_real_strength_light(tmp_path: Path) -> None:
    """Light strength should not apply gamma/contrast/JPEG (prob=0)."""
    import random
    img = Image.new("RGB", (120, 40), (200, 200, 200))
    rng = random.Random(42)
    out = _augment_real(img, rng, strength="light")
    assert isinstance(out, Image.Image)
    # Light should still return a valid image
    assert out.size[0] > 0 and out.size[1] > 0


def test_augment_real_strength_invalid_raises(tmp_path: Path) -> None:
    import random
    img = Image.new("RGB", (120, 40), (200, 200, 200))
    rng = random.Random(42)
    with pytest.raises(KeyError):
        _augment_real(img, rng, strength="nonexistent")


def test_augment_real_deterministic_with_seed(tmp_path: Path) -> None:
    import random
    img = Image.new("RGB", (120, 40), (200, 200, 200))
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    out1 = _augment_real(img, rng1)
    out2 = _augment_real(img, rng2)
    assert list(out1.getdata()) == list(out2.getdata())


def test_augment_dir_copies_originals_and_augments(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _make_pair(src, "line1", "Ala ma kota")
    _make_pair(src, "line2", "Zażółć gęśl")

    total = augment_dir(src, dst, copies=3, seed=42)

    # 2 originals + 2×3 augmented = 8
    assert total == 8
    pngs = sorted(dst.glob("*.png"))
    txts = sorted(dst.glob("*.txt"))
    assert len(pngs) == 8
    assert len(txts) == 8

    # Originals preserved
    assert (dst / "line1.png").exists()
    assert (dst / "line1.txt").exists()
    assert (dst / "line2.png").exists()
    assert (dst / "line2.txt").exists()

    # Augmented copies exist
    for k in range(3):
        assert (dst / f"line1_aug{k}.png").exists()
        assert (dst / f"line1_aug{k}.txt").exists()
        assert (dst / f"line2_aug{k}.png").exists()
        assert (dst / f"line2_aug{k}.txt").exists()

    # Augmented text matches original
    assert (dst / "line1_aug0.txt").read_text(encoding="utf-8") == "Ala ma kota"
    assert (dst / "line2_aug2.txt").read_text(encoding="utf-8") == "Zażółć gęśl"


def test_augment_dir_strength_light(tmp_path: Path) -> None:
    """Light strength should produce same count, just gentler augmentations."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _make_pair(src, "line1", "Ala ma kota")

    total = augment_dir(src, dst, copies=2, seed=42, strength="light")

    # 1 original + 2 augmented = 3
    assert total == 3
    assert (dst / "line1.png").exists()
    assert (dst / "line1_aug0.png").exists()
    assert (dst / "line1_aug1.png").exists()


def test_augment_dir_skips_empty_text(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _make_pair(src, "good", "Ala ma kota")
    # Empty text pair
    img = Image.new("RGB", (100, 30), (250, 250, 250))
    img.save(src / "empty.png")
    (src / "empty.txt").write_text("", encoding="utf-8")

    total = augment_dir(src, dst, copies=2, seed=42)

    # Only "good" counts: 1 original + 2 augmented = 3
    assert total == 3
    assert not (dst / "empty.png").exists()
    assert not (dst / "empty_aug0.png").exists()


def test_augment_dir_empty_input_raises(tmp_path: Path) -> None:
    src = tmp_path / "empty_src"
    dst = tmp_path / "dst"
    src.mkdir()
    with pytest.raises(SystemExit):
        augment_dir(src, dst, copies=2, seed=42)
