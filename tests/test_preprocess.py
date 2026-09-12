"""Testy preprocessingu (nie wymagają modeli ML)."""

from __future__ import annotations

import numpy as np
import pytest

from ocr.preprocess import (
    crop_to_bbox,
    deskew,
    load_image,
    to_grayscale,
    to_pil_rgb,
)


def _white_image(w: int, h: int, color=(255, 255, 255)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def test_load_image_from_array_rgb():
    arr = _white_image(20, 10, color=(10, 20, 30))
    loaded = load_image(arr)
    assert loaded.shape == (10, 20, 3)
    assert tuple(loaded[0, 0]) == (10, 20, 30)


def test_load_image_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path / "nieistnieje.png"))


def test_to_grayscale_shape():
    arr = _white_image(30, 15)
    gray = to_grayscale(arr)
    assert gray.shape == (15, 30)


def test_deskew_returns_same_for_blank():
    arr = _white_image(100, 100)
    out = deskew(arr)
    assert out.shape == arr.shape


def test_crop_to_bbox():
    arr = _white_image(100, 100, color=(123, 45, 67))
    crop = crop_to_bbox(arr, (10, 10, 30, 40))
    assert crop.shape == (30, 20, 3)


def test_crop_to_bbox_clips_bounds():
    arr = _white_image(50, 50)
    crop = crop_to_bbox(arr, (-5, -5, 100, 100))
    assert crop.shape == (50, 50, 3)


def test_to_pil_rgb():
    arr = _white_image(10, 10, color=(1, 2, 3))
    img = to_pil_rgb(arr)
    assert img.size == (10, 10)
    assert img.mode == "RGB"
