"""Testy dla struktur danych w ocr.result."""

from __future__ import annotations

import pytest

from ocr.result import BBox, OcrResult, TextLine


def test_bbox_geometry():
    bbox = BBox(10, 20, 110, 80)
    assert bbox.width == 100
    assert bbox.height == 60
    assert bbox.area == 6000
    assert bbox.to_tuple() == (10, 20, 110, 80)


def test_bbox_expand_clips_to_bounds():
    bbox = BBox(0, 0, 50, 50)
    expanded = bbox.expand(pad=10, max_w=100, max_h=100)
    assert expanded == BBox(0, 0, 60, 60)
    # obcięcie przy krawędzi
    edge = BBox(95, 95, 100, 100)
    expanded_edge = edge.expand(pad=10, max_w=100, max_h=100)
    assert expanded_edge == BBox(85, 85, 100, 100)


def test_text_line_confidence_validation():
    TextLine(text="hello", bbox=BBox(0, 0, 10, 10), language="en", confidence=0.5)
    with pytest.raises(ValueError):
        TextLine(text="hello", bbox=BBox(0, 0, 10, 10), language="en", confidence=1.5)
    with pytest.raises(ValueError):
        TextLine(text="hello", bbox=BBox(0, 0, 10, 10), language="en", confidence=-0.1)


def test_ocr_result_text_join():
    lines = [
        TextLine("Pierwsza linia", BBox(0, 0, 100, 20), "pl", 0.9),
        TextLine("Druga linia", BBox(0, 30, 100, 50), "pl", 0.8),
    ]
    result = OcrResult(lines=lines, image_size=(100, 50))
    assert result.text == "Pierwsza linia\nDruga linia"
    assert "pl" in result.languages


def test_ocr_result_to_dict():
    line = TextLine("test", BBox(1, 2, 3, 4), "en", 0.7)
    result = OcrResult(lines=[line], image_size=(10, 10))
    d = result.to_dict()
    assert d["text"] == "test"
    assert d["image_size"] == (10, 10)
    assert d["lines"][0]["bbox"] == [1, 2, 3, 4]
    assert d["lines"][0]["language"] == "en"
