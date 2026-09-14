"""Testy detektora: sortowanie porządku czytania (nie wymaga modelu CRAFT)."""

from __future__ import annotations

import cv2
import numpy as np

from ocr.detector import sort_reading_order
from ocr.opencv_detector import _dedup, _line_bands, _split_block
from ocr.result import BBox


def test_sort_empty():
    assert sort_reading_order([], 100) == []


def test_sort_single_row_left_to_right():
    bboxes = [BBox(100, 10, 200, 30), BBox(0, 10, 90, 30), BBox(210, 10, 300, 30)]
    ordered = sort_reading_order(bboxes, image_height=100)
    xs = [b.x1 for b in ordered]
    assert xs == [0, 100, 210]


def test_sort_multiple_rows_top_to_bottom():
    bboxes = [
        BBox(0, 100, 100, 120),  # rząd 2
        BBox(0, 0, 100, 20),      # rząd 1
        BBox(0, 200, 100, 220),  # rząd 3
    ]
    ordered = sort_reading_order(bboxes, image_height=300)
    ys = [b.y1 for b in ordered]
    assert ys == [0, 100, 200]


def test_sort_groups_close_lines_into_row():
    # dwie linie na podobnej wysokości -> jeden rząd, sortowane po x
    bboxes = [
        BBox(150, 12, 250, 30),
        BBox(0, 10, 100, 28),
    ]
    ordered = sort_reading_order(bboxes, image_height=100)
    assert [b.x1 for b in ordered] == [0, 150]


def test_split_block_separates_two_text_lines():
    gray = np.full((120, 500), 255, dtype=np.uint8)
    cv2.putText(gray, "Pierwsza linia", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    cv2.putText(gray, "Druga linia", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

    lines = _split_block(gray, BBox(0, 0, 500, 120), character_height=20)

    assert len(lines) == 2
    assert lines[0].y2 < lines[1].y1


def test_split_block_preserves_single_line_box():
    gray = np.full((40, 300), 255, dtype=np.uint8)
    bbox = BBox(0, 0, 300, 40)
    assert _split_block(gray, bbox, character_height=20) == [bbox]


def test_split_block_separates_many_lines():
    gray = np.full((320, 500), 255, dtype=np.uint8)
    for row in range(6):
        y = 40 + row * 50
        cv2.putText(gray, "linia tekstu", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

    lines = _split_block(gray, BBox(0, 0, 500, 320), character_height=20)
    assert len(lines) == 6
    tops = [line.y1 for line in lines]
    assert tops == sorted(tops)


def test_line_bands_finds_gaps():
    binary = np.zeros((100, 200), dtype=np.uint8)
    binary[10:25, :] = 255  # pierwsza linia
    binary[60:75, :] = 255  # druga linia (wyraźna dolina)
    bands = _line_bands(binary, character_height=15)
    assert len(bands) == 2


def test_line_bands_empty_region():
    assert _line_bands(np.zeros((50, 50), dtype=np.uint8), character_height=15) == []


def test_dedup_keeps_tighter_box():
    loose = BBox(0, 0, 200, 60)
    tight = BBox(5, 5, 195, 55)
    result = _dedup([loose, tight])
    assert result == [tight]


def test_dedup_preserves_distinct_lines():
    first = BBox(0, 0, 200, 40)
    second = BBox(0, 60, 200, 100)
    result = _dedup([first, second])
    assert len(result) == 2
