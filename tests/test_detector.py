"""Testy detektora: sortowanie porządku czytania (nie wymaga modelu CRAFT)."""

from __future__ import annotations

from ocr.detector import sort_reading_order
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
