"""Testy obsługi PDF: iter_pdf_pages, parse_page_range, recognize_pdf."""

from __future__ import annotations

import numpy as np
import pytest

from ocr.preprocess import iter_pdf_pages, parse_page_range

fitz = pytest.importorskip("fitz")


@pytest.fixture
def pdf_3_pages(tmp_path):
    """Testowy 3-stronicowy PDF z tekstem."""
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((50, 50), f"Strona {i + 1}", fontsize=20)
    path = tmp_path / "test.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_parse_page_range_all():
    assert parse_page_range(None, 4) == [0, 1, 2, 3]


def test_parse_page_range_single():
    assert parse_page_range("2", 5) == [1]


def test_parse_page_range_span():
    assert parse_page_range("1-3", 5) == [0, 1, 2]


def test_parse_page_range_mixed():
    assert parse_page_range("1-2,4", 5) == [0, 1, 3]


def test_parse_page_range_clamps():
    """Zakres poza dokumentem jest obcinany."""
    assert parse_page_range("2-99", 3) == [1, 2]


def test_iter_pdf_pages_all(pdf_3_pages):
    pages = list(iter_pdf_pages(pdf_3_pages, dpi=72))
    assert len(pages) == 3
    for no, arr in pages:
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3 and arr.shape[2] == 3
        assert arr.dtype == np.uint8


def test_iter_pdf_pages_subset(pdf_3_pages):
    pages = list(iter_pdf_pages(pdf_3_pages, dpi=72, pages="2"))
    assert len(pages) == 1
    assert pages[0][0] == 2  # numer strony 1-indeksowany


def test_iter_pdf_pages_dpi_scales_size(pdf_3_pages):
    arr72 = next(iter(iter_pdf_pages(pdf_3_pages, dpi=72)))[1]
    arr144 = next(iter(iter_pdf_pages(pdf_3_pages, dpi=144)))[1]
    assert arr144.shape[0] == arr72.shape[0] * 2
    assert arr144.shape[1] == arr72.shape[1] * 2


def test_recognize_pdf_calls_pipeline_per_page(monkeypatch, pdf_3_pages):
    """recognize_pdf wywołuje _recognize_array raz na stronę."""
    from ocr.config import OcrConfig
    from ocr.pipeline import OcrEngine
    from ocr.result import OcrResult

    engine = OcrEngine(OcrConfig(deskew=False))
    seen = []
    engine._recognize_array = lambda arr: seen.append(arr.shape) or OcrResult()  # type: ignore[method-assign]
    results = engine.recognize_pdf(pdf_3_pages, dpi=72)
    assert len(results) == 3
    assert len(seen) == 3
