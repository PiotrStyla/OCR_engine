"""Testy ewaluacji CER/WER (z mockowanym backendem — bez modelu)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from training.evaluate import EvalReport, evaluate


class _FakeBackend:
    """Udaje _TrOCRBackend — zwraca stałe hipotezy bez modelu."""

    instances = []

    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        _FakeBackend.instances.append(self)

    def recognize(self, images, batch_size):
        # celowo "zły" wynik — CER > 0
        return [("BLEDNY TEKST", 0.5) for _ in images]


@pytest.fixture
def pairs_dir(tmp_path):
    d = tmp_path / "pairs"
    d.mkdir()
    for i, text in enumerate(["Zażółć gęślą jaźń", "Ala ma kota"]):
        Image.new("RGB", (200, 40), "white").save(d / f"{i:05d}.png")
        (d / f"{i:05d}.txt").write_text(text, encoding="utf-8")
    return d


def test_evaluate_computes_metrics(monkeypatch, pairs_dir):
    # evaluate robi lokalny import _TrOCRBackend — patchuj atrybut modułu
    import ocr.recognizer
    monkeypatch.setattr(ocr.recognizer, "_TrOCRBackend", _FakeBackend)

    report = evaluate(pairs_dir, "fake/model", config=None)
    assert report.n_lines == 2
    assert report.cer > 0.0
    assert report.wer > 0.0
    assert report.cer_corrected is None


def test_evaluate_limit(monkeypatch, pairs_dir):
    import ocr.recognizer
    monkeypatch.setattr(ocr.recognizer, "_TrOCRBackend", _FakeBackend)

    report = evaluate(pairs_dir, "fake/model", limit=1)
    assert report.n_lines == 1


def test_evaluate_empty_dir_raises(tmp_path):
    with pytest.raises(SystemExit):
        evaluate(tmp_path / "empty", "fake/model")


def test_report_summary_format():
    r = EvalReport(cer=0.5, wer=0.6, n_lines=10, cer_corrected=0.4, wer_corrected=0.5)
    s = r.summary()
    assert "CER" in s and "50.00%" in s
    assert "Δ" in s and "+0.1000" in s
