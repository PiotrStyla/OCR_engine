"""Ewaluacja modelu TrOCR na zbiorze linii (CER/WER).

Mierzy baseline (np. EN przed fine-tunem) i wynik po treningu — ta sama metryka,
ten sam zbiór. Opcjonalnie mierzy zysk z korekty tekstu przez Fabryka API (Bielik).

Uruchomienie:

    python -m training.evaluate --data ./data/pl_lines_val \
        --model microsoft/trocr-base-printed

    # po fine-tunie:
    python -m training.evaluate --data ./data/pl_lines_val \
        --model ./ocr/trocr-pl-base

    # z korektą Bielik (wymaga FABRYKA_API_KEY):
    python -m training.evaluate --data ./data/pl_lines_val \
        --model microsoft/trocr-base-printed --correct
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from ocr.config import OcrConfig
from ocr.postprocess import TextCorrector

from .dataset import load_pairs

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    cer: float
    wer: float
    n_lines: int
    cer_corrected: float | None = None
    wer_corrected: float | None = None

    def summary(self) -> str:
        lines = [
            f"linii:      {self.n_lines}",
            f"CER:        {self.cer:.4f}  ({self.cer * 100:.2f}%)",
            f"WER:        {self.wer:.4f}  ({self.wer * 100:.2f}%)",
        ]
        if self.cer_corrected is not None:
            d_cer = self.cer - self.cer_corrected
            d_wer = self.wer - (self.wer_corrected or 0.0)
            lines += [
                f"CER+korekta: {self.cer_corrected:.4f}  ({self.cer_corrected * 100:.2f}%)  Δ {d_cer:+.4f}",
                f"WER+korekta: {self.wer_corrected:.4f}  ({self.wer_corrected * 100:.2f}%)  Δ {d_wer:+.4f}",
            ]
        return "\n".join(lines)


def evaluate(
    data_dir: str | Path,
    model_name: str,
    batch_size: int = 8,
    limit: int | None = None,
    correct: bool = False,
    config: OcrConfig | None = None,
) -> EvalReport:
    """Uruchamia model na parach z `data_dir` i liczy CER/WER.

    `correct=True` dodatkowo mierzy metryki po korekcie Fabryka/Bielik.
    """
    from jiwer import cer, wer

    from ocr.recognizer import _TrOCRBackend

    samples = load_pairs(data_dir)
    if limit:
        samples = samples[:limit]
    if not samples:
        raise SystemExit(f"Brak danych w {data_dir}")

    device = (config or OcrConfig()).resolved_device()
    backend = _TrOCRBackend(model_name, device)

    refs = [s.text for s in samples]
    hyps = [t for t, _ in backend.recognize([s.image for s in samples], batch_size)]

    report = EvalReport(cer=cer(refs, hyps), wer=wer(refs, hyps), n_lines=len(refs))

    if correct:
        cfg = config or OcrConfig(correct_text=True)
        cfg.correct_text = True
        corrector = TextCorrector(cfg)
        if corrector.enabled:
            corrected_hyps = [corrector.correct(h) for h in hyps]
            report.cer_corrected = cer(refs, corrected_hyps)
            report.wer_corrected = wer(refs, corrected_hyps)
        else:
            logger.warning("Korekta żądana, ale brak FABRYKA_API_KEY — pomijam.")

    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ewaluacja TrOCR (CER/WER)")
    p.add_argument("--data", required=True, help="Katalog z parami .png/.txt")
    p.add_argument("--model", default="microsoft/trocr-base-printed")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--correct", action="store_true",
                   help="Zmierz też CER/WER po korekcie Fabryka/Bielik")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a = _parse_args()
    cfg = OcrConfig(device=a.device, correct_text=a.correct)
    report = evaluate(
        a.data, a.model, batch_size=a.batch_size, limit=a.limit,
        correct=a.correct, config=cfg,
    )
    print(report.summary())


if __name__ == "__main__":
    main()
