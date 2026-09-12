"""Silnik OCR: CRAFT (detekcja) + TrOCR (rozpoznawanie) z wsparciem PL/EN.

Szybki start:

    from ocr import recognize
    result = recognize("dokument.png")
    print(result.text)

Lub z kontrolą cyklu życia (lepsze dla wielu obrazów):

    from ocr import OcrEngine, OcrConfig
    with OcrEngine(OcrConfig(force_language="pl")) as engine:
        result = engine.recognize("dokument.png")
"""

from __future__ import annotations

from .config import OcrConfig
from .result import BBox, Language, OcrResult, TextLine


def __getattr__(name):
    # Remote page parsing and preflight must not require local ML dependencies.
    if name in {'OcrEngine', 'recognize'}:
        from .pipeline import OcrEngine, recognize
        globals().update(OcrEngine=OcrEngine, recognize=recognize)
        return globals()[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = [
    "OcrConfig",
    "OcrEngine",
    "recognize",
    "OcrResult",
    "TextLine",
    "BBox",
    "Language",
]

__version__ = "0.1.0"
