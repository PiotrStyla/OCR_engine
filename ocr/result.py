"""Struktury danych wynikowych silnika OCR."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

Language = Literal["pl", "en", "unknown"]


@dataclass(frozen=True)
class BBox:
    """Prostokąt otaczający w współrzędnych pikselowych obrazu źródłowego."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def expand(self, pad: int, max_w: int, max_h: int) -> "BBox":
        """Zwraca BBox powiększony o `pad` pikseli z obcięciem do granic obrazu."""
        return BBox(
            max(0, self.x1 - pad),
            max(0, self.y1 - pad),
            min(max_w, self.x2 + pad),
            min(max_h, self.y2 + pad),
        )

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class TextLine:
    """Pojedyncza rozpoznana linia tekstu."""

    text: str
    bbox: BBox
    language: Language
    confidence: float  # NaN = backend nie raportuje pewności (np. VLM)

    def __post_init__(self) -> None:
        if math.isnan(self.confidence):
            return
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence poza zakresem [0,1]: {self.confidence}")


@dataclass
class OcrResult:
    """Wynik działania silnika OCR dla całego obrazu."""

    lines: list[TextLine] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)  # (width, height)

    @property
    def text(self) -> str:
        """Pełny tekst złączony w porządku czytania (linia po linii)."""
        return "\n".join(line.text for line in self.lines if line.text)

    @property
    def full_text(self) -> str:
        return self.text

    @property
    def languages(self) -> list[Language]:
        """Unikalne języki linii, w kolejności występowania."""
        return list(dict.fromkeys(line.language for line in self.lines))

    def to_dict(self, confidence_threshold: float = 0.0) -> dict:
        """Serializacja do dict. Przy confidence_threshold > 0 linie poniżej
        progu dostają flagę 'low_confidence: true'."""
        return {
            "text": self.text,
            "image_size": self.image_size,
            "lines": [
                {
                    "text": line.text,
                    "bbox": list(line.bbox.to_tuple()),
                    "language": line.language,
                    # NaN (backend bez confidence) → null w JSON
                    "confidence": (None if math.isnan(line.confidence)
                                   else line.confidence),
                    **({"low_confidence": True}
                       if confidence_threshold > 0
                       and not math.isnan(line.confidence)
                       and line.confidence < confidence_threshold else {}),
                }
                for line in self.lines
            ],
        }
