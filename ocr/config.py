"""Konfiguracja silnika OCR: nazwy modeli, progi, urządzenie."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

Device = Literal["auto", "cpu", "cuda"]


def _resolve_device(device: Device) -> str:
    if device == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class OcrConfig:
    """Konfiguracja pipeline'u OCR.

    Pola modeli pozwalają podmienić checkpointi (np. na własny fine-tune PL).
    """

    # Detektor (CRAFT)
    detector_model: str = "craft_base"
    text_threshold: float = 0.7
    link_threshold: float = 0.4
    low_text_threshold: float = 0.4
    # dopasowanie do poziomu przed rozpoznawaniem
    deskew: bool = True

    # Rozpoznawacz (TrOCR)
    recognizer_en: str = "microsoft/trocr-base-printed"
    recognizer_pl: str = "ocr/trocr-pl-base"  # lokalny fine-tune (po treningu)
    recognizer_fallback: str = "microsoft/trocr-base-printed"
    # jeśli model PL nie istnieje, użyj fallback (EN) zamiast rzucać błędem
    use_fallback_if_pl_missing: bool = True

    # Routing języka
    force_language: str | None = None  # "pl" | "en" | None (auto)
    min_line_chars_for_lang_detect: int = 5

    # Urządzenie
    device: Device = field(default="auto")

    # Przetwarzanie
    line_padding: int = 5  # piksele paddingu wokół linii przed rozpoznaniem
    batch_size: int = 8  # batch rozpoznawania linii
    min_line_area: int = 20  # minimalna powierzchnia bboxa linii (px^2)

    # Postprocessing: korekta tekstu przez Fabryka API (Bielik)
    correct_text: bool = False  # włącz korektę po OCR
    fabryka_api_key: str | None = None  # lub z env FABRYKA_API_KEY
    fabryka_base_url: str = "auto"  # auto-detekcja po prefixie klucza
    fabryka_model: str = "bielik-11b-v3"  # polski model do korekty
    fabryka_timeout: float = 30.0  # sekundy

    def resolved_device(self) -> str:
        return _resolve_device(self.device)

    @classmethod
    def from_env(cls) -> "OcrConfig":
        """Wczytuje nadpisania z zmiennych środowiskowych OCR_* / FABRYKA_*."""
        def env(key: str, default: str) -> str:
            return os.environ.get(key, default)

        def env_bool(key: str, default: bool) -> bool:
            v = os.environ.get(key)
            if v is None:
                return default
            return v.lower() in ("1", "true", "yes", "on")

        return cls(
            recognizer_en=env("OCR_RECOGNIZER_EN", "microsoft/trocr-base-printed"),
            recognizer_pl=env("OCR_RECOGNIZER_PL", "ocr/trocr-pl-base"),
            recognizer_fallback=env("OCR_RECOGNIZER_FALLBACK", "microsoft/trocr-base-printed"),
            force_language=os.environ.get("OCR_FORCE_LANGUAGE"),
            device=env("OCR_DEVICE", "auto"),  # type: ignore[arg-type]
            correct_text=env_bool("OCR_CORRECT_TEXT", False),
            fabryka_api_key=os.environ.get("FABRYKA_API_KEY"),
            fabryka_base_url=env("FABRYKA_BASE_URL", "auto"),
            fabryka_model=env("FABRYKA_MODEL", "bielik-11b-v3"),
        )
