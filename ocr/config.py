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

    # Detektor (progi CRAFT; detektor wybierany automatycznie — OpenCV fallback)
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
    # backend rozpoznawania: "trocr" (detektor→TrOCR per język) lub
    # "paddlevl" (PaddleOCR-VL-0.9B, VLM wielojęzyczny, bez fine-tuningu)
    # lub "kraken" (end-to-end Kraken: segmentacja baseline + rozpoznawanie
    # .mlmodel, np. polish_nfd_9313.mlmodel z EHRI — najlepszy dla maszynopisu)
    # lub "auto" (heurystyka: jeśli OpenCV wykrywa mało linii → Kraken, inaczej TrOCR)
    recognizer_backend: str = "trocr"
    paddlevl_model: str = "PaddlePaddle/PaddleOCR-VL"
    # Kraken: model rozpoznawania (.mlmodel). Ścieżka lokalna lub HF repo+plik.
    # Domyślnie polski model EHRI (93,1% accuracy na polskim maszynopisie).
    kraken_model: str = "PiotrSty/ehri-dataset::models/polish_nfd_9313.mlmodel"
    # Kraken: binarization (nlbin) przed segmentacją + rozpoznawaniem.
    # UWAGA: pogarsza CER na modelu EHRI (11.67% → 20.26%) — model był
    # trenowany na surowych obrazach grayscale, nie na zbinaryzowanych.
    # Wyłączone domyślnie; włącz tylko dla bardzo wyblakłych skanów.
    kraken_binarize: bool = False
    # Auto-routing: gdy recognizer_backend="auto", użyj Kraken jeśli OpenCV
    # wykryje mniej niż auto_kraken_min_lines na auto_kraken_min_height pikseli
    # wysokości obrazu. Heurystyka: OpenCV gubi linie na maszynopisach.
    auto_kraken_min_lines: int = 5
    auto_kraken_min_height: int = 500

    # Routing języka
    force_language: str | None = None  # "pl" | "en" | None (auto)

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

    # Próg pewności: linie z confidence < progu są flagowane w JSON
    # i (opcjonalnie) korygowane wyłącznie przez Fabryka. 0.0 = wyłączone.
    confidence_threshold: float = 0.0
    correct_low_confidence_only: bool = False  # korekta tylko linii o niskim confidence

    # Jev (TypeSafe AI) — model decyzyjny do scoringu, routingu i walidacji
    jev_api_key: str | None = None  # lub z env TYPESAFE_API_KEY
    jev_model: str = "jev-latest"
    jev_timeout: float = 10.0  # sekundy
    # Jev: routing backendu (wybiera trocr/kraken/paddlevl na podstawie statystyk)
    jev_route_backend: bool = False
    # Jev: scoring jakości linii OCR (flaguje linie do korekty)
    jev_score_lines: bool = False
    # Jev: walidacja korekty Fabryka (sprawdza czy poprawiony tekst jest lepszy)
    jev_validate_correction: bool = False
    # Próg jakości Jev (0-2): linie ze score < progu są flagowane do korekty
    jev_quality_threshold: float = 1.0

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

        def env_float(key: str, default: float) -> float:
            v = os.environ.get(key)
            if v is None:
                return default
            try:
                return float(v)
            except ValueError:
                return default

        return cls(
            recognizer_en=env("OCR_RECOGNIZER_EN", "microsoft/trocr-base-printed"),
            recognizer_pl=env("OCR_RECOGNIZER_PL", "ocr/trocr-pl-base"),
            recognizer_fallback=env("OCR_RECOGNIZER_FALLBACK", "microsoft/trocr-base-printed"),
            recognizer_backend=env("OCR_RECOGNIZER_BACKEND", "trocr"),
            paddlevl_model=env("OCR_PADDLEVL_MODEL", "PaddlePaddle/PaddleOCR-VL"),
            kraken_model=env("OCR_KRAKEN_MODEL", "PiotrSty/ehri-dataset::models/polish_nfd_9313.mlmodel"),
            kraken_binarize=env_bool("OCR_KRAKEN_BINARIZE", False),
            force_language=os.environ.get("OCR_FORCE_LANGUAGE"),
            device=env("OCR_DEVICE", "auto"),  # type: ignore[arg-type]
            correct_text=env_bool("OCR_CORRECT_TEXT", False),
            fabryka_api_key=os.environ.get("FABRYKA_API_KEY"),
            fabryka_base_url=env("FABRYKA_BASE_URL", "auto"),
            fabryka_model=env("FABRYKA_MODEL", "bielik-11b-v3"),
            confidence_threshold=env_float("OCR_CONFIDENCE_THRESHOLD", 0.0),
            correct_low_confidence_only=env_bool("OCR_CORRECT_LOW_ONLY", False),
            jev_api_key=os.environ.get("TYPESAFE_API_KEY"),
            jev_model=env("TYPESAFE_MODEL", "jev-latest"),
            jev_timeout=env_float("TYPESAFE_TIMEOUT", 10.0),
            jev_route_backend=env_bool("OCR_JEV_ROUTE_BACKEND", False),
            jev_score_lines=env_bool("OCR_JEV_SCORE_LINES", False),
            jev_validate_correction=env_bool("OCR_JEV_VALIDATE", False),
            jev_quality_threshold=env_float("OCR_JEV_QUALITY_THRESHOLD", 1.0),
        )
