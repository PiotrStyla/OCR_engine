"""CLI silnika OCR.

Komendy:
  ocr recognize <image> [--lang pl|en] [--json] [--correct] ...
  ocr check-fabryka [--fabryka-model MODEL]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import OcrConfig
from .pipeline import OcrEngine
from .postprocess import TextCorrector


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocr", description="Silnik OCR (CRAFT + TrOCR)")
    sub = parser.add_subparsers(dest="command", required=True)

    # recognize
    rec = sub.add_parser("recognize", help="Rozpoznaj tekst z obrazu lub PDF")
    rec.add_argument("image", type=Path, help="Ścieżka do obrazu lub PDF")
    rec.add_argument("--lang", choices=["pl", "en"], default=None, help="Wymuś język")
    rec.add_argument("--json", action="store_true", help="Wyjście JSON")
    rec.add_argument("--no-deskew", action="store_true", help="Pomiń korektę pochylenia")
    rec.add_argument("--pages", default=None,
                     help="Zakres stron PDF, np. '1-3,5' (domyślnie: wszystkie)")
    rec.add_argument("--dpi", type=int, default=300, help="DPI renderowania stron PDF")
    rec.add_argument("--correct", action=argparse.BooleanOptionalAction, default=None,
                     help="Korekta tekstu przez Fabryka API (wymaga FABRYKA_API_KEY)")
    rec.add_argument("--correct-low-only", action=argparse.BooleanOptionalAction, default=None,
                     help="Korekta tylko linii z confidence < --confidence-threshold")
    rec.add_argument("--confidence-threshold", type=float, default=None,
                     help="Próg niskiego confidence (flaga w JSON / selektywna korekta)")
    rec.add_argument("--fabryka-model", default=None,
                     help="Model Fabryka do korekty (domyślnie: bielik-11b-v3)")
    rec.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    rec.add_argument("--backend", choices=["trocr", "paddlevl", "kraken"], default=None,
                     help="Backend rozpoznawania: trocr, paddlevl (PaddleOCR-VL VLM), lub kraken (maszynopis/historyczne)")

    # check-fabryka
    chk = sub.add_parser("check-fabryka", help="Sprawdź połączenie z Fabryka API")
    chk.add_argument("--fabryka-model", default=None,
                     help="Model do przetestowania (domyślnie: bielik-11b-v3)")
    return parser


def _cmd_recognize(args) -> int:
    if not args.image.exists():
        print(f"Plik nie istnieje: {args.image}", file=sys.stderr)
        return 2
    config = OcrConfig.from_env()
    for arg, field in (("lang", "force_language"), ("correct", "correct_text"),
                       ("correct_low_only", "correct_low_confidence_only"),
                       ("confidence_threshold", "confidence_threshold"),
                       ("fabryka_model", "fabryka_model"), ("device", "device"),
                       ("backend", "recognizer_backend")):
        value = getattr(args, arg)
        if value is not None:
            setattr(config, field, value)
    if args.no_deskew:
        config.deskew = False
    if args.correct_low_only:
        config.correct_text = True
    if config.correct_low_confidence_only and config.confidence_threshold <= 0:
        raise ValueError("Selective correction requires confidence-threshold > 0")
    with OcrEngine(config) as engine:
        if args.image.suffix.lower() == ".pdf":
            results = engine.recognize_pdf(args.image, pages=args.pages, dpi=args.dpi)
        else:
            results = [engine.recognize(args.image)]

    if args.json:
        out = [r.to_dict(config.confidence_threshold) for r in results]
        print(json.dumps(out[0] if len(out) == 1 else out, ensure_ascii=False, indent=2))
    else:
        for i, r in enumerate(results):
            if len(results) > 1:
                print(f"--- strona {r.page_number or i + 1} ---")
            print(r.text)
    return 0


def _cmd_check_fabryka(args) -> int:
    config = OcrConfig.from_env()
    if args.fabryka_model is not None:
        config.fabryka_model = args.fabryka_model
    corrector = TextCorrector(config)

    print(f"Endpoint: {corrector.base_url}")
    print(f"Model:    {config.fabryka_model}")
    key = corrector.api_key
    if key:
        # pokaż tylko prefix ze względów bezpieczeństwa
        print(f"Klucz:    {key[:12]}...{key[-4:]} (długość: {len(key)})")
    else:
        print("Klucz:    BRAK (ustaw FABRYKA_API_KEY)")
        return 1

    print()
    ok, msg = corrector.check_connection()
    if ok:
        print(f"[OK] {msg}")
        return 0
    else:
        print(f"[FAIL] {msg}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "recognize":
        return _cmd_recognize(args)
    if args.command == "check-fabryka":
        return _cmd_check_fabryka(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
