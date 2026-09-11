"""Punkt wejścia `python -m ocr` — deleguje do CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
