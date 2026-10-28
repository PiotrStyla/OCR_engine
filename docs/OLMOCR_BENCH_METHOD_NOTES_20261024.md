# olmOCR-bench — metodyka i wnioski dla PolOCRBench (notatki, 2026-10-24)

Źródło: publiczny zbiór `pcuenq/dots.mocr-mlx-evals` (licencja MIT) — scoreboard
benchmarku olmOCR-bench z pełnym logiem pomiaru. olmOCR-bench to benchmark, na
którym wzoruje się opis zadania PolOCRBench (OCRBench/OmniDocBench-style).

## Jak ocenia olmOCR-bench

Zamiast metryk ciągłych (CER) testy sprawdzają **konkretne własności tekstu**
(każda próba: zdana/niezdana), a wynik to „pass ratio" z przedziałem ufności 95%:

| Typ testu | Co sprawdza |
| --- | --- |
| `baseline` | czy podstawowy tekst w ogóle występuje w wyjściu |
| `present` / `absent` | czy wskazany element (nagłówek, stopka, wzór) się pojawił / NIE pojawił |
| `order` | **kolejność czytania** — czy ciągi tekstu są w poprawnej kolejności |
| `math` | poprawność wzorów (renderowanie KaTeX) |
| `table` | zawartość tabel |

Kategorie trudności: `arxiv_math`, `headers_footers`, `long_tiny_text`,
`multi_column`, `old_scans`, `old_scans_math`, `tables`.

## Wyniki referencyjne (dots.mocr, z logu)

| Wariant | ogółem | order | present | absent | math | table | baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | 82,9% | 76,5% | 67,0% | 91,3% | 86,2% | 88,3% | 99,3% |
| 4-bit | 82,5% | 76,3% | 66,6% | 90,3% | 84,9% | 88,2% | 99,5% |
| 8-bit | 82,6% | 76,3% | 65,7% | 91,1% | 86,0% | 88,8% | 99,2% |

(8 413 testów; warianty 5/6-bit i mxfp4/mxfp8/nvfp4 w tym samym przedziale.)

## Wnioski dla PolOCRBench

1. **Kolejność czytania (`order`) to najsłabsze ogniwo modeli światowych**
   (~76% przy ~99% za samą obecność tekstu). CER nie wykrywa błędów kolejności —
   warto dodać testy typu `order`/`present`/`absent` jako **uzupełnienie** metryk
   ciągłych (CER/TEDS/F1), zamiast je zastępować.
2. **Kwantyzacja jest praktycznie bezstratna** dla OCR: warianty 4–8-bit
   mieszczą się w ±0,5% punktu od pełnej precyzji. Argument za kwantyzacją
   4-bit w planie małych modeli do wdrożeń.
3. Kandydaci na baseline'y (z tego samego ekosystemu): dots.mocr (~83%),
   PaddleOCR-VL (Apache-2.0), baidu/Unlimited-OCR (MIT) — do pomiarów obok
   Qwen3-VL/Donut/Nougat.

## Źródła

- `huggingface.co/datasets/pcuenq/dots.mocr-mlx-evals` (MIT) — log
  `logs/olmocr_bench_all_models.txt` z pełnym scoreboardem.
- `huggingface.co/PaddlePaddle/PaddleOCR-VL` (Apache-2.0).
