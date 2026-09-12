# OCR Engine

> Aktualizacja po audycie (2026-09-12): zobacz [plan CPU i zdalnych testów](docs/CPU_REMOTE_PLAN.md).
> CLI respektuje ENV, a jawne flagi mają pierwszeństwo. Tablice wejściowe muszą być RGB uint8.
> Wyniki zawierają źródłowy numer strony i geometrię po odwrotnym mapowaniu deskew.
> Korekta zachowuje `raw_text`/`raw_confidence`; zmieniony tekst nie dziedziczy pewności OCR.
> Niestandardowe checkpointy Paddle są odrzucane, dopóki nie ma ich jawnej integracji.
>
> [Pierwsze pomiary CPU i Fabryki](docs/EXPERIMENTS_2026-09-12.md): Tesseract stanowi
> osobny baseline; na małym teście korekta tekstowa zwiększała łączny CER.
> Wyniki nie są potwierdzeniem SOTA. Ucięte odpowiedzi korektora są odrzucane.

Silnik OCR w Pythonie: **detekcja linii** (CRAFT lub OpenCV) + **TrOCR** (rozpoznawanie),
z wsparciem języków **PL/EN**, routingiem języka per linia i **korektą tekstu przez
Fabryka API** (Bielik).

## Architektura

```
obraz → preprocess (deskew) → detekcja linii → routing języka (PL/EN)
      → TrOCR (rozpoznawanie) → [korekta przez Fabryka/Bielik] → OcrResult
```

- **Detekcja**: auto-wybór — **CRAFT** (`craft-text-detector`) gdy dostępny,
  w przeciwnym razie **fallback OpenCV** (morfologia + kontury; `ocr/opencv_detector.py`).
- **Rozpoznawanie**: TrOCR (`microsoft/trocr-base-printed` dla EN; lokalny fine-tune QLoRA dla PL).
- **Korekta tekstu**: [Fabryka AI](https://fabryka.ai) — Bielik (polski LLM) naprawia błędy
  OCR (diakrytyki, pocięte słowa, interpunkcja). Opcjonalna, wymaga klucza API.
- **Routing języka**: heurystyka polskich diakrytyków + `langdetect`. Bez
  `force_language` linie idą najpierw modelem EN, a te wykryte jako PL są
  **re-rozpoznawane modelem PL** w drugim przebiegu (gdy model PL istnieje).
- **Porządek czytania**: grupowanie linii w poziome pasy (top→bottom), wewnątrz left→right.

> **Uwaga PL:** oficjalne modele TrOCR są tylko angielskie. Polskie diakrytyki
> wymagają fine-tunu — patrz [`training/README.md`](training/README.md) (QLoRA, wg wskazówek
> [Slayer](https://slayer.fabryka.ai/trening)). Alternatywa bez treningu: włączyć korektę
> tekstu przez Fabryka API (Bielik) — patrz niżej.

## Instalacja

```bash
pip install -e .
# dev (testy):
pip install -e ".[dev]"
# detektor CRAFT (wymaga Py <3.11 — stary pin opencv):
pip install -e ".[craft]"
# korekta przez Fabryka API:
pip install -e ".[correct]"
# trening PL (GPU):
pip install -e ".[train]"
```

> Bez `craft-text-detector` działa automatyczny detektor OpenCV — wystarczy do
> dokumentów i skanów na jasnym tle.

## Użycie

### Biblioteka

```python
from ocr import recognize, OcrEngine, OcrConfig

# jednorazowo
result = recognize("dokument.png")
print(result.text)
for line in result.lines:
    print(line.text, line.bbox, line.language, line.confidence)

# wiele obrazów (współdzielenie modeli)
with OcrEngine(OcrConfig(force_language="pl")) as engine:
    for path in paths:
        print(engine.recognize(path).text)

# PDF — lista wyników per strona
with OcrEngine() as engine:
    pages = engine.recognize_pdf("dokument.pdf", pages="1-3", dpi=300)
    for i, page in enumerate(pages, 1):
        print(f"--- strona {i} ---\n{page.text}")
```

### CLI

```bash
ocr recognize dokument.png                 # tekst
ocr recognize dokument.png --lang pl --json # JSON z bboxami
ocr recognize dokument.pdf                  # PDF — wszystkie strony
ocr recognize dokument.pdf --pages 1-3,5    # wybrane strony
ocr recognize dokument.png --correct        # korekta tekstu przez Fabryka/Bielik
ocr recognize dokument.png --no-deskew
ocr recognize dokument.png --device cpu
ocr check-fabryka                           # sprawdź połączenie z Fabryka API
```

### Korekta tekstu przez Fabryka API (Bielik)

Korekta naprawia typowe błędy OCR (zamiana diakrytyków ą→a, ł→l, ę→e, pocięte słowa)
przy użyciu polskiego modelu LLM Bielik przez [Fabryka AI](https://fabryka.ai).

Endpoint jest wybierany automatycznie na podstawie prefixu klucza:
- `fab_live_...` → `https://fabryka.ai/v1` (lab)
- `sk-fab-...` → `https://router.fabryka.ai/v1` (router)

```bash
# 1. Pobierz klucz API z https://fabryka.ai (sekcja "Get API key")
# 2. Ustaw zmienną środowiskową:
export FABRYKA_API_KEY=fab_live_...   # lub sk-fab-...

# 3. Sprawdź połączenie:
ocr check-fabryka

# 4. Rozpoznaj z korektą:
ocr recognize dokument.png --correct
```

Lub w Pythonie:

```python
from ocr import OcrEngine, OcrConfig

config = OcrConfig(correct_text=True)  # wymaga FABRYKA_API_KEY w env
with OcrEngine(config) as engine:
    result = engine.recognize("dokument.png")
    print(result.text)  # tekst po korekcie
```

Wymaga: `pip install ocr-engine[correct]` (pakiet `openai`). Brak klucza → korekta
pominięta z ostrzeżeniem (graceful degradation). Błąd API → zwraca oryginalny tekst
z retry (429/502/503/504 z exponential backoff).

### Próg pewności (confidence)

Każda linia ma `confidence` ∈ [0,1] z rozpoznawania TrOCR. Próg pozwala
flagować niskopewne linie i ograniczyć korektę tylko do nich (oszczędność API):

```bash
# oznacz linie z confidence < 0.7 flagą "low_confidence" w JSON:
ocr recognize dokument.png --json --confidence-threshold 0.7

# korekta Fabryka TYLKO dla niskopewnych linii:
ocr recognize dokument.png --correct-low-only --confidence-threshold 0.7
```

W selektywnej korekcie do API trafiają wyłącznie linie poniżej progu; jeśli
korektor zwróci inną liczbę linii niż wysłano, oryginał zostaje zachowany.

### Konfiguracja (zmienne środowiskowe)

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `OCR_FORCE_LANGUAGE` | (auto) | `pl`/`en` — wymuś język |
| `OCR_CORRECT_TEXT` | `false` | `true`/`false` — włącz korektę przez Fabryka |
| `FABRYKA_API_KEY` | (brak) | klucz API Fabryka (https://fabryka.ai) |
| `FABRYKA_MODEL` | `bielik-11b-v3` | model do korekty |
| `FABRYKA_BASE_URL` | `auto` | `auto` lub pełny URL (np. `https://router.fabryka.ai/v1`) |
| `OCR_RECOGNIZER_EN` | `microsoft/trocr-base-printed` | model TrOCR EN |
| `OCR_RECOGNIZER_PL` | `ocr/trocr-pl-base` | lokalny fine-tune PL |
| `OCR_DEVICE` | `auto` | `auto`/`cpu`/`cuda` |
| `OCR_CONFIDENCE_THRESHOLD` | `0.0` | próg flagowania niskopewnych linii |
| `OCR_CORRECT_LOW_ONLY` | `false` | korekta tylko linii poniżej progu |
| `OCR_RECOGNIZER_BACKEND` | `trocr` | `trocr` lub `paddlevl` (PaddleOCR-VL VLM) |

## Backend PaddleOCR-VL (opcjonalny, SOTA)

Alternatywny backend rozpoznawania: [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
— model VLM 0.9B, 109 języków (w tym polski), #1 na OmniDocBench. Nie wymaga
fine-tuningu ani detektora per język.

```bash
pip install -e ".[vlm]"   # paddlepaddle + paddleocr[doc-parser]

ocr recognize dokument.png --backend paddlevl
python -m training.evaluate --data ./data/pl_lines_val --backend paddlevl
```

Uwagi: backend VLM nie raportuje `confidence` (w JSON: `null`), rozpoznaje
linie po jednej (wolniejsze na CPU). Porównanie head-to-head z TrOCR:
`python -m training.evaluate --data <zbiór> --backend trocr|paddlevl`.

## Testy

```bash
pytest
```

## Struktura

```
ocr/        # biblioteka (config, preprocess, detector, recognizer, pipeline, lang, cli)
training/   # fine-tuning polskiego TrOCR (GPU)
tests/      # testy jednostkowe (nie wymagają modeli ML)
```

## Ograniczenia / roadmapa

- Fine-tune PL wymaga danych + GPU (patrz `training/`). Alternatywa bez treningu:
  korekta tekstu przez Fabryka API (Bielik) — włącz `--correct`.
- Routing języka: bez `force_language` linie startują modelem EN, a wykryte jako PL
  są re-rozpoznawane modelem PL (drugi przebieg, gdy model istnieje). Gdy modelu PL
  brak — zostaje wynik EN (diakrytyki naprawia korekta Bielik).
- Brak obsługi układów wielokolumnowych / tabel (sortowanie czytania uproszczone).
- Detektor OpenCV (fallback) jest prostszy od CRAFT — dobry do dokumentów/skanów,
  słabszy do tekstu w naturze i złożonych tła.
- Alternatywa bez treningu: PaddleOCR-VL + LoRA RysOCR (lepsza polska diakrytyka od ręki).

Pilot rzeczywistych skanów: [wyniki i odtworzenie](docs/PUBLIC_SCAN_PILOT.md).
