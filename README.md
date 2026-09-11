# OCR Engine

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
- **Routing języka**: heurystyka polskich diakrytyków + `langdetect` (etykieta ustawiana
  po rozpoznaniu; do routingu modelu użyj `force_language`).
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
```

### CLI

```bash
ocr recognize dokument.png                 # tekst
ocr recognize dokument.png --lang pl --json # JSON z bboxami
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
- Routing języka wybiera model PRZED rozpoznaniem — bez `force_language` domyślnie EN
  (PL trafia na model EN; diakrytyki naprawia korekta Bielik albo fine-tune PL).
- Brak obsługi układów wielokolumnowych / tabel (sortowanie czytania uproszczone).
- Detektor OpenCV (fallback) jest prostszy od CRAFT — dobry do dokumentów/skanów,
  słabszy do tekstu w naturze i złożonych tła.
- Alternatywa bez treningu: PaddleOCR-VL + LoRA RysOCR (lepsza polska diakrytyka od ręki).
