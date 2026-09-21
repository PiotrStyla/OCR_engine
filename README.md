# OCR Engine / PolOCRBench

Silnik OCR dla polskich dokumentów oraz rozwijane zaplecze **PolOCRBench**:
publicznego benchmarku transkrypcji całych stron, ekstrakcji tabel i informacji
z dokumentów. Repozytorium zawiera backendy OCR, narzędzia treningowe,
ewaluatory i artefakty eksperymentów. **Nie jest jeszcze ukończonym benchmarkiem
ani potwierdzonym silnikiem SOTA.**

## Aktualny stan: 21 września 2026

- **Podzadanie A, transkrypcja:** zamrożony historyczny podzbiór IMPACT,
  36 stron testowych z 3 kolekcji oraz pula 2531 regionów treningowych.
- **Odtwarzalne dane:** importer przypiętej paczki Hugging Face sprawdza SHA-256,
  rozdzielenie kolekcji i brak wspólnych hashy obrazów train/test. Osobne kopie
  PNG zachowują piksele oryginalnych TIFF-ów i mają przenośne ścieżki.
- **Ewaluator v1.1:** CER/WER oraz przybliżona ocena struktury Markdown;
  brakujące i błędne odpowiedzi dostają zero punktów za strukturę.
- **Baseline CPU:** zapisane predykcje wszystkich stron, metadane silnika,
  hashe wag, wyniki obu protokołów i sumy kontrolne artefaktów.
- **Weryfikacja:** 17 testów importera, konwertera i ewaluatorów przeszło.
  To testy tego zakresu zmian, nie deklaracja uruchomienia całego zestawu testów.

### Wynik baseline'u CPU

Tesseract.js 7.0.0, `pol+eng`, OEM 1, PSM 3, jeden worker CPU:

| Zbiór | Strony | CER micro | WER micro |
| --- | ---: | ---: | ---: |
| IMPACT historyczny, całość | 36 | **34,94%** | **81,23%** |
| NA2_FT | 15 | 27,35% | 73,22% |
| Nowiny_z_Rakuz_FT | 15 | 43,93% | 87,78% |
| Powodzenia_FT | 6 | 34,83% | 87,94% |

Niższe CER/WER oznacza mniej błędów. Odczyt trwał 392,10 s; wszystkie strony
zwróciły tekst, ale nie oznacza to poprawnej transkrypcji. Jest to nowy pomiar,
a nie odtworzenie wcześniejszej konfiguracji natywnego Tesseract `pol tessdata_best`.

**Ograniczenie referencji:** 86 znaków zastępczych U+FFFD na 23 stronach oraz
624 znaki prywatnego zakresu Unicode na 33 stronach wymagają przeglądu adnotacji.
Zamrożonych referencji nie poprawiano na podstawie predykcji. Sam brak wspólnych
hashy nie wyklucza podobnych skanów ani obecności dokumentów w pretreningu modeli.

- [Wyniki, surowe predykcje i metadane](experiments/2026-09-19/impact-tesseractjs-png/README.md)
- [Odtworzenie benchmarku i baseline'u](docs/POLOCRBENCH_REPRODUCTION.md)
- [Zmiany protokołu ewaluacji](docs/POLOCRBENCH_EVALUATOR_2026-09-19.md)
- [Zamrożone manifesty PolOCRBench](benchmarks/polocrbench/README.md)
- [Lokalny panel audytu adnotacji](tools/annotation-review/README.md): skan obok
  transkrypcji, kolejka podejrzanych znaków i eksport historii propozycji zmian.
  Panel nie modyfikuje zamrożonych referencji.
- [Uzgadnianie recenzji i nowa wersja manifestu](tools/annotation-review/README.md#build-a-reviewed-candidate):
  zgodność dwóch recenzentów, raport konfliktów oraz pełna historia zmian.
  Wynik jest kandydatem do wydania, bez automatycznej publikacji.
- [Runner Krakena na GPU i walidator zgłoszeń A](docs/KRAKEN_REPRODUCIBLE_BASELINE.md):
  jawne hashe obu modeli, predykcje każdej strony i metadane środowiska.
  Test CUDA na Kaggle Tesla T4 zakończył się: 36/36 stron, bez błędów wykonania.
- [Notebook Kaggle: odtwarzalny baseline Krakena](training/kaggle_polocrbench_kraken_reproducible.ipynb):
  włącz Internet i GPU T4, uruchom Run All, pobierz ZIP wyników. Notebook wykonuje
  smoke-test jednej strony przed pełnym pomiarem 36 stron; nie trenuje modelu.

### Wynik Krakena i diagnostyka GPU

Kraken 7.1.1 z recognizerem i segmenterem fine-tunowanymi na EHRI osiągnął
**CER 79,56% / WER 107,61%**, wobec **34,94% / 81,23%** dla Tesseracta.
Referencje i hashe zdekodowanych pikseli są zgodne między przebiegami.
WER może przekraczać 100% przez nadmiarowe słowa. To wynik konkretnej
konfiguracji, nie ocena wszystkich modeli Krakena.

- [Raport przebiegu Kaggle](docs/KRAKEN_KAGGLE_RESULT_20260921.md).
- [Diagnoza segmentacji i wycinków](docs/KRAKEN_DIAGNOSTICS_20260921.md).
- [Kontrola wejścia recognizera i alfabetu](docs/KRAKEN_INPUT_CHECK_20260921.md).
- [Notebook diagnostyczny v2](training/kaggle_kraken_diagnostics.ipynb)
  oraz [pełny kod komórki](training/kaggle_kraken_diagnostics.py): uruchom kod
  jako jedną nową komórkę w tej samej sesji Kaggle po zakończonym baseline.
  Nie uruchamiaj ponownie Run All. Wymagane są zachowane obrazy i wyniki
  w `/kaggle/working/polocrbench-kraken-*/`. Wynik: `kraken-input-check-*.zip`.

Diagnostyka trzech stron tytułowych potwierdza błędy segmentacji i rozpoznawania.
175 rzeczywistych wycinków zgadza się z wcześniejszym eksportem; sprawdzone
podglądy po normalizacji zachowują czytelny tekst. Alfabet modelu nie obejmuje
części znaków historycznego druku. Nie wykluczono problemów checkpointu ani
całej ścieżki inferencji. Następny krok to kontrola na dokumentach EHRI, nie
kolejny trening na podstawie wyników zamrożonego testu.

### Lokalne próby gazet

- `training/sample_us_pd_newspapers.py`: mała próbka tekstowego datasetu,
  zachowanie surowego OCR, diagnostyka Unicode i pochodzenie danych.
- `training/probe_newspaper_correction.py`: ograniczona próba dwóch stron
  przez OpenRouter, domyślnie dry-run; wykonanie wymaga `--execute` i klucza.
  Używa płatnego modelu `openai/gpt-4o-mini`, nie wariantu `:free`.
- `training/segment_newspaper_columns.py`: podział według ręcznie określonych
  granic kolumn, wycinki bez zmiany pikseli i diagnostyka separatorów OpenCV.
  Nie jest automatyczną segmentacją artykułów.

Lokalne skany, wycinki i odpowiedzi API pozostają poza repozytorium (`data/`).

### Zakres docelowy

Planowane podzadania: **A** transkrypcja do Markdown, **B** tabele do HTML
z oceną TEDS, **C** pola dokumentu do JSON z oceną field-level F1.
Podzadania B/C, pełny zbiór współczesnych dokumentów i pisma ręcznego, ukryty
Test B oraz publiczny leaderboard wymagają dalszej implementacji i anotacji.
Docelowe tracki to constrained, open i zero-shot/API; nie są jeszcze wdrożone.

## Biblioteka OCR

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
  Opcjonalnie **Kraken** (baseline segmentation + `.mlmodel`) dla maszynopisów/historycznych.
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
# backend Kraken (historyczne dokumenty / maszynopis):
pip install -e ".[kraken]"
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
| `OCR_RECOGNIZER_BACKEND` | `trocr` | `trocr`, `paddlevl` (PaddleOCR-VL VLM), lub `kraken` |
| `OCR_KRAKEN_MODEL` | `PiotrSty/ehri-dataset::models/polish_nfd_9313.mlmodel` | model Kraken (.mlmodel) |
| `OCR_KRAKEN_BINARIZE` | `false` | `true`/`false` — binarization nlbin przed Kraken |

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

## Backend Kraken (opcjonalny, historyczne dokumenty / maszynopis)

[Kraken](https://kraken.re/) — system OCR/HTR zoptymalizowany pod dokumenty
historyczne i maszynopisy. Używa **trainable baseline segmentation** (sieć
neuronowa wykrywająca linie-bazy) oraz modeli rozpoznawania `.mlmodel` (CTC).

**Kiedy używać:** maszynopisy, dokumenty historyczne, wyblakłe skany — gdzie
OpenCV+TrOCR zawodzi przez słabą segmentację. Dla czystego druku TrOCR
pozostaje lepszy.

**Model domyślny:** `polish_nfd_9313.mlmodel` z EHRI (93,1% accuracy na polskim
maszynopisie).

```bash
pip install -e ".[kraken]"   # kraken>=5.0

ocr recognize dokument.png --backend kraken
python -m training.evaluate --data ./data/pl_lines_val --backend kraken
```

Lub w Pythonie:

```python
from ocr import OcrEngine, OcrConfig

cfg = OcrConfig(
    recognizer_backend="kraken",
    kraken_model="PiotrSty/ehri-dataset::models/polish_nfd_9313.mlmodel",
    device="cuda",
)
with OcrEngine(cfg) as engine:
    result = engine.recognize("maszynopis.tif")
    print(result.text)
```

### Benchmark: Kraken vs OpenCV+TrOCR na polskim maszynopisie EHRI

Pełny benchmark: [`training/kaggle_kraken_benchmark.ipynb`](training/kaggle_kraken_benchmark.ipynb)
(Kaggle GPU T4). Zbiór: 15 polskich stron EHRI (468 linii GT z ALTO XML).

| Backend | Segmentacja | CER | WER | Linie wykryte |
|---|---|---:|---:|---:|
| OpenCV+TrOCR run5 | OpenCV | 82,42% | 94,35% | 179/468 (38%) |
| **Kraken e2e** | **Kraken** | **14,25%** | **48,49%** | **467/468 (100%)** |
| Kraken + GT (ALTO) | GT baselines | 10,67% | 32,93% | 468/468 (100%) |
| Kraken e2e + deskew | Kraken | 14,71% | 48,89% | 467/468 (100%) |
| Kraken + GT + binarization | GT baselines | 20,26% | 65,16% | 468/468 (100%) |

**Wnioski:**

- **Kraken jest 5,8× lepszy** od OpenCV+TrOCR na maszynopisie (CER 14% vs 82%).
- **OpenCV gubi 62% linii** na maszynopisie — segmentacja to główna blokada.
- **Kraken wykrywa 100% linii** — segmentacja baseline działa na maszynopisie.
- **Binarization szkodzi** — model trenowany na grayscale (CER 14% → 20%).
- **Deskew nie pomaga** — strony EHRI są już wyrównane.
- **GT segmentacja poprawia** CER z 14% do 11% — wytrenowanie modelu segmentacji
  na polskich danych mogłoby poprawić e2e.

**Różnica 14% vs 7% EHRI:** możliwe przyczyny to legacy polygon extractor
(model nie trenowany z nową metodą), różnica wersji Kraken, oraz brak
dedykowanego modelu segmentacji dla polskiego maszynopisu.

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
  słabszy do tekstu w naturze i złożonych tła. **Na maszynopisach zawodzi**
  (wykrywa 38% linii) — użyj backendu Kraken (`--backend kraken`).
- Alternatywa bez treningu: PaddleOCR-VL + LoRA RysOCR (lepsza polska diakrytyka od ręki).

Pilot rzeczywistych skanów: [wyniki i odtworzenie](docs/PUBLIC_SCAN_PILOT.md).

Zwykły druk i vision API: [wyniki oraz przygotowany test](docs/PRINT_AND_VISION_PILOT.md).
