# PolOCRBench — baseline'y podzadań B/C (i A) w trackach

Organizatorskie baseline'y wg §5 procedury: co najmniej Surya, Qwen-VL i jeden
model API, każdy z raportem kosztów inferencji per strona. Wszystkie uruchomienia
są zero-shot (bez fine-tuning) z **zamrożonym promptem**
`benchmarks/polocrbench/prompts/zero_shot_prompt_v1.md` — track zero-shot/API.

| Baseline | Podzadania | Środowisko | Runner |
| --- | --- | --- | --- |
| Surya OCR 2 (open-weight, 0,65B) | A, B (C poza zakresem modelu) | lokalnie (llama.cpp) lub GPU (vllm) | `training/run_surya_benchmark.py` |
| Qwen2.5-VL-7B-Instruct 4-bit (open-weight) | A, B, C | Kaggle T4 | `training/kaggle_qwen_vl_bc.py` |
| API `openai/gpt-4o-mini` przez OpenRouter | A, B, C | dowolny host z kluczem | `training/run_vision_baseline.py` |

Surya nie ekstrahuje pól kluczowych (KIE) — podzadanie C obsługują Qwen-VL i
modele API. Wszystkie runnery domyślnie robią dry-run/preflight (tylko
biblioteka standardowa); sieć uruchamia `--execute`.

## Uruchomienie

```bash
# model API (klucz z GEMINI_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY,
# dobierany do hosta endpointu; --base-url dla Gemini: ...)
python -m training.run_vision_baseline --subtask C --manifest data/polocrbench-synth-v1/manifest-C.jsonl \
    --output runs/api-C --model openai/gpt-4o-mini --execute

# Surya 2: pip install surya-ocr (+ llama.cpp llama-server na CPU albo vllm na GPU)
python -m training.run_surya_benchmark --subtask B --manifest data/polocrbench-synth-v1/manifest-B.jsonl \
    --output runs/surya-B --execute

# Qwen-VL: training/kaggle_qwen_vl_bc.py jako jedna komórka (T4, Internet on)
```

Punktacja po każdym runnie (predykcje → manifesty B/C są w formacie
podzadań, walidator sprawdza format i kompletność):

```bash
python -m training.validate_submission --manifest M.jsonl --predictions runs/.../predictions.jsonl --subtask C
python -m training.table_eval --manifest manifest-B.jsonl --predictions runs/.../predictions.jsonl --output score-B.json
python -m training.kie_eval --manifest manifest-C.jsonl --predictions runs/.../predictions.jsonl --output score-C.json
python -m training.composite_score --report-a A.json --report-b B.json --report-c C.json --output composite.json
```

## Raport kosztów per strona

Każdy runner zapisuje `run.json` z sekcją `cost`: liczba stron, średni/maksymalny
czas na stronę i (dla API) sumy tokenów (`prompt_tokens`, `completion_tokens`,
`total_tokens`). Dla Suryi czas jest liczony per strona (jedno rozpoznanie strony
wypełnia wszystkie sloty tabel tej strony; `elapsed_seconds` trafia do pierwszego
wiersza slotu). Składnik `timing` w raporcie zbiorczym agreguje czasy z rekordów
predykcji.

## Pierwszy pomiar (2026-09-22): API `openai/gpt-4o-mini`

4 czyste strony syntetyczne (faktura, umowa, pismo urzędowe, formularz — seed 7),
prompt zero-shot v1, zero-shot/API track:

| Podzadanie | Wynik | Koszt |
| --- | --- | --- |
| A — transkrypcja | CER **10,49%**, WER 20,95%, struktura 0,799 → 0,895 | 7,7 s/str., 149 098 tok. |
| B — tabele | TEDS **0,595** (struct 0,601) | 9,3 s/str., 150 062 tok. |
| C — KIE | F1 **0,853** (P = R = 0,853) | 4,5 s/str., 148 486 tok. |
| **Wynik zbiorczy** | **0,781** | 12 wywołań, 0 błędów |

Pierwszy wniosek: nawet na czystych stronach syntetycznych najtrudniejsze są
tabele (dokładne HTML-e z colspan) — spójnie z tezą zadania, że obecne modele
wykładają się na strukturze. Pomiar jest punktem kontrolnym pipeline'u
(small-n, strony syntetyczne), nie rankingiem modeli.

## Znane ograniczenia

- Qwen-VL i Surya: kod i testy gotowe (przebiegi na falszywych predyktorach),
  pełne przebiegi wymagają Kaggle T4 (Qwen) oraz backendu inferencji Suryi
  (llama.cpp/vllm) — patrz statusy w README.
- Dla stron wielotabelowych runner API wypełnia slot `table_index` i-tą tabelą
  z odpowiedzi (pierwszą, gdy odpowiedzi jest mniej) — prompt pozostaje
  zamrożony i mówi o jednej tabeli; to znane utrudnienie wielostronicowych
  tabel w podzadaniu B.
- Baseline w tracku `constrained` (fine-tuning na danych organizatora) to
  osobne zadanie treningowe na syntetycznym trainie — dotąd nieuruchomione.
