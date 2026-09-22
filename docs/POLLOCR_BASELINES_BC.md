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
| Dwustopniowy: Tesseract.js + Bielik (`fabryka.ai`) | A, B, C | lokalnie CPU + API | `training/run_two_stage_baseline.py` |
| LoRA Qwen2.5-VL na danych organizatora (track `constrained`) | A, B, C | Kaggle T4 | `training/kaggle_qwen_vl_finetune.py` |

Surya nie ekstrahuje pól kluczowych (KIE) — podzadanie C obsługują Qwen-VL i
modele API. Wszystkie runnery domyślnie robią dry-run/preflight (tylko
biblioteka standardowa); sieć uruchamia `--execute`.

## System dwustopniowy (track `open`, polski akcent)

Stopień 1 dowolny — tekst stron z istniejących predykcji podzadania A
(Tesseract.js `tools/cpu-baseline`, Surya, Qwen-VL, API). Stopień 2: Fabryka AI
(`bielik-11b-v3`, prompt `polocrbench-two-stage-prompt-v1` — wariant tekstowy;
zamrożony `zero_shot_prompt_v1` dotyczy wejścia obrazowego w tracku zero-shot).
Strona bez tekstu z OCR-a (`UpstreamOCR`) kończy się wierszem `error`.

```bash
node tools/cpu-baseline/run.mjs manifest-A.jsonl runs/tesseract-A
python -m training.run_two_stage_baseline --subtask B --manifest manifest-B.jsonl \
    --text-run runs/tesseract-A/predictions.jsonl --output runs/bielik-B --execute
python -m training.run_two_stage_baseline --subtask C --manifest manifest-C.jsonl \
    --text-run runs/tesseract-A/predictions.jsonl --output runs/bielik-C --execute
```

Stan API Fabryka (2026-09-22): endpoint `https://fabryka.ai/v1` jest
OpenAI-compatible, zwraca `usage` (tokeny), ale **nie przyjmuje obrazów** —
`messages.content` musi być stringiem (422 na częściach obrazowych dla
wszystkich 7 modeli: `bielik-11b-v3`, `qwen3.8-27b`, `qwen-bielik-hybrid`,
`muse-glimmer`, `gollem-v4-250m-pl`, `slayerlab-sub150-32m-completion`, `auto`).
Do pomiarów zawsze jawna nazwa modelu (`auto` routuje). Klucz: `FABRYKA_API_KEY`
(dobierany do hosta `base_url` jak pozostałe endpointy).

## Baseline tracku `constrained` (fine-tune na danych organizatora)

`training/kaggle_qwen_vl_finetune.py` (jedna komórka na Kaggle T4, Internet on)
zamyka wymóg „baseline'y w każdym tracku": LoRA Qwen2.5-VL trenowane **wyłącznie
na syntetycznym treningu z generatora organizatora** (zamrożony prompt zero-shot
v1 na wejściu, więc system zostaje kompatybilny z promptem toru zero-shot/API).
Trzy podzadania idą w jednym adapterze (A/B/C). Ewaluacja na wydzielonym plasterku
generatora z innym seedem i degradacjami zawierającymi `photo` — plaster zastępuje
test A do czasu anotacji realiów i mierzy generalizację poza degradacje z treningu
(`photo` celowo wyjęte z treningu jako oś holdout Test B). Skrypt zapisuje
adapter LoRA, predykcje A/B/C, wyniki przez publiczne ewaluatory, `run.json`
z budżetem kroków i czasami oraz ZIP do pobrania.

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

## Drugi pomiar (2026-09-22): system dwustopniowy Tesseract.js + Bielik

Te same 4 czyste strony syntetyczne, track `open`:

| Podzadanie | Wynik | Koszt |
| --- | --- | --- |
| A — transkrypcja (Tesseract.js `pol+eng`) | CER **8,97%**, WER 8,09%, struktura 0,668 | 1,7–3,1 s/str. (CPU lokalnie) |
| B — tabele (Bielik z transkrypcji) | TEDS **0,849** (= struct 0,849) | 24,5 s/str., 4 141 tok. |
| C — KIE (Bielik z transkrypcji) | F1 **0,971** | 8,5 s/str., 3 359 tok. |
| **Wynik zbiorczy** | **0,910** | 0 błędów |

Porównanie z baseline'em API (te same strony): composite **0,910 vs 0,781** —
dwustopniowy system wygrywa zwłaszcza na tabelach (TEDS 0,849 vs 0,595):
ekstrakcja HTML z czystej transkrypcji tekstem jest łatwiejsza niż generowanie
go z obrazu. Tokeny wejściowe: ~1 tys./str. vs ~149 tys./str. (obraz).
Wyniki TEDS == TEDS-struct wskazują, że Bielik trafił treść komórek, a błędy
zostały w strukturze (kolspany). To wciąż small-n na czystym syntetyku —
kolejne pomiary na degradacjach i realiach zmienią obraz.

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
