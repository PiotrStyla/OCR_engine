# PolOCRBench test A: pomiar SOTA v1 — protokół

## Cel

Zmierzyć systemy klasy SOTA na zamrożonym teście A (36 stron, IMPACT
`history_print`) i podać policzalną lukę do SOTA. To pomiar, nie trening: bez
strojenia wag, adapterów, pamięci między wywołaniami i bez korekty wyników.
Nic nie jest promowane do treningu ani ewaluacji wewnętrznej.

## Systemy

| Id | System | Ścieżka | Prompt |
| --- | --- | --- | --- |
| `paddlevl` | PaddleOCR-VL-1.6 (pipeline; fallback `paddleocr.PaddleOCRVL` v1) | pipeline, własny Markdown | brak (model nie przyjmuje promptu) |
| `qwen3vl` | Qwen3-VL-4B-Instruct (`transformers` chat) | zero-shot, dekodowanie greedy | zamrożony szablon A z `zero_shot_prompt_v1.md` |

Odłożone (osobne protokoły/koszty): MinerU2, GLM-OCR, modele API frontier.
Pierwszy pomiar jest ograniczony do dwóch systemów GPU.

Zasady tracku zero-shot są zachowane: szablon A stosowany dosłownie, bez
dopisków i few-shot; wyciąganie odpowiedzi mechaniczne; `paddlevl` jako pipeline
nie ma promptu i jest oznaczony w `run.json` (`prompt_used: false`).

## Zamrożone wejście

- Pakiet: `PiotrSty/impact-print-v2`, rewizja `a2480fde6f15284701458ff370b81cce50dc5c2d`,
  `impact-print-v2-test.tar.gz`, SHA-256 `0a9ffa126029703726fc5883a8279ddccf15763c4fdd483ed9b8cc034bdb42f0`.
- Staging (`training/stage_impact_benchmark`) sprawdza sumę każdego obrazu
  względem zamrożonego `history_testA_manifest.jsonl` (36 stron, 3 kolekcje).
- Manifest stagingu ma ścieżki przenośne i te same teksty GT; ewaluator ponownie
  weryfikuje sumy obrazów.

## Metryka

`training/transcription_eval` v1.1 (jak dla zamrożonych baseline'ów): CER/WER
micro oraz podobieństwo struktury Markdown (średnia po stronach; strony z
błędem/brakiem dostają zero). Normalizacja: NFC, cudzysłowy typograficzne →
proste, białe znaki; wielkość liter i diakrytyki zachowane.

## Dowody (ZIP `polocrbench-sota-measurement-v1-evidence.zip`)

- `<model>-predictions.jsonl` — predykcje per strona: `id`, `text`, `status`,
  `error`, `elapsed_seconds`.
- `<model>-score.json` — raport ewaluatora z wynikami per strona i sumami
  kontrolnymi manifestu oraz predykcji.
- `<model>-run.json` — spec modelu, faktyczne wersje pakietów, hash i wersja
  promptu, dekodowanie, czasy, `measurement_only: true`, `training_performed: false`.
- `staged-verification.json`, `receipt.json` (rewizja kodu, piny wejścia),
  `summary.json`.
- Kopia ZIP-a na Dysku Google z ostatniej komórki notebooka.

## Notebook

`notebooks/colab_sota_testA.ipynb` (GPU T4, „Uruchom
wszystko", bez ręcznych wgrań): przypięta rewizja kodu i wejścia, testy CPU
przed pomiarem, smoke test 1 strony na model, każdy model w osobnym procesie,
spakowanie dowodów. Stos paddle (`paddlex`, `paddlepaddle-gpu`) nie jest
wstępnie przypięty — zmienne macierze kół GPU na Colab; faktyczne wersje
trafiają do `run.json` i to one identyfikują przebieg.

Budowa z przypiętą rewizją kodu:

```powershell
python -m training.build_polocrbench_sota_colab --code-revision <rev>
```

## Weryfikacja lokalna

```powershell
python -m pytest -q tests/test_sota_benchmark.py tests/test_polocrbench_sota_colab.py
```

Po zwróceniu ZIP-a audyt lokalny przelicza metryki z predykcji i sprawdza
piny:

```powershell
python -m training.audit_polocrbench_sota_evidence `
  --archive polocrbench-sota-measurement-v1-evidence.zip `
  --benchmark data/impact-test-a-staged --output runs/sota-audit
```

Audytor (`training/audit_polocrbench_sota_evidence.py`) wymaga dokładnego
zestawu członków ZIP-a, paragonu wejścia zgodnego z pinami, flag
`measurement_only`/`training_performed`, hasha zamrożonego promptu tam, gdzie
prompt obowiązuje, i **przelicza** CER/WER/strukturę zwróconych predykcji
przez `training.transcription_eval`; raportowane wartości są tylko porównywane.
Testy: `tests/test_audit_polocrbench_sota_evidence.py` (7 przypadków, m.in.
drążona predykcja, zły hash promptu, flaga treningu, nadmiarowy członek ZIP-a).

## Status pomiarów (2026-10-08)

- **CPU, Surya 2** (`surya-ocr` 0.22.1, backend `llama-server` b11503,
  `LLAMA_CPP_BINARY`): pełny pomiar 36 stron odpalony lokalnie, ~353 s
  inferencji na stronę; wynik w osobnym dokumencie po zakończeniu.
- **API zero-shot zablokowany**: `OPENROUTER_API_KEY` — 402 (brak kredytów,
  ~3,9k tokenów salda), `OPENAI_API_KEY` — 429 `credit_balance_exhausted`.
  Pomiar modeli API wymaga doładowania konta; to jedyny ręczny krok poza
  Colabem.
- **GPU (Colab)**: notebook gotowy do odpalenia (PaddleOCR-VL-1.6 + Qwen3-VL).

## Ograniczenia

- Referencje testu A zawierają 86 znaków U+FFFD i 624 znaki PUA (patrz
  `docs/POLLOCR_SPLIT_POLICY.md`) — bezwzględny CER ma szum referencyjny.
- `max_new_tokens=4096` dla VLM może uciąć bardzo długie strony; ucięcia widać
  w predykcjach i długości odpowiedzi.
- Paddlex/paddlepaddle nie są przypięte wersjami (patrz wyżej).
- To pomiar jednego zamrożonego podzbioru druku historycznego; nie jest
  rankingiem modeli na dokumentach współczesnych i nie jest twierdzeniem SOTA.
