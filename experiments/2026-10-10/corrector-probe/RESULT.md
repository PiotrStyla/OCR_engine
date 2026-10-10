# Korekta LLM na predykcjach Qwen3-VL-4B (test A) — wynik negatywny

Data: 2026-10-10. Pomiar na zamrożonym teście A (36 stron) na predykcjach
`experiments/2026-10-10/runpod-sota/qwen3vl-predictions.jsonl` (20,03% CER).

## Konfiguracja (wybrana na dev, test A nietknięty do pomiaru)

- narzędzie: `training/postcorrect.py` (korekta z bramką szumu, HIPE-OCRepair)
- model korektora: `qwen/qwen3-14b` (otwarte wagi) — wybrany na
  `benchmarks/correction-v1` (8 cases): CER 2,77% vs 3,08% dla
  `google/gemma-3-12b-it` i `openai/gpt-4o-mini`; repair 8,64% (remis)
- bramka szumu: 0.15 (domyślna)

## Wynik

| konfiguracja | CER micro | WER micro | struktura |
| --- | ---: | ---: | ---: |
| Qwen3-VL-4B (przed) | 20,03% | 61,36% | 0,588 |
| + korekta qwen3-14b | **20,19%** | **61,55%** | **0,583** |

36 stron: poprawione 0, pogorszone 4, nietknięte 32 (bramka).

## Diagnoza

1. Bramka szumu (`estimate_noise`: tokeny samogłoskowe, śmieciowe znaki, wysokie
   kodpunkty) **nie wykrywa błędów silnego VLM** — jego błędy to poprawne
   leksykalnie słowa z błędnymi diakrytykami/literami; bramka otworzyła się
   tylko na 4 stronach.
2. Na tych 4 korektor **pogorszył wszystkie** (prze-korygowanie mimo zakazu
   parafrazy) — klasyczny failure mode LLM post-correction na tekście
   historycznym.
3. Wniosek: przy CER ~20% z silnego VLM korekta LLM jednym przebiegiem nie
   jest dźwignią. Dalsze próby (inna bramka/model) wymagałyby strojenia na
   teście A — protokół tego nie pozwala.

Konfiguracja zamrożona przed pomiarem; wynik raportowany w całości (bez
wybierania). Dowody: `experiments/2026-10-10/corrector-probe/`.
