# Pakiet treningowy recognizer v3 (2026-10-10)

Komplet danych do treningu: **226 par linii** z druku realnego, scalonych z dwóch
nie-nakładających się źródeł, plus **65 par probe** jako hold-out (nigdy do
treningu).

| Źródło | Pary | Reguła cięcia |
|---|---:|---|
| mining expansion (przyjęte przez minera) | 127 | padded rectangle (zamrożona) |
| pula V2 — geometry-v2-accepted | 43 | pas atramentu + 1 px |
| pula V2 — ink-gap-approved | 56 | pas atramentu + 1 px |
| **Razem kandydaci** | **226** | |
| probe hold-out | 65 | — |

Rodziny pracy (kandydaci): may 116, witkiewicz 33, prokesch 25, schneider 23,
zawadzki 17, pamietniki 12. Typy treści: 35 publisher-backmatter (may-0557/0563),
191 nierozpoznanych.

- `train-candidates.jsonl`, `probe-heldout.jsonl` — wiersze z pełną prowenencją
  (tekst, pewność, rozpiętość źródła, hashe plików, reguła cięcia, status)
- `report.json` — sumy, hashe wejść, bramki otwarte
- Bajty wycinków/tekstów są cytowane ścieżką + SHA-256 z katalogów dowodowych
  (expansion audit root + replay-pool), nie duplikowane

**Nic nie jest dopuszczone do treningu** (`eligible_for_training: false`).

## Receptura recognizer v3 (do uruchomienia na GPU)

- dane: `train-candidates.jsonl` (226) + replay/protection sets z pakietu
  recognizer v2 (9 linii historycznych + 75 zwykłych dev — bramka ochrony);
  probe (65) wyłącznie do ewaluacji
- baza: `PiotrSty/trocr-pl-mixed-v3` (bieżący baseline)
- bramki jak w V2 (chroniona domena bez regresji: zwykłe 75 linii ≤ 5,33% CER;
  historyczne 9 linii poprawa; łączny ≤ 8,37%), dodatkowo ocena na zamrożonym
  teście A zamiast wewnętrznych dev
- cel liczbowy: historyczne <20% na liniach → test A <17% CER (poniżej frontiera
  zero-shot) → <13,5% (frontier, strony czyste)
- harness: `training/run_reviewed_recognizer_colab_v2.py` / `reviewed_recognizer_selection.py`
  (Kaggle/Colab/Runpod GPU)

## Otwarte bramki przed promocją

drugi recenzent kandydata referencji; 5 rzadkich PUA (23 miejsca); U+FFFD (86);
errata; klasyfikacja treści (191 wierszy); 10 flagowanych wycinków puli.
