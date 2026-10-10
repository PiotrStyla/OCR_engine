# Defekty referencji testu A (2026-10-10)

Audyt provenance wad referencji zamrożonego testu A: porównanie tekstu GT ze
źródłowymi PAGE XML w `impactcentre/groundtruth-pol`. Wniosek: **wszystkie
znane wady są źródłowe (upstream), nie nasze** — ekstrakcja jest wierna
(identyczne liczby znaków na próbkach).

- Errata `NA2_FT__434735`: XML źródłowy ma te same 4 znaki — adnotacja ucięta u
  źródła; naprawa wymaga nowej transkrypcji ludzkiej (bramka dwu-recenzyjna).
- U+FFFD (86×, 23 strony) + PUA (624×, 33 strony, 10 unikalnych znaków):
  historyczne glify w prywatnej mapie IMPACT; ≥1,36% CER wbudowanego szumu.

Raport i bramki decyzji: `docs/POLLOCR_REFERENCE_DEFECTS_20261010.md`.
`review-queue.jsonl`: 34 pozycje do przeglądu (priorytet 1: errata).
Żadna zamrożona referencja nie została zmieniona — to dowody i kolejka
recenzyjna.
