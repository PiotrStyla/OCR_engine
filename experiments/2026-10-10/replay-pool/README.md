# Pula kandydatów replay v1 (2026-10-10)

Złożona z 8 stron konfliktowych zamrożonego rozszerzenia replay. Reguła
cięcia zatwierdzona przez użytkownika (2026-10-10): **pas atramentu linii + 1 px
marginesu**, nie prostokąt pudełek słów. Każdy wycinek pionowy (oś = mediana
kąta strony), z pełną prowenencją i maszynową weryfikacją atramentu.

| Miara | Wartość |
|---|---|
| Kotwice razem | **99** |
| — geometry-v2-accepted (43) | strony bez konfliktów po V2 |
| — ink-gap-approved (56) | strony z blokadą ramek, atrament rozdzielony |
| Weryfikacja: 0 pikseli sąsiada w kadrze | **89 / 99** |
| Flagi do przeglądu (atramet sąsiada) | 10 (witkiewicz 3, zawadzki 7) |
| Typy treści | publisher-backmatter 35 (may-0557/0563), nierozpoznane 64 |

- `manifest.jsonl` — 99 wierszy: tekst, pewność, rozpiętość źródła, reguła
  cięcia, bbox kadru, macierz afiniczna, narożniki w natywie, piksele atramentu
  sąsiada, status przeglądu
- `crops/`, `texts/` — materiały treningowe w stanie kandydata
- `report.json` — sumy, bramki otwarte, ograniczenia

**Nic nie jest dopuszczone do treningu** (`eligible_for_training: false`).
Otwarte bramki: akceptacja kandydata referencji przez drugiego recenzenta, 5
rzadkich kodpunktów PUA, U+FFFD, errata, klasyfikacja treści poza backmatter,
oraz 10 flag atramentowych (w kolejce recenzyjnej).
