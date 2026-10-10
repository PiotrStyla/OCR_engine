# Recognizer v3 — wynik treningu guarded (2026-10-10)

## Decyzja bramek

**`SELECTED unchanged-baseline`** — żaden z 3 wariantów nie przeszedł bramek;
zgodnie z polityką pozostało niezmieniony baseline `PiotrSty/trocr-pl-mixed-v3`
(`85d0c91c26f8e088849096dded7c9ba10b4cd9c9`). Bez promocji, bez twierdzenia SOTA.

## Wyniki (CER; rozwój chroniony, 84 linie)

| wariant | historyczne (9) | zwykłe (75) | łączny | znaki zast. | decyzja |
| --- | --- | --- | --- | --- | --- |
| baseline | 33,03% | 5,33% | 8,37% | 2 | — |
| control-lr1e5-replay500 | 31,67% | 5,86% | 8,70% | 3 | odrzucony |
| low-lr3e6-replay500 | 31,90% | 5,53% | 8,42% | 2 | odrzucony |
| low-lr3e6-replay2000 | 32,58% | 5,69% | 8,65% | 2 | odrzucony |

Powody odrzuceń (`selection.json`): wszystkie warianty — regresja CER/WER na
zwykłych i brak poprawy łącznej; `control` dodatkowo +1 znak zastępczy.

## Diagnoza

- Pakiet 226 par **poprawia linie historyczne u wszystkich wariantów**
  (33,0% → 31,7–32,6%), ale efekt jest mały (≤1,4 pp) i okupiony regresją
  zwykłego druku (5,33% → 5,53–5,86%) — klasyczna zamiana domen.
- Cel „historyczne <20%" **nieosiągnięty** (najlepsze 31,7%). Sama pula
  replay nie wystarcza; potrzebna jest zmiana receptury lub więcej danych
  historycznych (patrz niżej).
- Bramki zadziałały dokładnie wg projektu: zamiana „historyczne lepsze ↔
  zwykłe gorsze" została odrzucona — wcześ­niejszy trening V1 takiej regresji
  nie wykrywał.

## Przebieg i poprawki infrastrukturalne

- Pod `01u6z0op2smaft` (RTX 4090 SECURE, EU-RO-1, $0,89/h), start 16:42 UTC,
  zakończenie 17:50 UTC, wyłączenie 17:54 UTC → **~72 min ≈ $1,07**.
- Naprawione w locie: (1) pętla retry clone musi czyścić niepusty katalog po
  przerwanym clone; (2) runner uruchamiany przez `-m` (pakiet `training`);
  (3) szablon ma `HF_HUB_ENABLE_HF_TRANSFER=1` → `hf_transfer` wymagany w
  instalacji. Poprawki uwzględnione w README przebiegu.
- Dowód: `recognizer-v3-evidence.zip` (SHA-256
  `6569974fb98868064f1ee651d177917e69b90a59a919c6f9a704f521d8d33c92`),
  226 par w manifeście, 65 probe trzymane poza treningiem, 310 auditów
  tokenizera.

## Dalsze kroki (do wyboru)

1. **v3.1 — inny miks**: więcej powtórzeń historycznych (×8), mniej syntetyki
   (replay 200 → waga domenowa), ten sam harness i bramki.
2. **Więcej danych historycznych**: rozszerzenie minera o kolejne rodziny
   prac przed kolejnym treningiem (efekt 1,4 pp przy 226 parach sugeruje
   krzywą nasyconą).
3. **Zaakceptować baseline** i iść do testu A (pomiar frontieru zero-shot).
