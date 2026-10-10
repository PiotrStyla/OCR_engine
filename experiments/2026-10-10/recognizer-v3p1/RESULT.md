# Recognizer v3.1 — wynik treningu guarded (2026-10-10)

## Decyzja bramek

**`SELECTED unchanged-baseline`** — ponownie żaden wariant nie przeszedł
bramek; pozostaje niezmieniony `PiotrSty/trocr-pl-mixed-v3`. Bez promocji,
bez twierdzenia SOTA.

## Wyniki (CER; rozwój chroniony, 84 linie)

| wariant | historyczne (9) | zwykłe (75) | łączny | znaki zast. | decyzja |
| --- | --- | --- | --- | --- | --- |
| baseline | 33,03% | **5,33%** | **8,37%** | 2 | — |
| control-lr1e5-replay200 | **31,22%** | 6,06% | 8,82% | 3 | odrzucony |
| low-lr3e6-replay200 | 33,03% | 5,61% | 8,62% | 2 | odrzucony |
| low-lr3e6-replay0 | 32,81% | 5,55% | 8,55% | 2 | odrzucony |

## Diagnoza — wariant replay 0 rozstrzyga

**Syntyka NIE jest przyczyną regresji zwykłego druku.** Wariant
`low-lr3e6-replay0` (historyczne ×8 + 226 par, zero replayu syntetycznego)
też regresuje zwykłe (5,33% → 5,55%). Źródłem zamiany domen jest **sam
pakiet printed-replay** (druk zabytkowy): trening na nim poprawia linie
historyczne kosztem czystego druku współczesnego. To cecha danych, nie
miksu.

Dalsze obserwacje:
- `control-lr1e5-replay200` daje **najlepszy wynik historyczny dotąd**
  (31,22%; v3: 31,67%; baseline: 33,03%), ale największą regresję zwykłych
  (6,06%) i +1 znak zastępczy — zamiana domen jest realna i skalowalna.
- LR 3e-6 przy ×8 historycznych niemal nie rusza historii (33,03% = baseline;
  32,81% bez syntyki) — za ostrożny przy tym miksie.
- Metryka łączna (84 linie) jest w 89% zdominowana przez zwykłe — każda
  poprawa historyczna musi z nawiązką zrównać regresję 75 linii zwykłych.
  To strukturalne uprzedzenie bramki „combined:CER-improves".

## Implikacja dla strategii

Dalsze strojenie miksu na tych samych danych nie rozstrzygnie zamiany.
Realne opcje:
1. **Routing domenowy** (silnik ma już routing PL/EN): strony z drukiem
   zabytkowym kierować do wyspecjalizowanego adaptera (`control-lr1e5…`),
   pozostałe do baseline — bramki wtedy oceniają domenę, w której model
   pracuje.
2. **Dane przeciwwagowe**: więcej druku współczesnego w treningu, żeby
   poprawa historyczna nie odbywała się kosztem zwykłych.
3. **Decyzja polityczna**: świadoma zamiana domen (wymaga zmiany bramek —
   dziś celowo jej blokują).
4. **Zamrozić baseline**, iść do testu A.

## Przebieg i koszt

- Pod `1lfbvmkqc5xwfr` (RTX 4090 SECURE, EU-RO-1, $0,89/h), 17:58:58–18:27
  UTC ≈ **28,5 min ≈ $0,42**. Stróż `V3P1_DONE` zadziałał — brak ponowień
  po sukcesie; `hf_transfer` skrócił setup do ~1,5 min (v3: ~10 min).
- Dowód: `recognizer-v3p1-evidence.zip`, SHA-256
  `13883285c77eaa34e637ba1cfc92448abc28c1d9818497179e36cd69f99ecce2`.
