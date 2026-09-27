# Historical full-page A/B v1 - wynik

Data audytu: 2026-09-27

## Decyzja

Prywatny recognizer v2 nie przechodzi zamrożonych bramek pełnostronicowych i
nie zastąpi `PiotrSty/trocr-pl-mixed-v3` jako domyślnego recognizera. V2
poprawia WER, lecz pogarsza podstawową metrykę CER i przekracza dopuszczalną
regresję CER w dwóch z trzech kolekcji. Model pozostaje prywatnym artefaktem
badawczym. Ten wynik nie jest dowodem SOTA.

## Integralność

- ZIP evidence SHA-256:
  `0c7dd17ffd1c38669ab62d335803ff5a9523184dab6f9482e28e841bbf12e263`;
- wszystkie cztery sumy plików wymienione w `checksums.json` są zgodne;
- konfiguracja eksperymentu jest zgodna z zamrożonym plikiem repozytorium;
- wykonano 36/36 stron, bez błędów i brakujących odpowiedzi;
- oba modele dostały te same 1215 wycinków z jednego przebiegu detektora;
- evidence nie zawiera skanów, referencji, surowych predykcji, wag ani tokenu.

Środowisko: Tesla T4, Python 3.13.15, Torch 2.11.0+cu128,
Transformers 4.57.6, Tokenizers 0.22.2, JiWER 4.0.0 i OpenCV 4.12.0.88.

## Wyniki

Niższa wartość jest lepsza.

| Model | CER micro | WER micro | Błędy/braki |
| --- | ---: | ---: | ---: |
| mixed-v3 baseline | 56,1126% | 97,8318% | 0 |
| historical-replay-v2 | 56,7479% | 97,1838% | 0 |
| Delta v2 - baseline | **+0,6353 pp** | **-0,6480 pp** | 0 |

| Kolekcja | Baseline CER | V2 CER | Delta CER |
| --- | ---: | ---: | ---: |
| NA2_FT | 51,4775% | 52,5300% | **+1,0525 pp** |
| Nowiny_z_Rakuz_FT | 60,8272% | 60,7757% | -0,0515 pp |
| Powodzenia_FT | 58,9118% | 60,2721% | **+1,3603 pp** |

V2 wygrywa CER na 20 stronach, baseline na 15, a jedna strona daje remis.
Dwustronny dokładny test znaków dla 35 nieremisowych par daje `p=0,4996`,
więc sama liczba zwycięstw nie jest dowodem przewagi v2.

## Bramki

| Bramka | Wynik |
| --- | --- |
| CER całego zbioru lepszy od baseline | FAIL |
| WER całego zbioru lepszy od baseline | PASS |
| Regresja CER każdej kolekcji nie większa niż 1 pp | FAIL |
| Brak błędów i brakujących odpowiedzi v2 | PASS |
| Wszystkie zamrożone bramki | **FAIL** |

## Diagnostyka referencji i układu

`NA2_FT__434735` ma tylko cztery znormalizowane znaki referencji (`Tam‑`),
ale detektor zwrócił 39 linii. Strona pozostaje w oficjalnym wyniku, ponieważ
protokół został zamrożony przed testem. Po jej diagnostycznym wyłączeniu v2
nadal pogarsza CER o około 0,3674 pp, zatem anomalia nie zmienia decyzji.

Detektor zwrócił 1215 linii wobec 1072 niepustych linii referencji: o 143
więcej, czyli 13,3%. Obecna kolejność jest row-major i nie rozumie kolumn.
Bezwzględne CER obu modeli powyżej 56% pokazuje, że ograniczeniem pozostaje
cały pipeline: segmentacja, kolejność czytania, niepełne referencje i
rozpoznawanie. Porównanie A/B izoluje tylko różnice recognizerów na wspólnych
wycinkach.

## Następny eksperyment

Nie trenować ani nie wybierać modelu na tych 36 stronach. Następny etap to
osobny pełnostronicowy development set z kompletnymi referencjami i etykietami
układu, zbudowany z kolekcji spoza zamrożonego testu. Na nim należy porównać
segmentację wraz z kolejnością kolumnową, a dopiero potem zamrozić recognizer
v3. Historyczna pisownia, w tym `á` i `ſ`, pozostaje bez modernizacji.

Dokładne pliki: [oryginalny ZIP](../experiments/2026-09-27/historical-full-page-ab-v1/historical-full-page-ab-v1-evidence.zip),
[rozpakowane evidence](../experiments/2026-09-27/historical-full-page-ab-v1/evidence/)
i [audyt decyzji](../experiments/2026-09-27/historical-full-page-ab-v1/result-audit.json).
