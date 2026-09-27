# Historical recognizer v2 - wynik treningu

Data analizy: 2026-09-27

## Decyzja

V2 spelnia zamrozone progi eksperymentu po poprawnym potraktowaniu granicy
`+2 pp` jako dopuszczalnej. Notebook zapisany w evidence odrzucil te granice
wylacznie przez reprezentacje zmiennoprzecinkowa: delta EHRI wyniosla
`0.020000000000000018` zamiast matematycznego `0.02`.

Model jest kandydatem do zachowania i dalszej niezaleznej oceny, ale nie jest
dowodem SOTA. Oryginalny ZIP pozostaje niezmieniony. Nie zawiera wag, poniewaz
bledna bramka zablokowala ich spakowanie.

## Integralnosc i wykonanie

- ZIP dowodowy SHA-256:
  `04ba8229dc906195a33a570e301da73c37cc5ba5b2065be83c79b861b6c14d99`;
- wszystkie 13 sum z `checksums.json` sa zgodne;
- kod treningu: `8b950b8f5c42af259269fb0bb5a706f1c6724148`;
- konfiguracja w evidence jest identyczna z zamrozona konfiguracja repo;
- GPU: Tesla T4; Python 3.13.15; Torch 2.11.0+cu128;
- 1365 ekspozycji treningowych, 139 linii walidacyjnych;
- 4 epoki, LR `2e-5`, LoRA fp16, 3 047 424 parametry trenowalne;
- najlepszy checkpoint: `checkpoint-513`, CER walidacyjny 31,4185%, epoka 4;
- wszystkie 9 kontroli dokladnego overlapu obrazu ma wynik zero.

Tokenizer poprawnie obsluzyl 1402 etykiety. Najdluzsza miala 88 tokenow,
zadna nie przekroczyla limitu 128 i nie bylo bledow round-trip. Historyczna
pisownia nie byla modernizowana.

## Metryki

| Zbior | Baseline CER | v2 CER | Delta CER | Baseline WER | v2 WER | Delta WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historyczna walidacja, 139 linii | 34,8450% | 31,4185% | **-3,4266 pp** | 84,7561% | 80,7927% | **-3,9634 pp** |
| `real-lines-v1`, 75 linii | 5,3586% | 5,9726% | **+0,6140 pp** | 22,8800% | 24,4800% | **+1,6000 pp** |
| EHRI test, 81 linii | 29,7442% | 31,7442% | **+2,0000 pp** | 74,9611% | 80,4044% | **+5,4432 pp** |

V2 zachowuje mniej zysku historycznego niz v1, ale ogranicza jego regresje:
wobec v1 CER poprawia sie o 3,3771 pp na `real-lines-v1` i 9,2093 pp na EHRI.

## Bramki promocji

| Bramka | Wynik |
| --- | --- |
| Historyczna walidacja lepsza od mixed-v3 | PASS |
| Regresja `real-lines-v1` nie wieksza niz 1 pp | PASS: +0,6140 pp |
| Regresja EHRI nie wieksza niz 2 pp | PASS: +2,0000 pp |
| Tokenizer round-trip bez bledow | PASS |
| Brak obcinania etykiet | PASS |
| Wszystkie zamrozone bramki | **PASS po korekcie porownania granicznego** |

## Nastepny krok

Najpierw trzeba odzyskac scalone wagi z aktywnej sesji Colab, bez ponownego
treningu. Potem uruchomic pelny, niezalezny benchmark stron i porownanie z
modelem bazowym. Publikacja pozostaje wylaczona do czasu tej oceny.
