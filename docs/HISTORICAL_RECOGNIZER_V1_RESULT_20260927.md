# Historical recognizer v1 - wynik treningu

Data analizy: 2026-09-27

## Decyzja

Kandydat poprawil rozpoznawanie historycznego druku, ale nie przeszedl bramek
ochrony pozostalych domen. Nie publikujemy ani nie promujemy modelu v1.

Wynik wskazuje na katastroficzne zapominanie po dostrojeniu wylacznie na 258
liniach historycznych. Nastepny eksperyment powinien uzyc zamrozonego replayu
danych treningowych z domen, ktore obslugiwal model bazowy. Zbiory
`real-lines-v1` i EHRI test pozostaja tylko bramkami oceny.

## Integralnosc i wykonanie

- ZIP dowodowy SHA-256:
  `7f180fb0f4793a849a4d809d52d3fe03569c052945a85ca639ce676d4d7d27c6`;
- 13 plikow, bez duplikatow i niebezpiecznych sciezek;
- wszystkie sumy z `checksums.json` zgodne;
- kod: `a18fd3aee050ba10ce3595875ea9a9c5f280c6fd`;
- GPU: Tesla T4; Python 3.13.15; Torch 2.11.0+cu128;
- 12 epok, LR `5e-5`, zwykle LoRA fp16, 3 047 424 parametry trenowalne;
- najlepszy checkpoint: `checkpoint-396`, CER walidacyjny 27,1857%, epoka 12.

## Audyt tekstu

Tokenizer poprawnie obsluzyl wszystkie 397 etykiet. Najdluzsza etykieta miala
63 tokeny, zadna nie przekroczyla limitu 128 i nie bylo bledow round-trip.
Pisownia historyczna nie byla modernizowana.

## Metryki

| Zbior | Baseline CER | v1 CER | Delta CER | Baseline WER | v1 WER | Delta WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historyczna walidacja, 139 linii | 34,8450% | 27,1857% | **-7,6594 pp** | 84,7561% | 73,7805% | **-10,9756 pp** |
| `real-lines-v1`, 75 linii | 5,3586% | 9,3497% | **+3,9911 pp** | 22,8800% | 34,7200% | **+11,8400 pp** |
| EHRI test, 81 linii | 29,7442% | 40,9535% | **+11,2093 pp** | 74,9611% | 85,8476% | **+10,8865 pp** |

Wzglednie historyczny CER poprawil sie o 22,0%, ale CER wspolczesnego druku
wzrosl o 74,5%, a EHRI o 37,7%.

## Bramki promocji

| Bramka | Wynik |
| --- | --- |
| Historyczna walidacja lepsza od mixed-v3 | PASS |
| Regresja `real-lines-v1` nie wieksza niz 1 pp | **FAIL: +3,99 pp** |
| Regresja EHRI nie wieksza niz 2 pp | **FAIL: +11,21 pp** |
| Tokenizer round-trip bez bledow | PASS |
| Brak obcinania etykiet | PASS |
| Wszystkie bramki | **FAIL** |

Notebook prawidlowo nie utworzyl archiwum modelu i niczego nie opublikowal.

## Nastepny eksperyment

V2 powinien byc pojedynczym, z gory zamrozonym testem replay:

1. Start z tego samego `trocr-pl-mixed-v3`.
2. Historyczny train polaczony wylacznie z rozlacznymi splitami treningowymi
   EHRI i syntetycznego druku, bez danych z bramek oceny.
3. Nizszy learning rate i mniej epok niz v1.
4. Te same trzy zestawy oceny i niezmienione progi promocji.
5. Brak publikacji lub pakowania wag, jezeli choc jedna bramka nie przejdzie.

Surowe dowody sa w
`experiments/2026-09-27/historical-recognizer-v1/`.
