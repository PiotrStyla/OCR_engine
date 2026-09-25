# SLAYER Vision ONNX - wynik testu OCR

Data analizy: 2026-09-25

## Decyzja

Checkpoint `PiotrSty/slayer-vision-onnx` w rewizji
`22b86dc311bb6c7213ddfc1924d5d90119f29c10` nie jest kandydatem do OCR
pelnych stron. Nie uruchamiamy kosztownego testu wszystkich 36 stron ani nie
integrujemy go z domyslnym pipeline'em OCR_engine.

Test wykonawczy byl poprawny technicznie, ale model zachowal sie jak model
podpisow obrazow. Dla dwoch stron historycznego druku wygenerowal odpowiednio:

- `To zaproszenie.`
- `To zaproszenie do pracy.`

Nie jest to blad dekodera: obie odpowiedzi zakonczyly sie EOS, nie zawieraly
U+FFFD, a pelna sekwencja identyfikatorow zostala zdekodowana przypietym
tokenizerem GoLLeM.

## Wyniki

| Metryka | Wynik |
| --- | ---: |
| Strony | 2 |
| Bledy wykonania | 0 |
| CER micro | 98,2766% |
| WER micro | 99,5745% |
| Predykcje z U+FFFD | 0 |
| Czas laczny | 1,437 s |

| Strona | Znaki ref. | Znaki pred. | Pokrycie dlugosci | Tokeny | CER | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `NA2_FT__433927` | 767 | 15 | 1,96% | 3 | 98,1747% | 100,0000% |
| `NA2_FT__433928` | 1438 | 24 | 1,67% | 5 | 98,3310% | 99,3103% |

## Co test rozstrzyga

- Eksport ONNX dziala na `CUDAExecutionProvider`; zadna strona nie zakonczyla
  sie bledem.
- Poprawne dekodowanie calej sekwencji usunelo problem znaku zastepczego.
- Wynik nie jest obciety przez limit 128 tokenow: model sam wyemitowal EOS po
  3 i 5 tokenach.
- Model nie wykonuje wiernej transkrypcji stron i nie zachowuje historycznej
  pisowni, poniewaz praktycznie nie przepisuje tresci.

## Ograniczenia i interpretacja

To diagnostyka dwoch stron, nie pelny benchmark. Jest jednak wystarczajaca do
odrzucenia checkpointu jako full-page OCR: obie odpowiedzi sa krotkimi,
generycznymi podpisami, a ich laczny WER wynosi prawie 100%.

Prawdopodobne przyczyny to niedopasowany cel treningowy image-captioning,
redukcja calej strony do 224x224 oraz architektura z tylko 316 pozycjami tekstu
po 196 tokenach obrazu. Test nie rozstrzyga, ktora przyczyna dominuje.

Model mozna zachowac jako eksperyment VLM lub kandydat do klasyfikacji/podpisow
obrazow, ale wymagalby osobnego zbioru i metryk. Dalsza praca OCR powinna isc w
strone segmentacji strony i recognizera linii albo modelu dokumentowego o
rozdzielczosci i kontekscie zaprojektowanych do transkrypcji.

## Pochodzenie dowodu

- ZIP uzytkownika SHA-256:
  `9dc54c0b78b7d23ccd500e2279b3418a062e8b33a662c82c0a736370e83fe58f`;
- dane: `PiotrSty/impact-print-v2` @
  `a2480fde6f15284701458ff370b81cce50dc5c2d`;
- LM/tokenizer: `SlayerLab/goLLeM-110M-PL-SFT-merged` @
  `a319aedb2b705f72c17823deef509225e59e3b0a`;
- vision: `google/siglip-base-patch16-224` @
  `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`;
- Python 3.13.15, ONNX Runtime GPU 1.23.0, CUDA provider.

Surowe pliki sa w
`experiments/2026-09-25/slayer-vision-onnx-smoke/`.
