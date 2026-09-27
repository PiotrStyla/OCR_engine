# Historical layout development v1 - wynik

Data audytu: 2026-09-27

## Decyzja

`column_aware_v1` nie przechodzi zamrożonych bramek development. Zmienił
kolejność linii na 6 z 15 stron, ale łączny CER i WER są dokładnie takie same
jak dla obecnego `row_major`. Diagnostyczny porządek regionów z PAGE XML także
nie poprawił wyniku. `row_major` pozostaje domyślny i wariant kolumnowy nie
zostanie sprawdzony na zamrożonych 36 stronach testowych.

Wynik pokazuje, że kolejność czytania nie jest teraz głównym ograniczeniem.
Następny eksperyment powinien izolować wykrywanie linii i geometrię cropów.

## Integralność

- ZIP evidence SHA-256:
  `84e2a0d0f666c4191b9b58d2e66deea72a708e681a58ebad94e9fc5d313f716e`;
- wszystkie cztery sumy wymienione w `checksums.json` są zgodne;
- konfiguracja jest identyczna z protokołem zamrożonym przed inferencją;
- wykonano 15/15 stron, bez błędów i brakujących odpowiedzi;
- każda strona była segmentowana raz, a każda linia rozpoznawana raz;
- ZIP nie zawiera obrazów, PAGE XML, referencji, predykcji, wag ani tokenu.

Środowisko: Tesla T4, Python 3.13.15, Torch 2.11.0+cu128,
Transformers 4.57.6, Tokenizers 0.22.2, JiWER 4.0.0 i OpenCV 4.12.0.88.

## Wyniki

Niższa wartość jest lepsza.

| Porządek | CER micro | WER micro | Delta CER |
| --- | ---: | ---: | ---: |
| row-major | 59,9387% | 95,7966% | - |
| column-aware v1 | 59,9387% | 95,7966% | **0,0000 pp** |
| PAGE XML region oracle | 59,9606% | 95,7966% | +0,0219 pp |

Column-aware wygrał CER na jednej stronie, row-major na jednej, a 13 stron
dało remis. Wszystkie kolekcje zmieściły się w limicie regresji 1 pp, lecz
bramki wymagające poprawy całego CER i WER nie zostały spełnione.

## Diagnostyka geometrii

Łączna liczba wykrytych linii, 309, jest bliska 301 liniom referencyjnym, ale
ta suma ukrywa błędy stron: 10 stron ma nadmiar detekcji, 4 niedobór, a tylko
jedna identyczną liczbę. Mediana bezwzględnej różnicy wynosi 6 linii.

Tylko 225 z 309 bboxów, czyli 72,8%, nakłada się na anotowany region tekstowy
PAGE XML. Pozostałe 84 musiały zostać przypisane do najbliższego regionu.
Największy niedobór jakości jest zatem bardziej zgodny z nadmiarowymi lub
źle położonymi wycinkami oraz błędami recognizera niż z samą permutacją linii.

## Następny eksperyment

Na tych samych 15 stronach development należy porównać obecne bboxy pełnej
strony z liniami wyprowadzonymi wewnątrz anotowanych regionów PAGE XML. Ten
oracle geometrii ma odpowiedzieć, czy poprawny podział regionów daje duży zysk
CER. Jeżeli tak, rozwijamy automatyczny detektor layoutu. Jeżeli nie, priorytet
wraca do recognizera. Oracle pozostaje diagnostyczny i nie może być wdrożony.

Stara pisownia, w tym `á` i `ſ`, pozostaje bez modernizacji. Zamrożony test nie
uczestniczy w tym etapie.

Pliki: [oryginalny ZIP](../experiments/2026-09-27/historical-layout-dev-v1/historical-layout-dev-v1-evidence.zip),
[evidence](../experiments/2026-09-27/historical-layout-dev-v1/evidence/)
i [audyt](../experiments/2026-09-27/historical-layout-dev-v1/result-audit.json).
