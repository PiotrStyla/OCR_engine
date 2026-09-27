# Historical layout development v1

## Cel

Eksperyment mierzy wpływ kolejności czytania na OCR całych stron. Nie dobiera
recognizera i nie używa 36 zamrożonych stron testowych. Model `mixed-v3`,
detekcja, wycinki i rozpoznane teksty pozostają identyczne między wariantami.

## Dane

- publiczny dataset `PiotrSty/impact-psnc-polish-ocr`;
- rewizja `c7cb156fb95d2880699c33725bbaf1fbc1008fea`;
- konfiguracja `pages`, split `validation`;
- 15 pełnych stron z pięciu kolekcji, licencja CC-BY-3.0;
- SHA-256 `pages/validation/metadata.jsonl`:
  `536fc014963c2c55626e89f9bd07500826cb930f05ca3f8848352ca80c8856b1`.

Ten split był wcześniej używany do rozwoju recognizera historycznego, dlatego
nie jest niezależnym dowodem jego jakości. Tutaj służy wyłącznie do rozwoju
układu strony przy zamrożonym `mixed-v3`.

Referencje zawierają 14 znaków U+FFFD. Nie są poprawiane na podstawie OCR.
Stara pisownia, w tym `á` i `ſ`, pozostaje bez modernizacji.

## Warianty

1. `row_major`: obecna kolejność od góry do dołu, a w jednym pasie od lewej.
2. `column_aware_v1`: automatyczne kolumny z tych samych bboxów; szerokie
   nagłówki rozdzielają pionowe sekcje strony.
3. `pagexml_region_oracle`: diagnostyczna kolejność regionów z PAGE XML. Ten
   wariant korzysta z anotacji i nigdy nie może zostać wdrożony ani promowany.

Każda strona jest deskewowana i segmentowana dokładnie raz. Każdy wykryty
wycinek jest rozpoznawany dokładnie raz. Warianty różnią się wyłącznie
permutacją tych samych rozpoznanych linii.

## Decyzja

`column_aware_v1` przechodzi etap development tylko wtedy, gdy jednocześnie:

- poprawia łączny CER i WER względem `row_major`;
- żadna kolekcja nie pogarsza CER o więcej niż 1 punkt procentowy;
- nie ma błędów ani brakujących stron.

PAGE XML pokazuje jedynie potencjalny zysk z lepszej kolejności. Jeżeli oracle
nie pomaga, kolejnym ograniczeniem jest detekcja albo recognition. Jeżeli pomaga
znacznie, następny eksperyment powinien rozwijać automatyczny layout parser.

Po pozytywnym wyniku kandydat może zostać zamrożony i dokładnie raz sprawdzony
na 36 stronach testowych. Wynik development nie jest dowodem SOTA.
