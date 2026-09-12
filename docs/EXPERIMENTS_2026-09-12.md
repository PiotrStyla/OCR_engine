# Pierwsze pomiary: korekta Fabryka i OCR CPU

Data: 2026-09-12. Dwa oddzielne eksperymenty. Żaden nie dowodzi SOTA.

## 1. Fabryka: czy korekta zachowuje transkrypcję?

8 skonstruowanych przypadków, ten sam produkcyjny prompt, 2 jawnie wybrane modele,
łącznie 16 sekwencyjnych zapytań. Limit 256 tokenów/odpowiedź, temperatura 0, brak retry.
Wysłano tylko jawnie przygotowane teksty diagnostyczne, nie obrazy ani prywatne dokumenty.
Wszystkie odpowiedzi zakończyły się `stop`. Łączne zużycie: 3787 tokenów według API;
jest to zużycie, nie kwota rozliczenia. Nie raportujemy ceny z niezweryfikowanego cennika.

| Wariant | CER na tym zestawie | Zmienione poprawne przypadki |
|---|---:|---:|
| Bez korekty | 2,15% | 0/6 |
| Bielik 11B v3 | 4,62% | 3/6 |
| Qwen3.8 27B | 2,46% | 2/6 |

CER to suma edycji znakowych / suma długości referencji, z zachowaniem zapisu i układu
linii. Dobór zestawu celowo bada nadmierną korektę; proporcja 6 poprawnych i 2 błędnych
wejść nie reprezentuje rozkładu błędów produkcyjnych. To nie ranking jakości modeli.

Najważniejsze obserwacje:

- Oba modele naprawiły `prze kazanie` → `przekazanie`.
- Oba zamieniły widoczne w testowym obrazie `Zolw` na `Żółw`, chociaż dla wiernej
  transkrypcji tej konkretnej strony należy zachować ASCII.
- Bielik zmienił fikcyjny identyfikator `AB-0O1I-2026` na `AB-001I-2026`.
- Dwa przypadki celowo mają identyczny tekst wejściowy i różne źródłowe referencje.
  Korektor tekstowy nie może rozstrzygnąć, czy diakrytyki zginęły w OCR, czy nie było
  ich w obrazie. Przewidywalność językowa nie zastępuje dowodu w obrazie.

Decyzja: korekta pozostaje opcjonalna; surowy odczyt musi być zachowany. Nie włączamy
automatycznie korekty dla wszystkich dokumentów. Potrzebna jest ponowna inferencja
z obrazu lub ocena człowieka przy niepewnych nazwach/liczbach. Nie zmieniamy modelu
domyślnego na podstawie ośmiu przykładów.

Dowody: `experiments/2026-09-12/fabryka/{run.json,responses.jsonl,summary.json}`.
Skrypt: `training/probe_correction.py`; domyślnie dry-run, klucz wyłącznie z ENV.
Prompt i hash przypadków zapisano przed wywołaniami. Sam publiczny model ID oraz
fingerprint nie zastępują niezmiennej rewizji checkpointu.

## 2. Tesseract.js na CPU

Tesseract.js 7.0.0, pol+eng, OEM 1, jeden worker WASM, CPU AMD Ryzen 5 5500U.
Odczyt lokalny. To osobny silnik porównawczy, nie obecny pipeline CRAFT/TrOCR.
Nie uruchomiono żadnego web serwera ani treningu.

| Tor | Próbki | Dokładne transkrypcje | CER | WER | Czas inferencji |
|---|---:|---:|---:|---:|---:|
| Cała syntetyczna strona, PSM 3 | 1 | 1/1 | 0% | 0% | 0,196 s |
| Gotowe wycinki linii, PSM 7 | 12 | 10/12 | 0,81% | 5,45% | 0,474 s łącznie |

Normalizacja: Unicode NFC i ujednolicenie białych znaków, bez zmiany wielkości liter
czy diakrytyków. Dodatkowe puste wiersze na stronie nie są błędem w tej metryce.
370 znaków referencji linii, 3 edycje znakowe. Błędy to `jaźń` → `jain` oraz
dodatkowa spacja przed nawiasem zamykającym. Start workera: 1,180 s, poza czasem
inferencji. Pojedynczy pomiar czasu, nie stabilny benchmark throughput.

Próbki pochodzą z istniejącego katalogu użytkownika `OCR/data/pl_lines_sample`;
obejrzano 12 obrazów i sprawdzono zgodność etykiet. Są syntetyczne i zawierają
powtórzoną transkrypcję. Nie wykazano niezależności od danych treningowych modeli.
Wycinki linii omijają detekcję. Nie sumujemy ich jakości z jakością całej strony.

Dowody: `experiments/2026-09-12/tesseract/{run.json,predictions.jsonl,summary.json}`.
Skrypt i lock: `tools/cpu-baseline/`. Zachowano hashe obrazów, wag i lockfile.
Źródłowe 12 obrazów pozostaje lokalnie; raport nie tworzy publicznego datasetu.

## Poprawka korektora

Ucięta odpowiedź (`finish_reason != stop`) nie może zastąpić pełnej transkrypcji.
Korektor zachowuje wtedy oryginał. Wyłączono wewnętrzne retry SDK, ponieważ aplikacja
ma własny ograniczony mechanizm ponowień. Testy korektora i metryk: 19 passed.

## Następny krok

Mamy działający, tani lokalny baseline. Kolejny benchmark powinien obejmować rzeczywiste
polskie skany, kolumny i tabele z ręcznymi referencjami. Dopiero porównanie z modelem
wizyjnym uzasadni wybór modelu do treningu. Fabryka pozostaje dostawcą korekty
tekstowej/danych nauczyciela; nie jest potwierdzonym endpointem obrazowym ani usługą
uruchamiającą nasz trening. Trening GPU pozostaje nieuruchomiony.
