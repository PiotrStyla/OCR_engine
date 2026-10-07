# OCR Engine / PolOCRBench

Silnik OCR dla polskich dokumentów oraz rozwijane zaplecze **PolOCRBench**:
publicznego benchmarku transkrypcji całych stron, ekstrakcji tabel i informacji
z dokumentów. Repozytorium zawiera backendy OCR, narzędzia treningowe,
ewaluatory i artefakty eksperymentów. **Nie jest jeszcze ukończonym benchmarkiem
ani potwierdzonym silnikiem SOTA.**

## Aktualny stan: 7 października 2026

### Teraz: DATA ENGINE dla recognizera v3

**Następny krok: rzeczywisty druk zamiast kolejnego identycznego treningu.**
Przygotowany pilotaż ma 12 skanów z uwierzytelnionymi tekstami Wikiźródeł:
8 stron do kandydatów replay i 4 strony z innej książki do osobnej kontroli.
Notebook CPU pobiera dane sam, dekoduje oryginalne DjVu i zachowuje tylko
dokładne kotwice tekstowe, bez modernizowania pisowni. To przygotowanie danych,
nie jeszcze trening ani certyfikowany benchmark.
[Uruchom pilotaż CPU w Colabie](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_printed_replay_pilot_v1.ipynb).
[Źródła, ograniczenia i następna bramka](docs/PRINTED_REPLAY_PILOT_V1_20261007.md).

**V2 zakończony i sprawdzony: pozostaje baza.** Wszystkie trzy warianty trenowały
na T4; 0 z 9 checkpointów przeszło bramki. Najbliższy kandydat ma historyczny
CER 31,90% zamiast 33,03%, ale zwykły 5,53% zamiast 5,33% i łączny 8,42%
zamiast 8,37%. ZIP oraz predykcje zostały zweryfikowane i przeliczone;
przesłany pakiet nie zawiera wag. Nie powtarzaj tego samego treningu.
[Audyt V2, wyniki i dalszy kierunek](docs/RECOGNIZER_REVIEWED_COLAB_V2_RESULT_20261007.md).

**Protokół V2 z ochroną zwykłego druku.** Jeden notebook pobiera dane
automatycznie i porównuje kontrolę, niższy learning rate oraz większy replay.
Poprawiony tor straty ewaluacyjnej ma preflight, a wybór checkpointu uwzględnia
osobno obie domeny. Jeżeli kandydaci nie przejdą bramek, pozostaje model bazowy.
[Uruchom Colab V2](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_recognizer_reviewed_training_v2.ipynb).
[Protokół i pobranie wyników](docs/RECOGNIZER_REVIEWED_COLAB_V2_20261007.md).
To archiwalny, zakończony eksperyment. Duży pakiet wag można zachować na Drive
z ostatniej komórki, jeśli przyszły wariant przejdzie bramki.

**Trening recognizera zakończony, kandydat niepromowany.** Pakiet wag i raport
zweryfikowany w całości. Na 9 historycznych liniach CER poprawił się z 33,03%
do 31,90%, ale na 75 zwykłych wzrósł z 5,33% do 6,17%. Łączny CER pogorszył
się z 8,37% do 8,99%. Zachowujemy model bazowy; nie ma potwierdzenia SOTA.
[Wynik, audyt i następne bramki](docs/RECOGNIZER_REVIEWED_COLAB_RESULT_20261006.md).
[Kompletny model i raport na HF](https://huggingface.co/PiotrSty/slayer-ocr-models/resolve/43583c3932fb0cddd3e6a92333360f1d67ca56b3/experiments/2026-10-06/recognizer-reviewed-colab-v1-result/recognizer-reviewed-colab-v1-result.zip?download=true).
79 testów CPU przeszło; nie powtarzaj tego samego treningu bez zmiany protokołu.

**Archiwalny protokół V1:** osobny eksperyment
z 70 sprawdzonymi liniami i 500 próbkami zwykłego druku, bez 36 linii z rodzin
objętych wykluczeniem. 9 linii z oddzielnej kolekcji i 75 zwykłych linii służy
do kontroli. To trening eksperymentalny, nie otwarcie zamrożonego benchmarku SOTA.
[Protokół, automatyczne dane i sposób uruchomienia](docs/RECOGNIZER_REVIEWED_COLAB_TRAINING_20261006.md).
[Archiwalny notebook Colab V1](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_recognizer_reviewed_training_v1.ipynb).
Nie powtarzaj V1 ani zakończonego V2; bieżący krok to pilotaż danych replay powyżej.

**Pierwszy zwrot z Colaba: trening nie wystartował.** Odczyt bazowy zakończony:
CER 5,33% na 75 zwykłych liniach i 33,03% na 9 historycznych. Tworzenie LoRA
zatrzymał konflikt odziedziczonego TorchAO 0.10.0 z PEFT 0.19.1. Notebook ma
poprawione wersje i sprawdza adaptery, gradienty oraz scalanie przed baseline.
Ten nieudany przebieg nie wytworzył nowych wag; 70 testów CPU przeszło, testy Torch
wymagają środowiska modelowego. [Audyt i poprawka](docs/RECOGNIZER_REVIEWED_COLAB_TRAINING_20261006.md#first-returned-run-failed-before-training).

**Potwierdzenie review zapisane:** 51 pełnych linii ma weryfikację jednego
człowieka, w tym 31 poprawionych propozycji jawnie potwierdzonych w rozmowie.
Surowy eksport, teksty, stare glify i historia pozostają bez zmian. Zostało
13 przypadków: 9 odrzuconych wycinków, 2 niesprawdzone granice i 2 brakujące
decyzje. Spośród 51 przyjętych adnotacji 17 jest objętych wykluczeniem wspólnych
dzieł, a 34 wymagają audytu bibliograficznego. **0 wierszy dopuszczonych do treningu.**
295 testów CPU przeszło; potwierdzenie jest oddzielone od surowych decyzji.
[Nowa wersja v2: potwierdzenie, dane i ograniczenia](docs/RECOGNIZER_EXPANSION_VISUAL_CONFIRMATION_20261006.md).
[Pełny ZIP v2 na HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/51696dca7a9fa0e3dd49c1edb5013d72fff71c3b/data/recognizer-expansion-confirmed-review-v2-20261006/recognizer-expansion-confirmed-review-v2-20261006.zip?download=true).
[Archiwalny import v1](docs/RECOGNIZER_EXPANSION_REVIEW_IMPORT_20261006.md).
Nie trzeba ponownie uruchamiać Colaba ani przeglądać wszystkich 64 linii.

**Audyt dzieł: dwie rodziny przecinają splity.** Tomy „Nowych Aten” oraz obie
„Wyprawy” wymagają wspólnego grupowania ochronnego. Nowa polityka wyłącza
z przyszłego treningu 9 stron: dotyczy to 19 zaakceptowanych linii i 20 nowych
propozycji. Twoje teksty i historia review są zachowane; pozostałe kolekcje
nie są jeszcze bibliograficznie zatwierdzone. 256 testów CPU przeszło.
[Raport, źródła i następne bramki](docs/RECOGNIZER_WORK_FAMILY_AUDIT_20261005.md).
[Audyt na HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/ee918a0c9b7c4e2511eb114fa93f2711a7b8815f/experiments/2026-10-05/recognizer-work-family-audit-v1).
[ZIP ośmiu skanów dowodowych](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/eb55ef61378ead1b5614f83540b121a667c6b5d3/data/recognizer-work-family-visual-evidence-v1-20261005/recognizer-work-family-visual-evidence-v1.zip).
**Nie uruchamiaj ponownie Colaba.** Kontynuuj istniejące review i zwróć
`slayer-recognizer-line-review-31debfc142b5.json`.

**Nowa partia odczytana: 64/64 przez oba modele na T4.** Wszystkie odpowiedzi
zakończyły się EOS, bez błędów i limitów tokenów. Modele różnią się na wszystkich
64 liniach, więc żaden odczyt nie trafia automatycznie do treningu.
Przygotowano review 64 nowych wycinków i 34 pełnych regionów; poprzednie 64
zaakceptowane pary pozostają bez zmian. Zweryfikowano ZIP, ślady generacji
i odtworzono raporty; 231 testów CPU przeszło. **Nie powtarzaj Colaba.**
[Wynik i dokładna instrukcja review](docs/RECOGNIZER_DATA_V3_EXPANSION_RESULT_20261005.md).
[Kompletny ZIP nowych danych review ze skanami](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/b0a44817cf7da627ac27bb82b873415067330b0c/data/recognizer-data-v3-expansion-review-v1-20261005/recognizer-data-v3-expansion-v1-review-data.zip).
[Surowy run, audyt i QA na HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/848e4ddcfa8f6c13219687ae6bbd972ae94cc354/experiments/2026-10-05/recognizer-data-v3-expansion-result-v1).
Zwróć `slayer-recognizer-line-review-31debfc142b5.json`.

**Archiwum wykonanego notebooka: 64 NOWE linie, bez powtórek.**
[Colab: rozszerzenie danych v3](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_recognizer_data_v3_expansion_v1.ipynb).
Notebook służy do odtworzenia już wykonanego eksperymentu, nie jest aktualnym
krokiem do uruchomienia. Dane pobiera z HF bez ręcznego uploadu.
Wykluczono wszystkie wcześniej przejrzane root ID i wycinki. Nowa partia
obejmuje 22 strony i 9 kolekcji treningowych; oba modele już wykonały odczyty,
a teraz potrzebne jest review. Poprzednie 64 pary nie są ponownie przetwarzane.
231 testów CPU przeszło, notebook zwalidowano, a opublikowany ZIP pobrano
i sprawdzono. To generowanie propozycji, nie trening ani pomiar SOTA.
[Protokół nowej partii](docs/RECOGNIZER_DATA_V3_EXPANSION_20261005.md).
[Pełny ZIP wejściowy na HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/18943ce0560a8312ebda38e9e27f7dc883470a18/data/recognizer-data-v3-expansion-v1-20261005/recognizer-data-v3-expansion-v1-input.zip).

**Pilot review zamknięty: 64 aktywne pary.** Wszystkie sześć decyzji follow-up
potwierdza tekst i pełną linię: 58 poprzednich par + 3 potwierdzone teksty
+ 3 nowe wycinki. Zachowano oba eksporty, starą pisownię i nieaktywne stare
wycinki. Pula obejmuje 25 stron i 10 kolekcji treningowych; 215 testów CPU
przeszło. Nie powtarzaj poprzedniego review ani poprzedniego Colaba. Następne bramki to
identyfikacja dzieł/wydań, większy czysty train i niezależny zamrożony dev.
To dane z pojedynczym review, nie gold ani SOTA; trening jeszcze nie ruszył.
[Wynik, pełna historia i zasady scalania](docs/RECOGNIZER_REVIEWED_POOL_20261005.md).
[Pobierz kompletny ZIP 64 par ze skanami i historią](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/619e675e4abb4bfbbb4613d308cbb1509f7ae734/data/recognizer-reviewed-pool-v1-20261005/recognizer-reviewed-pool-v1-20261005.zip).
[Decyzje i raporty na HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/2b33416719ddd60aecb8ed8fbc27e3b2364eeaa9/experiments/2026-10-05/recognizer-reviewed-pool-v1).
Świeże scalenie odtworzyło identyczne pliki; pobraną publikację sprawdzono
plik po pliku i wewnątrz ZIP-u.

**Poprzedni etap:** audyt 89 skanów źródłowych zweryfikował hashe
i wykonał 2739 porównań między splitami. Przy zamrożonych progach nie znalazł
kandydatów na duplikaty; identyfikacja wspólnych dzieł/wydań nadal wymaga
osobnego sprawdzenia. Powstał edytor tylko 6 nierozstrzygniętych przypadków:
3 propozycji tekstu i 3 nowych propozycji wycinków. Nie powtarzamy zaakceptowanych
58 par i nie uruchamiamy jeszcze kolejnego treningu.
[Audyt, follow-up i dokładna instrukcja](docs/RECOGNIZER_SOURCE_AUDIT_AND_FOLLOWUP_20261005.md).
[Komplet danych sześciu przypadków na HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/981802f3b8a50937452be12c09e4fb85395e18b6/data/recognizer-line-remediation-v1-20261005/recognizer-line-remediation-final-20261005.zip).
199 testów CPU przeszło; edytor sprawdzono na desktopie i telefonie.

**Archiwum pierwszego importu review:** 58 par do pilota treningowego
(57 pierwotnych wycinków + 1 nowa para z pełnego kontekstu potwierdzonego przez
użytkownika). Trzy propozycje tekstu pozostają niezatwierdzone, trzy wycinki
odrzucone, a stary wycinek-fragment pozostaje poza treningiem. Zachowano
oryginalny JSON, historię decyzji, surowe odczyty i starą pisownię.
To dane z pojedynczym review, nie niezależny gold ani wynik SOTA. 174 testy CPU
przeszły; powtórzony import odtworzył wyniki, a publikację pobrano i sprawdzono.
[Kompletny ZIP danych po review](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/76dad0d2f727f6b927030e8ec62c5861ef00fe9c/data/recognizer-line-reviewed-candidates-v1-20261005/recognizer-line-reviewed-candidates-v1-20261005.zip).
[Import, zasady i następny etap](docs/RECOGNIZER_LINE_REVIEW_RESULT_20261005.md).

**Pilot wykonany i sprawdzony:** oba modele zwróciły 64 odczyty na T4;
Qwen zakończył 63 EOS i jeden zapętlił na limicie, TrOCR zakończył wszystkie EOS.
Mamy 62 rozbieżności, 1 abstencję i tylko 1 zgodność, która nie zgadza się
z etykietą źródłową. Potwierdzono także błędną parę wycinka i tekstu:
fragment słowa ma etykietę całego zdania. **Nie trenujemy na tych parach bez
review tekstu i granic linii. Nie powtarzaj teraz Colaba.**
Przygotowano edytor dla 64 linii z 35 pełnymi regionami źródłowymi.
[Komplet danych do review na HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/11ee19644073a0fd31e48bdd3c4cf171d761c814/data/recognizer-data-v3-line-review-v1-20261005/recognizer-data-v3-line-review-data.zip).
[Wynik pilota i instrukcja review](docs/RECOGNIZER_DATA_V3_TEACHER_RESULT_20261005.md).

**Notebook do odtworzenia pilota:**
[Colab: recognizer v3 / pilot danych](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/b90ae82fb496d88320e8698cef8debd73785bece/training/colab_recognizer_data_v3_pilot.ipynb).
Wybierz GPU i uruchom wszystkie komórki. Niczego nie wgrywaj: notebook sam
pobierze 64 wycinki z 25 stron i 10 kolekcji treningowych z publicznego HF.
Qwen3-VL i TrOCR wykonają odczyty kolejno, bez etykiet referencyjnych.
Pobierz i przekaż `recognizer-data-v3-teacher-evidence.zip`.

Pula jest oddzielona od kolekcji walidacji, testu i geometry holdout.
Zachowujemy `ſ`, `á`, `ɇ`; zgodność modeli jest tylko propozycją do review,
nie automatyczną etykietą. To przygotowanie czystych danych, **jeszcze nie
trening v3 ani wynik SOTA**. Kontrola granic linii i tekstu oraz audyt
dokumentów/wydań poprzedzą trening. Pakiet HF pobrano i sprawdzono ponownie;
wynik zdalnego runu opisano powyżej. Po audycie i rozszerzeniu edytora
142 testy CPU przeszły; modele nie były uruchamiane lokalnie.
[Protokół i następne bramki](docs/RECOGNIZER_DATA_V3_PILOT_PROTOCOL_20261005.md).

### Zakończone: OCR całych stron

**Wynik v7 (pełne 4 MP):** 15/15 stron na T4, 15 EOS, bez OOM,
limitów i błędów wykonania. CER wynosi **17,52%**, WER **61,63%**, wobec
**63,35% / 132,72%** zachowanego v5. CER poprawił się na 12 stronach,
pogorszył na 3. Dużą część poprawy daje usunięcie dawnej pętli; na 14 stronach
bez niej CER spada z 19,51% do 16,97% (diagnostyka wtórna).
Audyt sprawdził 69 plików i odtworzył metryki; **143 testy CPU przeszły**.
`Poſłał` nadal jest błędne, a referencje nie są gold. To techniczny baseline,
nie zatwierdzony teacher ani SOTA.
[Wynik i ograniczenia v7](docs/FULL_PAGE_VALIDATION_4MP_V7_RESULT_20261003.md).

**Wynik v6:** oba warianty wykonały po 3 strony na T4. Przy 4 MP zniknęła
pętla `44.`: wszystkie odpowiedzi zakończyły się EOS, bez OOM. CER całej
próby spadł z **217,82% do 21,76%**, głównie przez usunięcie tej pętli.
Na dwóch stronach tekstowych CER spadł z **22,10% do 19,45%** (diagnostyka
wtórna, nie nowy benchmark). Oba warianty nadal nie emitują `ſ`, a `Poſłał`
pozostaje błędne. Sprawdzono 40 plików i odtworzono metryki.
Późniejszy v7 potwierdził wykonanie na wszystkich 15 stronach, lecz nie rozwiązał
historycznej pisowni ani nie zatwierdził automatycznego teachera.
[Wynik i ograniczenia v6](docs/FULL_PAGE_RESOLUTION_V6_RESULT_20261003.md).

**Wynik porównania v5:** Qwen3-VL-4B wykonał 15/15 stron na raportowanym T4,
zakończył 14 odpowiedzi EOS i jedną zapętlił na `44.`. Na prowizorycznych
referencjach v2 ma CER **63,35%**, WER **132,72%**, wobec **237,23% / 514,01%**
zapisanego Ovis. Audyt odtworzył metryki i zweryfikował 30 plików ZIP-u.
Historyczne `ſ` nadal są często błędnie odczytywane: profil nie jest gotowym
teacherem ani silnikiem SOTA. Przygotowano kontrolowany test rozdzielczości v6.
[Wynik i ograniczenia v5](docs/FULL_PAGE_COMPARISON_V5_RESULT_20261003.md).

Walidacja v4 objęła wszystkie 15 stron splitu validation: 15/15 wykonań bez OOM,
ale tylko 11 zakończeń EOS i 4 zapętlone odpowiedzi na limicie 4096 tokenów.
Surowy wynik całego testu: **CER 243,74%, WER 515,86%**. EOS v3 działał na dwóch
stronach, lecz ta poprawka nie zapewniła niezawodności na większej próbie.
13/15 referencji zawiera prywatne znaki Unicode; oględziny skanu potwierdziły też
przestawione strofy w jednej referencji. Nie promujemy tego profilu Ovis do
produkcji ani automatycznego teachera. Następny etap: kontrola referencji i
trudnych przypadków, potem porównanie drugiego backendu na sprawdzonych tekstach.
Przygotowano komplet 15 skanów i oryginalnych transkrypcji do review.
[Raport walidacji v4 i instrukcja](docs/FULL_PAGE_VALIDATION_V4_RESULT_20261003.md).
Nie usuwamy nieudanych stron ani powtórzeń z metryk; zachowujemy starą pisownię.

Zaimportowano decyzje dla wszystkich 15 stron: 14 propozycji i 1 weryfikacja
jednego recenzenta. W proponowanych transkrypcjach usunięto 117 znaków PUA/`�`,
zachowując m.in. `ſt`, `ſi`, `ſł` i `á`. Powstał osobny draft z oryginałami i
historią; nie jest to gold ani zbiór treningowy. Te same predykcje nadal mają
4 zapętlenia. [Import korekt i pozostałe kwestie](docs/FULL_PAGE_REVIEW_IMPORT_V1_20261003.md).

**V7 zakończone; nie trzeba powtarzać runu.**
[Notebook do odtworzenia w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/9e1b98c2d429489609b64d4e0f115e535d72df32/training/colab_full_page_validation_4mp_v7.ipynb)
pobiera 15 stron i zweryfikowane predykcje Qwen 1 MP z v5 z HF, bez uploadów.
Uruchamia tylko niezmieniony profil Qwen 4 MP z v6, na wszystkich stronach.
V5 nie jest uruchamiany ponownie.
To porównanie różnych sesji, nie kontrolowany test szybkości; wszystkie błędy
i limity pozostają w metrykach. Referencje nadal nie są gold.
[Zakres, konfiguracja i ograniczenia v7](docs/FULL_PAGE_VALIDATION_4MP_V7_PROTOCOL_20261003.md).
Następny etap DATA ENGINE: hard-example mining dla `ſ`, `á`, `ɇ`, review
wycinków i transkrypcji dyplomatycznych, następnie trening specjalizowanego
recognizera na rozłącznych dokumentach. Nie zamieniamy draftów ani predykcji
Qwen automatycznie w etykiety treningowe.
**Mining v1 gotowe:** 479 propozycji różnic `ſ`, `á`, `ɇ` na 14 z 15 stron,
ze skanami, surowymi odczytami v5/v7 i kolejką review. To diagnostyka walidacji,
nie 479 pewnych błędów ani dane treningowe. **156 testów CPU przeszło**;
sprawdzono też desktop/mobile i eksport historii.
[Kolejka, decyzje i następna bramka treningowa](docs/HISTORICAL_GLYPH_MINING_V1_20261003.md).
**Review zaimportowane:** 12 stron zweryfikowanych wzrokowo przez użytkownika,
3 bez decyzji. Trzy statusy „Do wyjaśnienia” potwierdzono osobną, jawnie
autoryzowaną decyzją. Nie zmieniono tekstów, odczytów ani metryk; strony nadal
nie są treningiem ani niezależnym gold.
[Wynik importu i pozostałe strony](docs/HISTORICAL_GLYPH_REVIEW_V1_20261005.md).

Audyt referencji obejmuje pulę 80 stron poza historycznym testem; do celu
100 stron potrzeba nowych źródeł i kompletnej kontroli transkrypcji.
[Protokół i kolejność następnych etapów](docs/FULL_PAGE_PILOT_V1.md).

### SLAYER-OCR 2.0: DATA ENGINE v2

- Projekt przechodzi na ścieżkę `generator -> teacher ensemble -> consensus ->
  annotation review -> clean dataset -> RF-DETR layout -> recognizer ->
  hard-example mining`. [Architektura, granice twierdzeń i bramki](docs/SLAYER_OCR_2_DATA_ENGINE.md).
- Dane, checkpointy i dowody eksperymentalne są archiwizowane poza repozytorium
  kodu w [publicznej kolekcji OCR experiment registry na Hugging Face](https://huggingface.co/collections/PiotrSty/ocr-experiment-registry-6abca642387af872378a7fee).
  Każda część ma manifest SHA-256 i wskazanie commita źródłowego. [Raport rejestru](docs/SLAYER_OCR_HF_REGISTRY_20260930.md).
- Dodano deterministyczny, lokalny moduł konsensusu layoutu. Zachowuje pełne
  propozycje teacherów, przyjmuje wyłącznie kworum różnych modeli, kieruje
  konflikty do review i eksportuje zaakceptowane weak labels jako COCO JSON dla
  RF-DETR. To infrastruktura danych, nie wynik jakości ani deklaracja SOTA.
- **Gotowy prywatny pilot teacherów:** [uruchom notebook GPU w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_slayer_layout_teacher_pilot_v1.ipynb)
  osobno w trzech świeżych sesjach dla `qwen3-vl-4b`, `doclayout-yolo` i
  `surya-layout2`. Zacznij od domyślnych dwóch stron. Każda sesja pobiera
  przypięte dane i model oraz zwraca osobny ZIP dowodowy bez skanów i referencji.
  Dla smoke testu są też notebooki bez przełączników:
  [DocLayout-YOLO](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_slayer_layout_doclayout_yolo_smoke_v1.ipynb)
  oraz [Surya Layout](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_slayer_layout_surya_smoke_v1.ipynb).
- **Konsensus pełnych runów:** [otwórz notebook v3 CPU w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_slayer_layout_consensus_v3.ipynb),
  wgraj dokładnie trzy ZIP-y teacherów i pobierz COCO weak labels, kolejkę review,
  hard examples oraz pełne provenance. Polityka v3 dopuszcza maksymalnie cztery
  jawne abstencje teachera na 60 stronach; nie liczy ich jako pustych głosów,
  zachowuje quorum dwóch niezależnych modeli i kieruje każdą taką stronę do
  hard-example mining. Dwustronicowy smoke ujawnił konflikty klas i poziomu
  szczegółowości.
  Pełny przebieg 60 stron zakończył się technicznie poprawnie: 161 obiektów
  przyjęto, 371 skierowano do review, a 58/60 stron oznaczono jako hard examples.
  Pełna adjudykacja została zakończona: 143 obiekty review przyjęto (w tym 5
  relabeli), 228 odrzucono, a clean candidate zawiera 304 obiekty na 60
  stronach. Eksporter RF-DETR tworzy zweryfikowany split 48/12 bez wspólnych
  kolekcji i dokładnych hashy obrazów. Artefakty eksperymentalne są publicznie
  katalogowane w [OCR experiment registry](https://huggingface.co/collections/PiotrSty/ocr-experiment-registry-6abca642387af872378a7fee).
  [Raport pełnego przebiegu](docs/SLAYER_LAYOUT_CONSENSUS_FULL_RESULT_20260929.md).
  [Wynik smoke i decyzja](docs/SLAYER_LAYOUT_CONSENSUS_SMOKE_RESULT_20260928.md).
- Polityka konsensusu v3 wykrywa konflikty skali bez mnożenia głosu jednego
  teachera i jawnie rejestruje ograniczone abstencje. Zamrożono też definicje
  klas layoutu. Walidator
  `training.adjudicate_layout_consensus` odrzuca niepełne decyzje i eksportuje
  wersjonowany kandydat JSONL/COCO. Następnie
  `training.build_layout_rfdetr_dataset` sprawdza candidate i obrazy po SHA-256,
  buduje rozłączny kolekcyjnie format `train/valid` zgodny z RF-DETR oraz zapisuje
  manifest, provenance i sumy kontrolne.
- **Następny run:** [otwórz przypięty trening RF-DETR Small w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_slayer_rfdetr_layout_v1.ipynb),
  wybierz GPU, uruchom wszystko i wgraj prywatny ZIP przygotowany przez
  eksporter. Notebook sprawdza hash całego ZIP-a i wszystkich plików, nie
  publikuje danych, zapisuje metryki per klasa oraz zwraca osobno ZIP dowodowy i
  ZIP modelu. Wynik jest pilotem deweloperskim, nie pomiarem końcowym.
- **Wynik pilota RF-DETR:** trening zakończył się early stopping po epoce 30,
  a najlepszy checkpoint EMA z epoki 15 osiągnął `mAP@50:95 = 0,3273` na
  12 stronach z czterech rozłącznych kolekcji. Najmocniejsza jest klasa
  `figure` (`EMA AP = 0,8238`); `table` i `marginalia` pozostają słabe, a trzy klasy
  nie mają przykładów validation. Checkpoint przechodzi bramkę techniczną, ale
  jeszcze nie jakościową. [Pełny raport](docs/SLAYER_RFDETR_LAYOUT_V1_RESULT_20260929.md).
- **Audyt najlepszego checkpointu:** [otwórz prywatny audyt RF-DETR w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_slayer_rfdetr_layout_audit_v1.ipynb),
  wybierz GPU i uruchom wszystko. Notebook prosi osobno o ZIP danych, modelu i
  dowodów treningowych, sprawdza ich pełne hashe, ponownie mierzy zapisany
  checkpoint na 12 stronach oraz pobiera jeden prywatny ZIP z predykcjami,
  porównaniami GT/model i kolejką hard examples. Niczego nie publikuje.
  Audyt został ukończony: checkpoint dokładnie odtwarza `mAP@50:95 = 0,3273`,
  ale przy progu diagnostycznym 0,25 wszystkie 12 stron zawiera FP lub FN.
  Nakładki ujawniły także niespójną granulację ramek i prawdopodobne braki GT,
  dlatego kolejną bramką jest ponowna adjudykacja 12 stron, a nie retrening na
  tych samych etykietach. [Wynik audytu](docs/SLAYER_RFDETR_LAYOUT_AUDIT_V1_RESULT_20260929.md).
- **Adjudykacja GT:** `python -m training.build_rfdetr_layout_adjudication
  --dataset <dataset-dir> --audit <audit-dir> --audit-archive-sha256 <sha256>
  --output <review-dir>` buduje offline editor 12 stron z warstwami
  oryginalnego GT i predykcji RF-DETR, edycją ramek, statusem strony oraz
  eksportem decyzji JSON. Repo zawiera generator i
  [zamrożoną politykę anotacji](docs/SLAYER_LAYOUT_ANNOTATION_POLICY_V1.md).
- **Po zakończeniu review:** `python -m training.apply_rfdetr_layout_gt_review
  --review <final-review.json> --review-source <review-dir>
  --dataset <dataset-dir> --output <corrected-v2-dir>
  --archive <corrected-v2.zip>` odrzuca niepełny lub niespójny eksport, zachowuje
  treningowy split bez zmian i tworzy poprawiony validation COCO z pełnym
  lineage oraz SHA-256. Audytor rozróżnia odtworzenie starego GT od ponownego
  pomiaru tego samego checkpointu na poprawionym GT; stare `mAP=0,3273` nie jest
  wtedy traktowane jako oczekiwany wynik. [Protokół v2](docs/SLAYER_RFDETR_CORRECTED_GT_V2.md).
  Po zbudowaniu ZIP-a uruchom
  [notebook ponownego audytu v2 w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_slayer_rfdetr_corrected_gt_v2.ipynb).
  Reanaliza została ukończona: po 18 dodaniach, 3 usunięciach, 4 relabelach i
  13 korektach geometrii ten sam checkpoint osiąga `mAP@50:95 = 0,2570`, wobec
  `0,3273` na starym GT. Przy stałym progu F1 lekko rośnie z `0,4583` do
  `0,4780`, ale wszystkie 12 stron nadal jest hard examples. Checkpoint nie
  przechodzi bramki jakości; następny etap to rozszerzenie i zbalansowanie
  zweryfikowanych danych, nie strojenie jednego progu na dev secie.
  [Raport reanalizy v2](docs/SLAYER_RFDETR_CORRECTED_GT_V2_RESULT_20261002.md).
- **Pełny pilot 60 stron:** DocLayout-YOLO i Surya ukończyły kompletne runy bez
  błędów. Poprawiony Qwen v3 ukończył 56/60 stron; cztery błędy generacji są
  zachowane jako jawne abstencje w polityce konsensusu v3.
  Zachowaj istniejące ZIP-y z notebooków
  [DocLayout-YOLO](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_slayer_layout_doclayout_yolo_full_v1.ipynb),
  [Surya Layout](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_slayer_layout_surya_full_v1.ipynb).
  Notebook Qwen v3 ponawia odpowiedź o złym JSON-ie, zachowuje surowe odpowiedzi
  przy błędzie i odrzuca wadliwe ramki bez utraty całej strony. Domyślnie
  konsensus odrzuca niekompletny run; zamrożona polityka v3 dopuszcza wyłącznie
  cztery udokumentowane abstencje bez detekcji. Przed pobraniem notebook sprawdza
  commit, a wynik nazywa `qwen3-vl-4b-evidence-v3.zip`. Notebooki są zablokowane
  na właściwym teacherze i `PAGES = 60`; ZIP zapisuje przypięty commit kodu i nie
  zawiera skanów ani tekstu referencyjnego.
- Zamrożona konfiguracja pilota znajduje się w
  [`experiments/2026-09-28/slayer-layout-teacher-pilot-v1/config.json`](experiments/2026-09-28/slayer-layout-teacher-pilot-v1/config.json).
  Wykonano smoke oraz pełny prywatny przebieg 60 stron. Obrazy, predykcje i wynik
  konsensusu pozostają prywatne do czasu osobnej decyzji publikacyjnej oraz
  przeglądu licencji.

### Eksperymentalny model SLAYER Vision ONNX

- Paczka IR9/opset 18 przechodzi pełną statyczną walidację trzech grafów ONNX,
  ale nie jest jeszcze kandydatem na domyślny silnik OCR. Brakuje poprawnego
  tokenizera, kompletnej umowy preprocessingu i generacji oraz pomiaru CER/WER.
  Obecny `tokens_decoded.json` ma 167 tokenów z U+FFFD, więc dekodowanie przez
  konkatenację może psuć polskie znaki. [Audyt i bramka integracji](docs/SLAYER_VISION_ONNX_IR9_AUDIT_20260925.md).
- **Gotowy test wykonawczy:** [uruchom dwie zamrożone strony IMPACT w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_slayer_vision_onnx_smoke.ipynb).
  Notebook używa przypiętych rewizji modelu, tokenizera, SigLIP i danych,
  dekoduje pełną sekwencję tokenów oraz pobiera ZIP dowodowy. To diagnostyka;
  dopiero wynik pokaże, czy model ma jakąkolwiek zdolność wiernej transkrypcji.
- **Wynik negatywny:** na dwóch stronach model osiągnął CER 98,28% i WER
  99,57%, generując krótkie podpisy zamiast transkrypcji. Checkpoint nie będzie
  rozwijany jako OCR pełnych stron. [Wynik, dowody i decyzja](docs/SLAYER_VISION_ONNX_SMOKE_RESULT_20260925.md).

### Google Colab (dodano 24 września)

- **Zakończony holdout geometrii:** [12 nowych kolekcji w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_geometry_holdout.ipynb).
  Dla mixed-v3 prostokąt uzyskał CER **37,9231%**, a line-band **38,0092%**;
  przedział bootstrap 95% różnicy przecina zero. Prostokąt pozostaje wariantem
  domyślnym, a strojenie geometrii na tej próbie jest zamknięte.
  [Zweryfikowany wynik i decyzja](docs/GEOMETRY_HOLDOUT_RESULT_20260924.md).
- **Następny etap:** przegląd 45 znaków prywatnego alfabetu na skanach oraz
  trening recognizera dla historycznego druku. Starej pisowni, w tym `á` i `ſ`,
  nie modernizujemy; pozostaje częścią podstawowej metryki.
- **Gotowy korpus pilota recognizera:** 258 linii train i 139 validation z
  rozłącznych kolekcji; linie z PUA/U+FFFD są w kwarantannie. Konfiguracja nie
  używa 12 kolekcji holdoutu geometrii ani końcowego testu.
  [Uruchom trening w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_historical_recognizer_v1.ipynb) oraz zobacz
  [protokół i warunki promocji](docs/HISTORICAL_RECOGNIZER_V1.md).
- **Recognizer v1 odrzucony:** historyczny CER poprawił się z 34,85% do
  27,19%, lecz `real-lines-v1` pogorszył się o 3,99 pp, a EHRI o 11,21 pp.
  Wagi nie zostały opublikowane. [Wynik i następny eksperyment](docs/HISTORICAL_RECOGNIZER_V1_RESULT_20260927.md).
- **Recognizer v2 zamrożony przed treningiem:** jeden test replay łączy
  historyczne linie wyłącznie z treningowymi splitami EHRI i syntetycznego
  druku; progi promocji pozostają bez zmian. [Uruchom w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_historical_recognizer_v2.ipynb)
  · [wynik v2](docs/HISTORICAL_RECOGNIZER_V2_RESULT_20260927.md)
  · [protokół v2](docs/HISTORICAL_RECOGNIZER_V2.md). V2 przechodzi zamrożone
  progi po naprawie porównania dokładnej granicy `+2 pp`; wagi nie zostały
  opublikowane.
- **Pełnostronicowy A/B zamrożony:** 36 niezależnych stron, jeden przebieg
  detektora OpenCV i identyczne cropy dla mixed-v3 oraz prywatnego recognizera
  v2. [Uruchom w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_historical_full_page_ab_v1.ipynb)
  · [protokół](docs/HISTORICAL_FULL_PAGE_AB_V1.md)
  · [wynik](docs/HISTORICAL_FULL_PAGE_AB_V1_RESULT_20260927.md). V2 poprawił
  WER o 0,6480 pp, ale pogorszył główny CER o 0,6353 pp i nie przeszedł
  zamrożonych bramek. Mixed-v3 pozostaje domyślnym recognizerem całej strony.
- **Zakończony development kolejności czytania:** 15 publicznych stron validation,
  jeden przebieg detektora i recognizera oraz trzy porządki tych samych linii:
  obecny row-major, automatyczny column-aware i diagnostyczny PAGE XML oracle.
  [Uruchom w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_historical_layout_dev_v1.ipynb)
  · [zamrożony protokół](docs/HISTORICAL_LAYOUT_DEV_V1.md)
  · [wynik](docs/HISTORICAL_LAYOUT_DEV_V1_RESULT_20260927.md). Column-aware
  zmienił kolejność na 6 stronach, ale CER i WER pozostały identyczne;
  PAGE XML oracle także nie pomógł. Row-major pozostaje domyślny, a następnym
  celem jest geometria detekcji. Końcowe 36 stron nie uczestniczyło w wyborze.
- **Zakończony eksperyment:** [granice wierszy i interpunkcja w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_geometry_bands.ipynb).
  33 nowe wycinki i 30 niezmienionych; automatyczne pobieranie pełnej paczki,
  kontrola wersji bibliotek i ZIP wyników. [Metoda](docs/AUTO_GEOMETRY_BANDS_20260924.md).
  Mixed-v3: CER 29,40% wobec 29,44% poprzednio, tylko jeden błąd mniej.
  Brak podstaw do zmiany wariantu domyślnego. [Wynik i ograniczenia](docs/AUTO_GEOMETRY_BANDS_RESULT_20260924.md).
  Następny krok: osobna próbka walidacyjna, bez dalszego strojenia na tych 63 liniach.
- **Poprzedni eksperyment:** [porównanie automatycznych ramek na 63 liniach](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_auto_geometry.ipynb).
  31 propozycji nowych wycinków i 32 jawne powroty do oryginału; obie wersje
  porównywane na tych samych referencjach. [Metoda i ograniczenia](docs/AUTO_LINE_GEOMETRY_20260924.md).
  Mixed-v3: CER 31,64% → 29,44%, ale 9 regresji; robocze referencje, nie benchmark.
  [Zweryfikowany wynik i audyt](docs/AUTO_GEOMETRY_RESULT_20260924.md).
- [Otwórz test 63 linii w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_body_dev_diagnostic.ipynb).
- [Otwórz kontrolę geometrii A/B w Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_body_crop_ab.ipynb).

Wybierz środowisko GPU i uruchom wszystkie komórki. Dane są pobierane automatycznie
lub osadzone; nie dodawaj osobnego datasetu. Ostatnia komórka pobiera ZIP wyników,
który pozostaje też w `/content`. Zachowano modele, dane i metryki z Kaggle,
w tym historyczną pisownię. To diagnostyka, nie trening. Wersje Colab sprawdzono
lokalnie testami struktury i metryk; inferencja GPU wymaga uruchomienia w Colab.

### Dane i ewaluacja

- **Podzadanie A, transkrypcja:** zamrożony historyczny podzbiór IMPACT,
  36 stron testowych z 3 kolekcji oraz pula 2531 regionów treningowych.
- **Odtwarzalne dane:** importer przypiętej paczki Hugging Face sprawdza SHA-256,
  rozdzielenie kolekcji i brak wspólnych hashy obrazów train/test. Osobne kopie
  PNG zachowują piksele oryginalnych TIFF-ów i mają przenośne ścieżki.
- **Ewaluator v1.1:** CER/WER oraz przybliżona ocena struktury Markdown;
  brakujące i błędne odpowiedzi dostają zero punktów za strukturę.
- **Baseline CPU:** zapisane predykcje wszystkich stron, metadane silnika,
  hashe wag, wyniki obu protokołów i sumy kontrolne artefaktów.
- **Weryfikacja:** 17 testów importera, konwertera i ewaluatorów przeszło.
  To testy tego zakresu zmian, nie deklaracja uruchomienia całego zestawu testów.

### Wynik baseline'u CPU

Tesseract.js 7.0.0, `pol+eng`, OEM 1, PSM 3, jeden worker CPU:

| Zbiór | Strony | CER micro | WER micro |
| --- | ---: | ---: | ---: |
| IMPACT historyczny, całość | 36 | **34,94%** | **81,23%** |
| NA2_FT | 15 | 27,35% | 73,22% |
| Nowiny_z_Rakuz_FT | 15 | 43,93% | 87,78% |
| Powodzenia_FT | 6 | 34,83% | 87,94% |

Niższe CER/WER oznacza mniej błędów. Odczyt trwał 392,10 s; wszystkie strony
zwróciły tekst, ale nie oznacza to poprawnej transkrypcji. Jest to nowy pomiar,
a nie odtworzenie wcześniejszej konfiguracji natywnego Tesseract `pol tessdata_best`.

**Ograniczenie referencji:** 86 znaków zastępczych U+FFFD na 23 stronach oraz
624 znaki prywatnego zakresu Unicode na 33 stronach wymagają przeglądu adnotacji.
Zamrożonych referencji nie poprawiano na podstawie predykcji. Sam brak wspólnych
hashy nie wyklucza podobnych skanów ani obecności dokumentów w pretreningu modeli.

- [Wyniki, surowe predykcje i metadane](experiments/2026-09-19/impact-tesseractjs-png/README.md)
- [Odtworzenie benchmarku i baseline'u](docs/POLOCRBENCH_REPRODUCTION.md)
- [Zmiany protokołu ewaluacji](docs/POLOCRBENCH_EVALUATOR_2026-09-19.md)
- [Zamrożone manifesty PolOCRBench](benchmarks/polocrbench/README.md)
- [Lokalny panel audytu adnotacji](tools/annotation-review/README.md): skan obok
  transkrypcji, kolejka podejrzanych znaków i eksport historii propozycji zmian.
  Panel nie modyfikuje zamrożonych referencji.
- [Uzgadnianie recenzji i nowa wersja manifestu](tools/annotation-review/README.md#build-a-reviewed-candidate):
  zgodność dwóch recenzentów, raport konfliktów oraz pełna historia zmian.
  Wynik jest kandydatem do wydania, bez automatycznej publikacji.
- [Runner Krakena na GPU i walidator zgłoszeń A](docs/KRAKEN_REPRODUCIBLE_BASELINE.md):
  jawne hashe obu modeli, predykcje każdej strony i metadane środowiska.
  Test CUDA na Kaggle Tesla T4 zakończył się: 36/36 stron, bez błędów wykonania.
- [Notebook Kaggle: odtwarzalny baseline Krakena](training/kaggle_polocrbench_kraken_reproducible.ipynb):
  włącz Internet i GPU T4, uruchom Run All, pobierz ZIP wyników. Notebook wykonuje
  smoke-test jednej strony przed pełnym pomiarem 36 stron; nie trenuje modelu.

### Wynik Krakena i diagnostyka GPU

Kraken 7.1.1 z recognizerem i segmenterem fine-tunowanymi na EHRI osiągnął
**CER 79,56% / WER 107,61%**, wobec **34,94% / 81,23%** dla Tesseracta.
Referencje i hashe zdekodowanych pikseli są zgodne między przebiegami.
WER może przekraczać 100% przez nadmiarowe słowa. To wynik konkretnej
konfiguracji, nie ocena wszystkich modeli Krakena.

- [Raport przebiegu Kaggle](docs/KRAKEN_KAGGLE_RESULT_20260921.md).
- [Diagnoza segmentacji i wycinków](docs/KRAKEN_DIAGNOSTICS_20260921.md).
- [Kontrola wejścia recognizera i alfabetu](docs/KRAKEN_INPUT_CHECK_20260921.md).
- [Notebook diagnostyczny v2](training/kaggle_kraken_diagnostics.ipynb)
  oraz [pełny kod komórki](training/kaggle_kraken_diagnostics.py): uruchom kod
  jako jedną nową komórkę w tej samej sesji Kaggle po zakończonym baseline.
  Nie uruchamiaj ponownie Run All. Wymagane są zachowane obrazy i wyniki
  w `/kaggle/working/polocrbench-kraken-*/`. Wynik: `kraken-input-check-*.zip`.

Diagnostyka trzech stron tytułowych potwierdza błędy segmentacji i rozpoznawania.
175 rzeczywistych wycinków zgadza się z wcześniejszym eksportem; sprawdzone
podglądy po normalizacji zachowują czytelny tekst. Alfabet modelu nie obejmuje
części znaków historycznego druku. Nie wykluczono problemów checkpointu ani
całej ścieżki inferencji. Poniższa kontrola EHRI osłabia hipotezę globalnie
uszkodzonego checkpointu, ale nie dowodzi poprawności wszystkich konfiguracji.

### Kontrole EHRI i historycznego druku

- **EHRI, jedna strona:** CER **2,81%** z geometrią ALTO i **7,60%**
  z przewidywaną segmentacją. Możliwe nakładanie z treningiem lub walidacją;
  to kontrola działania, nie niezależny benchmark.
  [Raport](docs/EHRI_CONTROL_RESULT_20260921.md) i
  [notebook](training/kaggle_ehri_control.ipynb).
- **TrOCR, 15 regionów deweloperskich:** Microsoft base-printed uzyskał
  CER **69,41%**, a PiotrSty mixed-v3 **23,42%**. Duża część różnicy wynika
  z wielkości liter: dodatkowy pomiar po zamianie na małe litery daje
  odpowiednio **25,95%** i **20,25%**. Podstawowych wyników nie zastępujemy
  tym pomiarem diagnostycznym.
  [Raport](docs/PRINTED_DEV_RESULT_20260921.md),
  [instrukcja](docs/PRINTED_DEV_CONTROL.md) i
  [notebook Kaggle](training/kaggle_printed_dev_control.ipynb).

Próbka druku obejmuje głównie nagłówki z trzech stron; referencje zawierają
problematyczne znaki Unicode. Wyników regionów nie porównujemy bezpośrednio
z benchmarkiem całych stron. Następny krok: większy zestaw deweloperski
zwykłych wierszy tekstu, ze sprawdzoną geometrią i transkrypcjami oraz podziałem
na dokumenty. Zamrożony test pozostaje poza doborem modeli i treningiem.

[Przygotowanie przeglądu zwykłego tekstu](docs/BODY_DEV_REVIEW.md): lokalna
próbka 19 regionów z 10 stron zawiera 96 wierszy referencyjnych. Powstały 63
niezweryfikowane propozycje wycinków; 8 regionów wymaga ręcznego podziału.
To materiał do przeglądu adnotacji, jeszcze nie nowy zbiór do ewaluacji.

### Lokalne próby gazet

- `training/sample_us_pd_newspapers.py`: mała próbka tekstowego datasetu,
  zachowanie surowego OCR, diagnostyka Unicode i pochodzenie danych.
- `training/probe_newspaper_correction.py`: ograniczona próba dwóch stron
  przez OpenRouter, domyślnie dry-run; wykonanie wymaga `--execute` i klucza.
  Używa płatnego modelu `openai/gpt-4o-mini`, nie wariantu `:free`.
- `training/segment_newspaper_columns.py`: podział według ręcznie określonych
  granic kolumn, wycinki bez zmiany pikseli i diagnostyka separatorów OpenCV.
  Nie jest automatyczną segmentacją artykułów.

Lokalne skany, wycinki i odpowiedzi API pozostają poza repozytorium (`data/`).

### Zakres docelowy

Podzadania: **A** transkrypcja do Markdown, **B** tabele do HTML z oceną
TEDS, **C** pola dokumentu do JSON z oceną field-level F1. Ewaluatory A/B/C,
wynik zbiorczy, format zgłoszeń zgodny z AmuEval (`out.tsv`) oraz walidacja
deklaracji tracków (constrained, open, zero-shot/API) są zaimplementowane —
zobacz [protokół ewaluacji](docs/POLOCRBENCH_SUBTASKS_BC.md) i
[zamrożony prompt tracku zero-shot](benchmarks/polocrbench/prompts/zero_shot_prompt_v1.md).
Dalszej pracy wymagają: zbiory podzadania B/C (współczesne dokumenty, pismo
ręczne, pełne strony z tabelami i polami) wraz z anotacją, ukryty Test B oraz
publiczny leaderboard. Zaimplementowane miary nie są jeszcze zwalidowane jako
ostateczne metryki rankingowe.

### Podzadania B/C, wynik zbiorczy i tracki

```bash
python -m training.table_eval --manifest tables.jsonl --predictions run-b.jsonl --output b.json
python -m training.kie_eval --manifest kie.jsonl --predictions run-c.jsonl --output c.json   # --dump-schemas wypisuje schemat
python -m training.composite_score --report-a a.json --report-b b.json --report-c c.json --output composite.json
python -m training.submission_tsv --mode pack --subtask A --in-tsv in.tsv --predictions run.jsonl --out-tsv out.tsv
python -m training.validate_submission --manifest M.jsonl --predictions run.jsonl --subtask C --meta submission_meta.json
```

Wynik zbiorczy to średnia znormalizowanych wyników zgłoszonych podzadań
(1 − CER, TEDS, F1); testy normalizacji i miar: `tests/test_table_eval.py`,
`tests/test_kie_eval.py`, `tests/test_composite_score.py`,
`tests/test_submission_tsv.py`.
Baseline'y (Surya 2, Qwen-VL na Kaggle, model API z promptem zero-shot v1)
i raport kosztów per strona: [docs/POLLOCR_BASELINES_BC.md](docs/POLLOCR_BASELINES_BC.md)
— pierwszy pomiar `gpt-4o-mini` na 4 stronach syntetycznych: composite 0,781.
Podziały train/test A/Test B, bramka integralności (`training/check_split_integrity.py`)
i wykryty nakład tekstu train↔testA: [docs/POLLOCR_SPLIT_POLICY.md](docs/POLLOCR_SPLIT_POLICY.md).
Wydanie zbioru (HuggingFace + pakiety AmuEval + stub leaderboardu):
[docs/POLLOCR_RELEASE.md](docs/POLLOCR_RELEASE.md), budowniczy `training/build_release.py`,
strona `tools/leaderboard/index.html`.

### Generator dokumentów syntetycznych (A+B+C)

```bash
python -m training.generate_documents --output data/polocrbench-synth-v1 --count 1200 --seed 20260922 --split train
```

Generuje strony faktur, umów, pism urzędowych i formularzy z kompletnym ground
truth dla trzech podzadań naraz (`manifest-A/B/C.jsonl`): transkrypcja
Markdown, HTML-e tabel (z colspan) i pola KIE zgodne ze schematami
`training.kie_eval.SCHEMAS`. Szablony treści (`training/document_templates.py`)
są czyste i deterministyczne (NIP/PESEL z poprawnymi sumami kontrolnymi, kwoty
spójne arytmetycznie z pozycjami); degradacje: `clean`, `scan`, `photo`,
`print_scan`, `compress`, z parametrami w `generation.json`. Test B może
wstrzymać typy i degradacje przez `--types`/`--degradations`. Ten sam seed
odtwarza identyczne pliki. Testy: `tests/test_generate_documents.py`
(m.in. samopunktacja wygenerowanego zbioru przez wszystkie ewaluatory = 1.0).

## Biblioteka OCR

> Aktualizacja po audycie (2026-09-12): zobacz [plan CPU i zdalnych testów](docs/CPU_REMOTE_PLAN.md).
> CLI respektuje ENV, a jawne flagi mają pierwszeństwo. Tablice wejściowe muszą być RGB uint8.
> Wyniki zawierają źródłowy numer strony i geometrię po odwrotnym mapowaniu deskew.
> Korekta zachowuje `raw_text`/`raw_confidence`; zmieniony tekst nie dziedziczy pewności OCR.
> Niestandardowe checkpointy Paddle są odrzucane, dopóki nie ma ich jawnej integracji.
>
> [Pierwsze pomiary CPU i Fabryki](docs/EXPERIMENTS_2026-09-12.md): Tesseract stanowi
> osobny baseline; na małym teście korekta tekstowa zwiększała łączny CER.
> Wyniki nie są potwierdzeniem SOTA. Ucięte odpowiedzi korektora są odrzucane.

Silnik OCR w Pythonie: **detekcja linii** (CRAFT lub OpenCV) + **TrOCR** (rozpoznawanie),
z wsparciem języków **PL/EN**, routingiem języka per linia i **korektą tekstu przez
Fabryka API** (Bielik).

## Architektura

```
obraz → preprocess (deskew) → detekcja linii → routing języka (PL/EN)
      → TrOCR (rozpoznawanie) → [korekta przez Fabryka/Bielik] → OcrResult
```

- **Detekcja**: auto-wybór — **CRAFT** (`craft-text-detector`) gdy dostępny,
  w przeciwnym razie **fallback OpenCV** (morfologia + kontury; `ocr/opencv_detector.py`).
- **Rozpoznawanie**: TrOCR (`microsoft/trocr-base-printed` dla EN; lokalny fine-tune QLoRA dla PL).
  Opcjonalnie **Kraken** (baseline segmentation + `.mlmodel`) dla maszynopisów/historycznych.
- **Korekta tekstu**: [Fabryka AI](https://fabryka.ai) — Bielik (polski LLM) naprawia błędy
  OCR (diakrytyki, pocięte słowa, interpunkcja). Opcjonalna, wymaga klucza API.
- **Routing języka**: heurystyka polskich diakrytyków + `langdetect`. Bez
  `force_language` linie idą najpierw modelem EN, a te wykryte jako PL są
  **re-rozpoznawane modelem PL** w drugim przebiegu (gdy model PL istnieje).
- **Porządek czytania**: grupowanie linii w poziome pasy (top→bottom), wewnątrz left→right.

> **Uwaga PL:** oficjalne modele TrOCR są tylko angielskie. Polskie diakrytyki
> wymagają fine-tunu — patrz [`training/README.md`](training/README.md) (QLoRA, wg wskazówek
> [Slayer](https://slayer.fabryka.ai/trening)). Alternatywa bez treningu: włączyć korektę
> tekstu przez Fabryka API (Bielik) — patrz niżej.

## Instalacja

```bash
pip install -e .
# dev (testy):
pip install -e ".[dev]"
# detektor CRAFT (wymaga Py <3.11 — stary pin opencv):
pip install -e ".[craft]"
# korekta przez Fabryka API:
pip install -e ".[correct]"
# backend Kraken (historyczne dokumenty / maszynopis):
pip install -e ".[kraken]"
# trening PL (GPU):
pip install -e ".[train]"
```

> Bez `craft-text-detector` działa automatyczny detektor OpenCV — wystarczy do
> dokumentów i skanów na jasnym tle.

## Użycie

### Biblioteka

```python
from ocr import recognize, OcrEngine, OcrConfig

# jednorazowo
result = recognize("dokument.png")
print(result.text)
for line in result.lines:
    print(line.text, line.bbox, line.language, line.confidence)

# wiele obrazów (współdzielenie modeli)
with OcrEngine(OcrConfig(force_language="pl")) as engine:
    for path in paths:
        print(engine.recognize(path).text)

# PDF — lista wyników per strona
with OcrEngine() as engine:
    pages = engine.recognize_pdf("dokument.pdf", pages="1-3", dpi=300)
    for i, page in enumerate(pages, 1):
        print(f"--- strona {i} ---\n{page.text}")
```

### CLI

```bash
ocr recognize dokument.png                 # tekst
ocr recognize dokument.png --lang pl --json # JSON z bboxami
ocr recognize dokument.pdf                  # PDF — wszystkie strony
ocr recognize dokument.pdf --pages 1-3,5    # wybrane strony
ocr recognize dokument.png --correct        # korekta tekstu przez Fabryka/Bielik
ocr recognize dokument.png --no-deskew
ocr recognize dokument.png --device cpu
ocr check-fabryka                           # sprawdź połączenie z Fabryka API
```

### Korekta tekstu przez Fabryka API (Bielik)

Korekta naprawia typowe błędy OCR (zamiana diakrytyków ą→a, ł→l, ę→e, pocięte słowa)
przy użyciu polskiego modelu LLM Bielik przez [Fabryka AI](https://fabryka.ai).

Endpoint jest wybierany automatycznie na podstawie prefixu klucza:
- `fab_live_...` → `https://fabryka.ai/v1` (lab)
- `sk-fab-...` → `https://router.fabryka.ai/v1` (router)

```bash
# 1. Pobierz klucz API z https://fabryka.ai (sekcja "Get API key")
# 2. Ustaw zmienną środowiskową:
export FABRYKA_API_KEY=fab_live_...   # lub sk-fab-...

# 3. Sprawdź połączenie:
ocr check-fabryka

# 4. Rozpoznaj z korektą:
ocr recognize dokument.png --correct
```

Lub w Pythonie:

```python
from ocr import OcrEngine, OcrConfig

config = OcrConfig(correct_text=True)  # wymaga FABRYKA_API_KEY w env
with OcrEngine(config) as engine:
    result = engine.recognize("dokument.png")
    print(result.text)  # tekst po korekcie
```

Wymaga: `pip install ocr-engine[correct]` (pakiet `openai`). Brak klucza → korekta
pominięta z ostrzeżeniem (graceful degradation). Błąd API → zwraca oryginalny tekst
z retry (429/502/503/504 z exponential backoff).

### Próg pewności (confidence)

Każda linia ma `confidence` ∈ [0,1] z rozpoznawania TrOCR. Próg pozwala
flagować niskopewne linie i ograniczyć korektę tylko do nich (oszczędność API):

```bash
# oznacz linie z confidence < 0.7 flagą "low_confidence" w JSON:
ocr recognize dokument.png --json --confidence-threshold 0.7

# korekta Fabryka TYLKO dla niskopewnych linii:
ocr recognize dokument.png --correct-low-only --confidence-threshold 0.7
```

W selektywnej korekcie do API trafiają wyłącznie linie poniżej progu; jeśli
korektor zwróci inną liczbę linii niż wysłano, oryginał zostaje zachowany.

### Konfiguracja (zmienne środowiskowe)

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `OCR_FORCE_LANGUAGE` | (auto) | `pl`/`en` — wymuś język |
| `OCR_CORRECT_TEXT` | `false` | `true`/`false` — włącz korektę przez Fabryka |
| `FABRYKA_API_KEY` | (brak) | klucz API Fabryka (https://fabryka.ai) |
| `FABRYKA_MODEL` | `bielik-11b-v3` | model do korekty |
| `FABRYKA_BASE_URL` | `auto` | `auto` lub pełny URL (np. `https://router.fabryka.ai/v1`) |
| `OCR_RECOGNIZER_EN` | `microsoft/trocr-base-printed` | model TrOCR EN |
| `OCR_RECOGNIZER_PL` | `ocr/trocr-pl-base` | lokalny fine-tune PL |
| `OCR_DEVICE` | `auto` | `auto`/`cpu`/`cuda` |
| `OCR_CONFIDENCE_THRESHOLD` | `0.0` | próg flagowania niskopewnych linii |
| `OCR_CORRECT_LOW_ONLY` | `false` | korekta tylko linii poniżej progu |
| `OCR_RECOGNIZER_BACKEND` | `trocr` | `trocr`, `paddlevl` (PaddleOCR-VL VLM), lub `kraken` |
| `OCR_KRAKEN_MODEL` | `PiotrSty/ehri-dataset::models/polish_nfd_9313.mlmodel` | model Kraken (.mlmodel) |
| `OCR_KRAKEN_BINARIZE` | `false` | `true`/`false` — binarization nlbin przed Kraken |

## Backend PaddleOCR-VL (opcjonalny, SOTA)

Alternatywny backend rozpoznawania: [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
— model VLM 0.9B, 109 języków (w tym polski), #1 na OmniDocBench. Nie wymaga
fine-tuningu ani detektora per język.

```bash
pip install -e ".[vlm]"   # paddlepaddle + paddleocr[doc-parser]

ocr recognize dokument.png --backend paddlevl
python -m training.evaluate --data ./data/pl_lines_val --backend paddlevl
```

Uwagi: backend VLM nie raportuje `confidence` (w JSON: `null`), rozpoznaje
linie po jednej (wolniejsze na CPU). Porównanie head-to-head z TrOCR:
`python -m training.evaluate --data <zbiór> --backend trocr|paddlevl`.

## Backend Kraken (opcjonalny, historyczne dokumenty / maszynopis)

[Kraken](https://kraken.re/) — system OCR/HTR zoptymalizowany pod dokumenty
historyczne i maszynopisy. Używa **trainable baseline segmentation** (sieć
neuronowa wykrywająca linie-bazy) oraz modeli rozpoznawania `.mlmodel` (CTC).

**Kiedy używać:** maszynopisy, dokumenty historyczne, wyblakłe skany — gdzie
OpenCV+TrOCR zawodzi przez słabą segmentację. Dla czystego druku TrOCR
pozostaje lepszy.

**Model domyślny:** `polish_nfd_9313.mlmodel` z EHRI (93,1% accuracy na polskim
maszynopisie).

```bash
pip install -e ".[kraken]"   # kraken>=5.0

ocr recognize dokument.png --backend kraken
python -m training.evaluate --data ./data/pl_lines_val --backend kraken
```

Lub w Pythonie:

```python
from ocr import OcrEngine, OcrConfig

cfg = OcrConfig(
    recognizer_backend="kraken",
    kraken_model="PiotrSty/ehri-dataset::models/polish_nfd_9313.mlmodel",
    device="cuda",
)
with OcrEngine(cfg) as engine:
    result = engine.recognize("maszynopis.tif")
    print(result.text)
```

### Benchmark: Kraken vs OpenCV+TrOCR na polskim maszynopisie EHRI

Pełny benchmark: [`training/kaggle_kraken_benchmark.ipynb`](training/kaggle_kraken_benchmark.ipynb)
(Kaggle GPU T4). Zbiór: 15 polskich stron EHRI (468 linii GT z ALTO XML).

| Backend | Segmentacja | CER | WER | Linie wykryte |
|---|---|---:|---:|---:|
| OpenCV+TrOCR run5 | OpenCV | 82,42% | 94,35% | 179/468 (38%) |
| **Kraken e2e** | **Kraken** | **14,25%** | **48,49%** | **467/468 (100%)** |
| Kraken + GT (ALTO) | GT baselines | 10,67% | 32,93% | 468/468 (100%) |
| Kraken e2e + deskew | Kraken | 14,71% | 48,89% | 467/468 (100%) |
| Kraken + GT + binarization | GT baselines | 20,26% | 65,16% | 468/468 (100%) |

**Wnioski:**

- **Kraken jest 5,8× lepszy** od OpenCV+TrOCR na maszynopisie (CER 14% vs 82%).
- **OpenCV gubi 62% linii** na maszynopisie — segmentacja to główna blokada.
- **Kraken wykrywa 100% linii** — segmentacja baseline działa na maszynopisie.
- **Binarization szkodzi** — model trenowany na grayscale (CER 14% → 20%).
- **Deskew nie pomaga** — strony EHRI są już wyrównane.
- **GT segmentacja poprawia** CER z 14% do 11% — wytrenowanie modelu segmentacji
  na polskich danych mogłoby poprawić e2e.

**Różnica 14% vs 7% EHRI:** możliwe przyczyny to legacy polygon extractor
(model nie trenowany z nową metodą), różnica wersji Kraken, oraz brak
dedykowanego modelu segmentacji dla polskiego maszynopisu.

## Testy

```bash
pytest
```

## Struktura

```
ocr/        # biblioteka (config, preprocess, detector, recognizer, pipeline, lang, cli)
training/   # fine-tuning polskiego TrOCR (GPU)
tests/      # testy jednostkowe (nie wymagają modeli ML)
```

## Ograniczenia / roadmapa

- Fine-tune PL wymaga danych + GPU (patrz `training/`). Alternatywa bez treningu:
  korekta tekstu przez Fabryka API (Bielik) — włącz `--correct`.
- Routing języka: bez `force_language` linie startują modelem EN, a wykryte jako PL
  są re-rozpoznawane modelem PL (drugi przebieg, gdy model istnieje). Gdy modelu PL
  brak — zostaje wynik EN (diakrytyki naprawia korekta Bielik).
- Brak obsługi układów wielokolumnowych / tabel (sortowanie czytania uproszczone).
- Detektor OpenCV (fallback) jest prostszy od CRAFT — dobry do dokumentów/skanów,
  słabszy do tekstu w naturze i złożonych tła. **Na maszynopisach zawodzi**
  (wykrywa 38% linii) — użyj backendu Kraken (`--backend kraken`).
- Alternatywa bez treningu: PaddleOCR-VL + LoRA RysOCR (lepsza polska diakrytyka od ręki).

Pilot rzeczywistych skanów: [wyniki i odtworzenie](docs/PUBLIC_SCAN_PILOT.md).

Zwykły druk i vision API: [wyniki oraz przygotowany test](docs/PRINT_AND_VISION_PILOT.md).
