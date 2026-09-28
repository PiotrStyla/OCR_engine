# SLAYER-OCR 2.0: DATA ENGINE v2

Status: architektura, przypięty pilot trzech teacherów, moduł konsensusu i
zweryfikowany technicznie smoke na dwóch stronach. Nie wykonano treningu RF-DETR.

## Decyzja

Rozwój przechodzi z kolejnych drobnych zmian heurystyk do silnika danych:

`generator -> teacher ensemble -> consensus -> annotation review -> clean dataset -> RF-DETR layout -> recognizer -> hard-example mining`

To jest kierunek eksperymentalny, nie deklaracja SOTA. Ostatni test 15 stron
pokazał identyczny CER/WER dla row-major i column-aware, a tylko 225 z 309
wykrytych ramek nakładało się na anotowane regiony. Uzasadnia to inwestycję w
dane i detekcję layoutu przed dalszym strojeniem kolejności czytania.

## Co pochodzi ze źródeł, a co jest naszą syntezą

- Kolekcja Merve zawiera kolejne artefakty: etykietowane dane, zbiory
  `judged`/`ensemble-agree1` oraz modele RF-DETR trenowane na uzgodnionych
  etykietach. To praktyczny wzorzec weak supervision, ale nie kompletna
  specyfikacja naszego procesu:
  <https://huggingface.co/collections/merve/vision-intern>
- Dolphin jawnie rozdziela analizę dokumentu i kolejności czytania od późniejszego
  parsowania elementów. To podstawa etapu `analyze-then-parse`:
  <https://github.com/bytedance/Dolphin>
- DeepSeek-OCR łączy cechy CLIP i SAM oraz bada kompresję tekstu wizualnego.
  Traktujemy to jako inspirację dla reprezentacji semantyczno-przestrzennej,
  nie jako gotowy teacher lub dowód jakości na polskich drukach historycznych:
  <https://github.com/deepseek-ai/DeepSeek-OCR>
- RF-DETR przyjmuje zbiory COCO JSON i jest sensownym kandydatem na mały model
  layoutu. Jego wyniki na COCO nie dowodzą przewagi na naszych dokumentach:
  <https://github.com/roboflow/rf-detr/blob/develop/docs/index.md>

Połączenie tych elementów w DATA ENGINE v2 jest decyzją projektową SLAYER-OCR,
a nie opisem jednego opublikowanego systemu.

## Kontrakt danych

Każda propozycja teachera jest osobnym rekordem
`slayer-layout-teacher-proposal-v1` i zawiera:

- identyfikator strony, ścieżkę, rozmiar i deklarowany SHA-256 obrazu;
- `teacher.id`, przypiętą rewizję, identyfikator runu i hash promptu;
- etykietę, bbox `xyxy`, score oraz identyfikator każdej detekcji.

Jedna rodzina/model (`teacher.id`) ma jeden głos na stronę. Identyfikator jest
kanonicznym slugiem zapisanym małymi literami i w jednym buildzie wskazuje jedną
rewizję, jeden run oraz jeden hash promptu. Zmiana promptu, temperatury lub runu
nie tworzy niezależnego nauczyciela. Obrazy i pełne surowe odpowiedzi pozostają
prywatnymi dowodami runu. Klasy regionów są wzajemnie wykluczające; relacje
zagnieżdżone wymagają w przyszłości osobnej warstwy ontologii.

## Konsensus i abstencja

`training.build_layout_consensus`:

1. waliduje wszystkie wejścia przed utworzeniem wyników;
2. grupuje boksy tej samej klasy, wymagając IoU z każdym elementem klastra;
3. przyjmuje tylko klaster z kworum różnych `teacher.id`;
4. scala współrzędne medianą;
5. kieruje singletony, niski score i konflikty klas do `review-queue.jsonl`;
6. eksportuje zaakceptowane obiekty do `annotations.coco.json`;
7. zapisuje wejściowe hashe, politykę, ograniczenia i sumy kontrolne.

Przykład:

```bash
python -m training.build_layout_consensus \
  --proposals private/run-a.jsonl private/run-b.jsonl private/run-c.jsonl \
  --config experiments/2026-09-27/slayer-ocr-2-data-engine/config.json \
  --output private/layout-consensus-v1
```

Wyjście jest kandydatem weak-label, nie ground truth. Prywatny panel bboxów można
zbudować z ZIP-a konsensusu i lokalnych obrazów poleceniem:

```bash
python -m training.build_layout_consensus_review \
  --evidence private/slayer-layout-consensus-evidence.zip \
  --images private/images \
  --output private/layout-review/index.html
```

Generator weryfikuje sumy kontrolne ZIP-a i obrazów, nie osadza obrazów ani
referencji w HTML i nie nadpisuje istniejącego wyniku. Review oraz
`training.adjudicate_reviews` pozostają końcową bramką transkrypcji.

## Zamrożony pilot v1

Pilot używa 60 deterministycznie wybranych stron `train` z 22 kolekcji
`PiotrSty/impact-psnc-polish-ocr`. Końcowy test i validation nie uczestniczą w
wyborze. Referencyjna transkrypcja nie jest przekazywana teacherom, a stara
pisownia nie jest modernizowana. ZIP-y dowodowe nie zawierają skanów ani tekstu
referencyjnego.

Trzy niezależne rodziny teacherów uruchamiamy w osobnych, świeżych sesjach:

| Teacher | Rola | Score | Licencja wymagająca zachowania w provenance |
| --- | --- | --- | --- |
| Qwen3-VL 4B Instruct | semantyczne wskazanie regionów i klas | neutralne `0.5`; model nie zwraca kalibrowanej pewności | Apache-2.0 |
| DocLayout-YOLO DocStructBench | wyspecjalizowany detektor layoutu | confidence modelu | wagi Apache-2.0, pakiet inferencyjny AGPL-3.0 |
| Surya Layout fast | drugi niezależny detektor layoutu | confidence modelu albo jawne neutralne `0.5` | AI Pubs OpenRAIL-M |

Źródła techniczne: [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),
[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) oraz
[Surya Layout fast](https://huggingface.co/datalab-to/surya_layout2). Zgodność
dwóch modeli jest słabą etykietą, nie dowodem prawdy. Przed wydaniem datasetu lub
wag studenta obowiązuje osobny przegląd licencji i ręczny audyt próbki.

Konfiguracja: `experiments/2026-09-28/slayer-layout-teacher-pilot-v1/config.json`.
Kod runu zapisuje model/revision, hash promptu, środowisko, wybór stron, błędy,
surową odpowiedź Qwen, propozycje kanoniczne i sumy kontrolne. Konsensus odrzuca
ZIP-y z traversal/symlink, zmienionymi hashami, inną konfiguracją, rewizją,
runem albo zestawem stron.

## Uruchomienie w Colab

1. Otwórz `training/colab_slayer_layout_teacher_pilot_v1.ipynb`, wybierz GPU i
   pozostaw `PAGES = 2` oraz `TEACHER_ID = "qwen3-vl-4b"`. Uruchom wszystko i
   pobierz ZIP.
2. Powtórz w dwóch świeżych runtime'ach dla `doclayout-yolo` oraz
   `surya-layout2`. Nie zmieniaj liczby stron pomiędzy teacherami.
3. Otwórz `training/colab_slayer_layout_consensus_v1.ipynb`, wgraj dokładnie te
   trzy ZIP-y, uruchom wszystko i pobierz ZIP konsensusu.
4. Sprawdź `run.json`, `consensus/report.json` i `review-queue.jsonl`, a następnie
   zbuduj prywatny panel nakładek poleceniem powyżej. Nie zwiększaj `PAGES`,
   dopóki bramka opisana w wyniku smoke nie zostanie zamknięta.

Notebook teacherów potrzebuje GPU; notebook konsensusu działa na CPU. Żaden z
nich nie publikuje artefaktów na GitHub ani Hugging Face.

## Plan etapów i bramki

| Etap | Artefakt | Bramka przejścia |
| --- | --- | --- |
| 0. Ontologia | zamrożone klasy regionów | brak klas mieszających regiony i linie |
| 1. Pilot teacherów | prywatne JSONL + hashe | pełne provenance, >=3 różne rodziny modeli |
| 2. Konsensus | COCO + review queue | audyt próbki zgód i wszystkich konfliktów |
| 3. Review | wersjonowany clean dataset | poprawki człowieka oddzielone od teacherów |
| 4. RF-DETR | checkpoint + run manifest | poprawa mAP/recall na rozłącznych dokumentach |
| 5. OCR A/B | pełnostronicowe predykcje | poprawa CER i WER bez regresji kolekcji |
| 6. Mining | kolejka trudnych stron | nowe dane tylko do następnej wersji train |

Końcowy test PolOCRBench pozostaje zamknięty. Progi konsensusu, klasy, prompty,
modele i checkpointy wybieramy wyłącznie na train/development. Split wykonujemy
po dokumentach i kolekcjach, nie po cropach.

## Hard-example mining

Pierwsza wersja zapisuje strony z brakiem kworum, niskim score lub konfliktem
klas. Następne wersje dodadzą:

- duży rozdźwięk bboxów teacherów;
- poprawkę człowieka po wcześniejszym konsensusie;
- rozbieżność student RF-DETR vs consensus;
- niski recall regionów lub duży pełnostronicowy CER na zweryfikowanym GT;
- rzadką klasę, typ dokumentu, degradację albo historyczny krój pisma.

Mining nigdy nie zmienia istniejącej etykiety w miejscu. Tworzy kolejkę do
następnej wersji danych z pełnym pochodzeniem.

## Najbliższy eksperyment

Dwustronicowy smoke został wykonany. Następny krok to zamrożenie definicji
`heading`, `figure` i poziomu szczegółowości `text_region`, dodanie jawnego powodu
`granularity-conflict` oraz ponowienie smoke na tych samych dwóch stronach.
Dopiero po przejściu tej bramki wykonamy trzy pełne przebiegi po 60 stron,
zmierzymy zgodność klas, rozkład IoU, odsetek abstencji i czas review, a następnie
przygotujemy dane do pierwszego treningu RF-DETR.
