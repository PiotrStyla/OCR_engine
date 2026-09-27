# SLAYER-OCR 2.0: DATA ENGINE v2

Status: architektura i lokalny moduł konsensusu. Nie wykonano jeszcze inferencji
teacherów ani treningu RF-DETR.

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

Wyjście jest kandydatem weak-label, nie ground truth. Spory przechodzą do
istniejącego panelu offline po przygotowaniu widoku bboxów; obecny panel tekstowy
i `training.adjudicate_reviews` pozostają końcową bramką transkrypcji.

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

Na małej, prywatnej próbce development uruchomić co najmniej trzy różne rodziny
teacherów. Zmierzyć zgodność klas, rozkład IoU, odsetek abstencji i czas review.
Dopiero po ręcznym audycie zamrozić rewizje teacherów, prompty oraz progi i
wygenerować większy zbiór do pierwszego treningu RF-DETR.
