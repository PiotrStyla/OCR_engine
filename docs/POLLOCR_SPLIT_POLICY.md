# PolOCRBench — podziały zbioru, integralność i Test B

Zamrożona polityka podziałów (§4 zadania) plus bramka integralności, którą
należy uruchamiać przed każdym wydaniem zbioru.

## Role podziałów

| Podział | Rola | Widoczność |
| --- | --- | --- |
| `train` (~2 000 stron, w tym ~1 200 syntetycznych) | trening w tracku `constrained` | publiczny z GT |
| `test A` (~500 stron) | development, ranking pomocniczy | publiczny z GT po konkursie |
| `test B` (~500 stron) | **ranking główny**, ocena generalizacji | ukryty do końca konkursu |

Test B celowo zawiera **typy dokumentów i degradacje nieobecne w train/test A**:
druki frakturowe/szwabachą, pismo ręczne, zdjęcia telefonu (perspektywa, cień,
rozmycie), gęste tabele mieszane PL/EN i wzory. W zbiorze syntetycznym holdout
realizuje się przez `training.generate_documents --types/--degradations`
(np. train: `clean,scan`; test B: `photo,compress`); w realiach przez dobór
materiału z osiami z poniższej listy.

## Bramka integralności

```bash
python -m training.check_split_integrity \
    --split train=data/train/manifest-A.jsonl \
    --split testA=data/testA/manifest-A.jsonl \
    --split testB=data/testB/manifest-A.jsonl \
    --holdout-field degradation --holdout-split testB \
    --output validation/split-integrity.json
```

Sprawdzenia (deterministyczne, exit 1 przy naruszeniach):

- **duplikaty dokładne**: SHA-256 obrazu i SHA-256 tekstu referencyjnego między
  podziałami;
- **duplikaty bliskie**: dHash obrazów (odległość Hamminga ≤ 6 domyślnie) oraz
  shingle słowne 3-gram (Jaccard lub **containment** ≥ 0,9) — containment łapie
  region treningowy skopiowany ze strony testowej, którego Jaccard jest mały;
- **holdout Test B**: wartości pola (`degradation`, `doc_type`, `subset`) w
  teście B muszą być rozłączne z pozostałymi podziałami; wartości czytane
  z wierszy manifestu albo z `generation.json` obok manifestu.

Pola bez wartości nie wywalają bramki (raportowane jako `rows_without_value`).
Obrazy niedostępne lokalnie są pomijane przy dHash i liczone w
`images_missing` (teksty i sumy kontrolne sprawdzane są zawsze).

## Wynik kontroli zamrożonych manifestów (2026-09-22)

`history_train_pool.jsonl` (2531 regionów) vs `history_testA_manifest.jsonl`
(36 stron), świeży checkout bez bajtów obrazów (2687 `images_missing` —
dHash do ponowienia po przywróceniu obrazów):

| Typ | Wynik |
| --- | --- |
| duplikaty dokładne (obraz/tekst) | 0 |
| duplikaty bliskie (dHash) | nie do sprawdzenia (brak obrazów) |
| **bliskie wg tekstu** | **2 pary, containment 1,0** |

Pary to regiony `NA1_FT__433090__r19` („Delicye Włoſkiey Ziemi.") i
`NA1_FT__433090__r29` („Gladius contra Turcas.") w całości zawarte w stronie
testowej `NA2_FT__433928` — dwuwyrazowe motta paratekstowe drukowane w obu
tomach „Nowych Aten". Wpływ na punktację strony pomijalny (≤ 4 słowa na stronę),
ale to realny nakład tekstu train↔testA: dokładnie hipoteza z
`benchmarks/polocrbench/README.md`, której wcześniejsze kontrole nie wykluczały.
Zamrożone manifesty pozostają bez zmian; decyzja o wyłączeniu tych dwóch
regionów z przyszłej puli treningowej (v2) należy do właściciela zbioru i musi
być odnotowana w historii wydania.

## Weryfikacja Test B przed wydaniem

1. bramka integralności na trzech podziałach (zero naruszeń);
2. kontrola holdout: osie `doc_type`/`degradation` rozłączne (powyższy przykład
   syntetyczny kończy się `violations: 0`, exit 0);
3. przegląd flag referencji: kolejka U+FFFD i znaków prywatnych Unicode budowana
   przez `python -m training.build_annotation_review --manifest M.jsonl --output DIR`
   (panel `tools/annotation-review`); dla zamrożonego historycznego testu
   zgłoszono 86×U+FFFD na 23 stronach i 624 znaki prywatne na 33 stronach —
   przegląd wymaga przywróconych obrazów i pracy recenzentów.

Znane ograniczenia: dHash nie rozstrzyga podobnych layoutów syntetycznych przy
progu 6 (próg jest konfigurowalny); tekst nie łapie skanów bez wspólnej
transkrypcji (te obsłuży dHash po przywróceniu obrazów); bramka nie dowodzi
braku obecności dokumentów w pretreningu modeli uczestników.
