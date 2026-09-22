# PolOCRBench — protokół ewaluacji podzadań A/B/C

Ewaluacja jest w pełni automatyczna i deterministyczna, bez sędziów LLM.
Skrypty są publiczne i nie wymagają pakietów ML poza `jiwer` (tylko podzadanie
A). Wersje protokołów:

| Protokół | Moduł | Zakres |
| --- | --- | --- |
| `polocrbench-transcription-v1.1` | `training.transcription_eval` | A: CER/WER + struktura Markdown (zamrożony) |
| `polocrbench-table-teds-v1` | `training.table_eval` | B: TEDS / TEDS-struct |
| `polocrbench-kie-v1` | `training.kie_eval` | C: field-level F1 |
| `polocrbench-composite-v1` | `training.composite_score` | wynik zbiorczy |
| `polocrbench-submission-tsv-v1` | `training.submission_tsv` | format zgłoszeń AmuEval |

## Format danych

Każde podzadanie ma własny manifest i własny plik predykcji (JSONL, UTF-8,
jeden obiekt na linię, rozdzielnik LF). Uczestnik może startować w dowolnym
podzbiorze podzadań.

| Podzadanie | Manifest (referencja) | Predykcje |
| --- | --- | --- |
| A | `{id, image, sha256, text}` | `{id, status: ok\|error, text: Markdown, elapsed_seconds?}` |
| B | `{id, image, sha256, html, table_index?}` | `{id, status: ok\|error, html, elapsed_seconds?}` |
| C | `{id, image, sha256, doc_type, fields}` | `{id, status: ok\|error, fields, elapsed_seconds?}` |

Jedna pozycja podzadania B to jedna tabela ze strony (kolejne tabele tej samej
strony to osobne `id`). `status: error` oznacza nieudane wykonanie i musi mieć
pusty payload; brakujący `id` jest dozwolony tylko przy walidacji z
`--allow-missing`, a w obu przypadkach pozostaje w mianowniku z wynikiem 0.
Ewaluatory weryfikują sumy SHA-256 obrazów wskazanych w manifeście.

Dane syntetyczne (`training.generate_documents`) zapisują manifesty w tym samym
formacie. Konwencja podzadania A dla treści tabel w referencjach syntetycznych:
jeden wiersz na linię, komórki rozdzielone pojedynczą spacją, bez znaczników
Markdown/HTML — punktacja CER nie może być zdominowana wyborem znaczników.

## Metryki

### A — transkrypcja (zamrożona)

CER (główna) i WER po normalizacji: NFC, cudzysłowy typograficzne -> proste,
zwinięcie białych znaków; wielkość liter i diakrytyki zachowane. Dodatkowo
przybliżona odległość edycyjna szkieletu Markdown (nagłówki, listy, akapity).
Szczegóły: `docs/POLOCRBENCH_EVALUATOR_2026-09-19.md`.

### B — tabele: TEDS

Przed porównaniem obie strony przechodzą tę samą kanonizację:

- strukturą są tylko `table`, `tr`, `td`, `th`; wrappery `thead`/`tbody`/
  `tfoot` są usuwane, a ich wiersze awansowane (`<table><tr>` równa się
  `<table><tbody><tr>`);
- inne elementy są przezroczyste (ich tekst zostaje w otaczającej komórce),
  `<br>` daje spację, tekst poza komórkami jest ignorowany;
- poza `rowspan`/`colspan` (domyślnie 1) atrybuty są ignorowane;
- tekst komórek: normalizacja jak w A.

$$\mathrm{TEDS} = 1 - \frac{\mathrm{TED}(T_{ref}, T_{hyp})}{\max(|T_{ref}|, |T_{hyp}|)}$$

TED to odległość edycyjna drzew uporządkowanych (Zhang-Shasha), wstawienie i
usunięcie węzła kosztują 1, a zamiana: 0 dla zgodnych etykiet (tag +
rozpiętości), $1 - \mathrm{lev}(a,b)/\max(|a|,|b|)$ dla liści komórkowych
o zgodnych rozpiętościach, 1 w pozostałych przypadkach. Strona bez
rozpoznanych elementów dostaje 0,0, o ile druga strona je ma; obie puste to
1,0. `TEDS-struct` (raportowany pomocniczo) ignoruje treść komórek.

Wynik główny: średnia TEDS po wszystkich tabelach z manifestu; `status: error`
i braki dostają 0 i zostają w mianowniku.

### C — KIE: field-level F1

Typy dokumentów i pola są zamrożone w `training.kie_eval.SCHEMAS`
(`python -m training.kie_eval --dump-schemas` wypisuje schemat JSON):
`faktura` (13 pól), `umowa` (9), `pismo_urzedowe` (6), `formularz` (6).
Przewidywane klucze muszą należeć do schematu typu dokumentu; wartości to
liczby lub napisy. Brak klucza albo pusta wartość = „nie odczytano".

Pole jest poprawne tylko przy dokładnym dopasowaniu po normalizacji:

| Typ | Normalizacja | Przykład |
| --- | --- | --- |
| `date` | do `YYYY-MM-DD`; formaty `13.10.2026`, `2026-10-13`, `13 października 2026 r.`; dzień pierwszy dla dat liczbowych; nieprawidłowa data porównywana dosłownie | `13.10.2026` = `2026-10-13` |
| `money` | do zwykłego ułamka dziesiętnego z kropką, bez separatorów tysięcy i walut; `,-` = `,00`; zera końcowe usuwane | `1 234,56 zł` = `1234.56` |
| `nip` | 10 cyfr (separatory usuwane) | `123-456-78-90` = `1234567890` |
| `pesel` | 11 cyfr | `020708 03654` = `02070803654` |
| `address` | NFC + cudzysłowy + małe litery + usunięta interpunkcja (myślniki zostają) + zwinięte spacje | |
| `text`, `name` | NFC + cudzysłowy + małe litery + zwinięte spacje | `FV/1` = `fv/1` |

Wartość, której nie da się sparsować typowo, jest porównywana dosłownie po
normalizacji tekstowej. Waluta jest osobnym polem, nie częścią `money`.
Reguła separatorów w kwotach: pojedynczy separator po grupach 3-cyfrowych to
separator tysięcy (`12,500` = `12500`), w pozostałych przypadkach separator
dziesiętny (`12,50` = `12.5`), a przy obu separatorach dziesiętny jest ten
prawy.

Liczenie mikro: trafienie = TP; pomyłka = FP + FN; pominięte pole = FN;
zmyślone pole (brak w referencji) = FP; dokument z `status: error` lub brak
zachowuje wszystkie niepuste pola referencji jako FN. Werdykty per pole w
raporcie: `match`, `mismatch`, `missing`, `spurious`, `absent`.

$$F1 = \frac{2TP}{2TP + FP + FN}$$

$F1 = 1{,}0$, gdy nie ma nic do wyciągnięcia i nic nie zgłoszono. Raporty
pomocnicze: `per_type` (per typ pola) i `per_field` (per nazwa pola).

## Wynik zbiorczy

$$\mathrm{composite} = \mathrm{mean}\left(1 - \mathrm{CER}_{micro},\ \mathrm{TEDS}_{mean},\ \mathrm{F1}_{micro}\right)$$

średnia po podzadaniach faktycznie zgłoszonych. $1 - \mathrm{CER}$ może być
ujemne (CER > 1); wynik nie jest przycinany. Raport wejściowy pochodzi
z ewaluatora danego podzadania:

```bash
python -m training.composite_score --report-a a.json --report-b b.json --report-c c.json --output composite.json
```

Gdy rekordy predykcji zawierają `elapsed_seconds`, raport zbiorczy zawiera
czas inferencji na rekord (raport kosztów per strona dla baseline'ów).
Zamrożony raport podzadania A nie przenosi `elapsed_seconds` do rekordów
wynikowych, więc sekcja `timing` obejmuje B i C; czas A jest raportowany
przez runner (`run.json`).

## Format zgłoszeń (AmuEval)

Zgłoszenie do serwisu to `out.tsv` w kolejności `test/in.tsv`, zgodnie z
konwencją AmuEval (PolEval 2025 Śmigiel: osobne wyzwania Test-A/Test-B).

- `in.tsv`: jeden `id` na linię; ta kolejność jest kolejnością zgłoszenia;
- `out.tsv`: jeden payload na linię, UTF-8: A — Markdown, B — HTML tabeli,
  C — zwarty obiekt JSON (`sort_keys`);
- komórka nie może zawierać surowych znaków sterujących: `\\` -> `\\\\`,
  tab -> `\t`, LF -> `\n`, CR -> `\r`; rozpakowanie jest dokładną odwrotnością
  i odrzuca nieznane escape'y;
- pusta komórka = pusty wynik. Kanoniczny JSONL rozróżnia `status: error`,
  ale TSV nie: packuje się je do pustych komórek, a `unpack` zwraca
  `status: ok` z pustym payloadem.

```bash
python -m training.submission_tsv --mode pack --subtask A --in-tsv in.tsv \
    --predictions run.jsonl --out-tsv out.tsv
python -m training.submission_tsv --mode unpack --subtask A --in-tsv in.tsv \
    --out-tsv out.tsv --predictions run.jsonl
```

Walidacja kompletności i formatu (przed wysyłką i przed punktacją):

```bash
python -m training.validate_submission --manifest M.jsonl --predictions run.jsonl \
    --subtask C --meta submission_meta.json
```

## Tracki i deklaracja zgłoszenia

| Track | Trening | Modele | Prompt |
| --- | --- | --- | --- |
| `constrained` | wyłącznie dane organizatora (`organizer:...`) | open-weight, pretrenowane | dowolny |
| `open` | dowolne dane i modele, w tym fine-tuning | dowolne | dowolny |
| `zero-shot` | bez treningu i bez dodatkowych danych | zamknięte i otwarte | wyłącznie zamrożony prompt organizatora |

`submission_meta.json` (sprawdzany format; deklaracje pozostają
oświadczeniem uczestnika, nie są weryfikowane dowodowo):

```json
{
  "schema": "polocrbench-submission-meta-v1",
  "team": "Nazwa zespołu",
  "track": "zero-shot",
  "subtasks": ["A", "B"],
  "models": [{"name": "gpt-5", "open_weight": false}],
  "finetuned": false,
  "training_data": [],
  "prompt_version": "polocrbench-zero-shot-prompt-v1",
  "prompt_sha256": "69f80f1c0bbc2a0f1531babee5fa4a2e8b3f8db50ef8ea3426879469169c909e"
}
```

Reguły: `constrained` wymaga wyłącznie identyfikatorów `organizer:*` oraz
`open_weight: true`; `zero-shot` wymaga `finetuned: false`, pustego
`training_data` oraz wersji i hasha promptu zgodnych z rejestrem
`ZERO_SHOT_PROMPTS` w `training/validate_submission.py`. Prompt tracku
zero-shot: `benchmarks/polocrbench/prompts/zero_shot_prompt_v1.md`
(szablony A/B/C dosłownie, bez dopisków i few-shot).

## Znane ograniczenia

- Schematy KIE i reguły normalizacji są zamrożone dla wersji protokołu; ich
  zmiana wymaga nowej wersji `polocrbench-kie-*` i przeliczenia wyników.
- Normalizacja nie obejmuje lematyzacji (fleksja w nazwach i nagłówkach
  zostaje problemem otwartym dla wersji przyszłych).
- Kanonizacja B ignoruje podział nagłówek/treść (`thead` vs `tbody`) oraz
  atrybuty inne niż `rowspan`/`colspan`.
- Walidator sprawdza format deklaracji tracku, nie prawdziwość deklarowanych
  danych treningowych i modeli.
