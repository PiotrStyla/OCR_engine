# PolOCRBench — wydanie zbioru (HuggingFace, AmuEval, leaderboard)

Procedura publikacji wersji zbioru. Buduje ją `training/build_release.py`
(wykrywa układy katalogów z `training.generate_documents`, przepina przez bramkę
integralności i składa pakiety AmuEval przez publiczny `training.submission_tsv`).

## Budowa wydania

```bash
python -m training.build_release --version 0.1 \
    --split train=data/polocrbench-train \
    --split testA=data/polocrbench-testA \
    --split testB=data/polocrbench-testB \
    --holdout-field degradation --holdout-split testB \
    --output releases/polocrbench-0.1
```

Każdy `--split name=DIR` wskazuje katalog z `images/`, `manifest-A/B/C.jsonl`
i `generation.json`. Nazwa niesie rolę: `train` (publiczny z GT), `testA`
(publiczne obrazy i id, GT prywatne), `testB` (w całości prywatny). Budowa
odmawia wydania przy naruszeniach integralności; świadoma zgoda właściciela to
`--allow-violations` (decyzja ląduje w `RELEASE.json` jako
`integrity_overridden`). Przy wyjściu:

```
releases/<wersja>/
  hf/train/        images/ + manifest-A/B/C.jsonl + generation.json
  hf/testA/        images/ + in-<podzadanie>.tsv
  private/<split>/ manifesty + generation.json (+ images/ dla testB)
  amueval/<split>/<podzadanie>/{in.tsv, expected.tsv, sample-out.tsv}
  DATASET_CARD.md  LICENSE-DATA.txt  LICENSE-CODE.txt  RELEASE.json
```

`RELEASE.json` trzyma sumy kontrolne każdego pliku, liczebności, rolę podziałów
i raport integralności. `sample-out.tsv` to pusty szablon poprawnego zgłoszenia;
prawdziwe predykcje baseline'ów pakuje się osobno
(`python -m training.submission_tsv --mode pack ...`).

## Publikacja na HuggingFace

1. Zbuduj wydanie i przejrzyj `RELEASE.json` (integralność, liczebności).
2. Wgraj katalog `hf/` jako dataset (`huggingface_hub`):

```python
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='SlayerLab/PolOCRBench', repo_type='dataset',
                      folder_path='releases/polocrbench-0.1/hf')
```

3. `DATASET_CARD.md` to gotowa karta (CC BY 4.0 dla danych, MIT dla skryptów);
   Test B **nie trafia** na HuggingFace przed końcem konkursu.
4. Katalog `private/` pozostaje u organizatora (archiwum wydania + źródło
   `expected.tsv` dla AmuEval).

## Pakiet AmuEval

`amueval/<split>/<podzadanie>/` zawiera dokładnie format serwisu
(poleval.amueval.pl): `in.tsv` (id w kolejności zgłoszeń), `expected.tsv`
(ukryte złoto, spakowane tymi samymi regułami escapowania komórek co publiczny
`training.submission_tsv`) i `sample-out.tsv` (szablon). Dla wyzwań Test-A/Test-B
wgrywa się `in.tsv` + `expected.tsv` do wyzwań organizatora, a uczestnicy
przesyłają `out.tsv` w kolejności `in.tsv`.

Checklista przed wydaniem:

1. bramka integralności w `build_release` bez naruszeń (albo udokumentowane
   `--allow-violations` z decyzją właściciela);
2. holdout Test B potwierdzony (`--holdout-field`, np. `degradation`/`doc_type`);
3. przegląd flag referencji (U+FFFD, znaki prywatne) przez
   `training.build_annotation_review` + `tools/annotation-review` — zamknięty
   dla wersji testowych;
4. wyniki baseline'ów w `docs/POLLOCR_BASELINES_BC.md` zaktualizowane o pomiar
   na wersji zbioru, którą się wydaje.

## Leaderboard (codesota.com/ocr)

`tools/leaderboard/index.html` to samodzielny stub (HTML bez JS): zasady,
tracki, tabela bieżących pomiarów i instrukcja zgłoszenia. Wdrożenie: wgrać plik
pod `codesota.com/ocr` (np. przez istniejący hosting strony); po uruchomieniu
zgłoszeń AmuEval stub rozbudować o rankingi Test A/Test B ładowane z wyników
wyzwań.

## Licencje

Dane: **CC BY 4.0** (`LICENSE-DATA.txt`). Skrypty ewaluacyjne i tooling: **MIT**
(`LICENSE-CODE.txt`, jak `LICENSE` w repo).
