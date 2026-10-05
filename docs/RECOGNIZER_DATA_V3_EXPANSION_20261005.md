# Recognizer DATA ENGINE: next 64 new lines

## Scope

The first pilot is complete: 64 single-human-reviewed training candidates.
This step expands the data pool, **not model training or a SOTA benchmark**.
It selects a different 64 line crops from 22 training-source pages and nine
collections. All 64 previously reviewed original line identities are excluded,
including roots whose active images changed through recropping/context replacement.
Previously reviewed crop hashes are also excluded and rechecked during staging.
Training pages may overlap the first training pilot; these are new lines,
not independent validation documents.

The existing corpus has 258 training-source lines. After excluding 64 reviewed
roots, 194 remain. Of those, 137 unreviewed labels contain at least one targeted
historical character and are eligible for this bounded *proposal-generation*
selection. The new batch uses the existing deterministic round-robin ranking,
at most eight lines per collection and four per page. Selection does not use
model predictions. Its unverified labels contain `ſ` in 42 lines, `á` in 47
and `ɇ` in 13; these overlapping counts do not certify glyph presence in images.
The glyph-prioritized, previously filtered corpus is not representative of all
Polish printing or enough to establish SOTA.

Source: IMPACT/PSNC via `PiotrSty/impact-psnc-polish-ocr`, CC-BY-3.0, revision
`c7cb156fb95d2880699c33725bbaf1fbc1008fea`.
The prior reviewed pool is bound to manifest SHA-256
`3b4184f45be39756e5e5f2068dd363c9a3c7a535502e953525690b62e5c85b77`.
The NEW candidate manifest SHA-256 is
`1534aa2f40eea1d9da982ff06c78fd2cccd708d0433b7ea8f9ee5a6ce251687a`.

## One Colab, No Uploads

Use the new expansion notebook, select a GPU and run all cells. It automatically
downloads the complete frozen crop bundle from public HF, checks its checksums
and checks out a published full Git commit. The previous 64-line notebook is an
archive and should not be rerun for this batch.

Qwen3-VL-4B and TrOCR mixed-v3 run sequentially using the same frozen models,
dependency versions and generation settings as the completed pilot. The changed
input batch has not yet been run on GPU; previous pilot execution does not prove
its success or quality. No GPU inference is performed on the local laptop.
No weights or manual JSON/ZIP uploads are required.

Workers receive only IDs, crop images, hashes/dimensions and empty source regions.
Reference text is not passed to model prompts. Raw output and errors are retained.
Agreement remains a proposal; all texts and complete-line geometry need human
review. Missing/error/empty/capped/non-EOS results abstain, not vote.
Historical spelling is never modernized; agreement uses NFC/whitespace only.
Every candidate stays `eligible_for_training=false` and is not gold.

Download and return **`recognizer-data-v3-expansion-v1-evidence.zip`** from the
last cell. This separate filename distinguishes it from the first pilot.
It contains provenance, predictions, failures and checksums, not scans/weights/code;
the images are reproducible from the pinned input package.
The bootstrap uses the existing isolated-environment helper without ensurepip;
failure logs and partial results can still be downloaded.

## Remaining Freeze Gates

The existing collection/page/exact-hash firewall keeps validation, test and the
geometry holdout outside training. The 89-page near-duplicate screening remains
a bounded heuristic, not proof of bibliographic independence or absence of
external pretraining overlap. Original PAGE XML has OCR regions, but does not
provide verified bibliographic work/edition identities.

Bibliographic identity resolution remains a separate required gate: do not fill
work IDs from one-per-page digital edition IDs. Expand/review clean training data
and independently adjudicate frozen document-disjoint dev labels before selecting
a recognizer and evaluating the full-page pipeline on untouched final test.
The new 64 inputs do not increase the accepted pool until review is imported.

## Reproduce Selection

```powershell
python -m training.prepare_recognizer_data_pilot `
  --corpus data/historical-lines-v1 `
  --train-regions data/impact-historical-lines-source/regions/train/metadata.jsonl `
  --test-regions data/impact-historical-lines-source/regions/test/metadata.jsonl `
  --validation data/recognizer-source-split-audit-final-20261005/validation-source-metadata.jsonl `
  --holdout experiments/2026-09-24/geometry-holdout-v1/manifest.json `
  --reviewed-pool data/recognizer-reviewed-pool-v1-20261005 `
  --count 64 --output data/recognizer-data-v3-expansion-input-v1-20261005
```

Use a fresh output directory. The validation input is the pinned complete
15-page source-validation metadata, not a subset or a transcription-review
status filter. The report records its digest, all forbidden collections and
the prior-pool exclusion binding. The original 64 reviewed pairs remain unchanged.
