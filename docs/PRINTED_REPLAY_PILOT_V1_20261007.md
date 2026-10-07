# Real printed replay: source and exact-anchor pilot

V2 did not pass the protected-domain gate. Keep the unchanged baseline and do
not repeat that recipe. This pilot builds a **new candidate data pool**, not
another trained model and not a SOTA result.

## Sources And Boundaries

Twelve validated Wikisource pages were acquired on 2026-10-07:

| Work family | Edition | Pages | Purpose |
| --- | --- | ---: | --- |
| Zeromski, Dzieje grzechu | 1928 | 8 | replay candidates |
| Prus, Lalka | 1890 | 4 | separate replay probe |

Sources: Wikimedia Commons/CBN Polona physical-book scans, public-domain images;
Wikisource contributor transcriptions, CC BY-SA 4.0. Keep `NOTICE.md`, original
revision links, source snapshots, scan hash pins and attribution with derivatives.
All twelve acquired pages had source `pagequality=4`; this is the source's review,
**not a new review by the project or the user**. No spelling modernization.

The two works are different from the ordinary development works
`krolicki-odezwa-do-matek` and `torunski-elementarz-1910`, and from the protected
Nowe Ateny and Wyprawa work families. No existing development lines are used for
replay. Other editions of these new books must remain in the same work family.
The 75 existing ordinary development lines are historical printed material,
not representative modern administrative documents.

PolEval's public post-correction corpus was investigated but its numeric IDs
are not a verified image/work mapping. This pilot goes directly to Wikisource;
it does not reconstruct or claim PolEval train/test independence.

## CPU Notebook

`training/colab_printed_replay_pilot_v1.ipynb` downloads the complete pinned input.
Choose a normal CPU runtime and Run All. No upload or GPU is required. It:

1. Verifies the input ZIP and every payload checksum.
2. Downloads two original DjVu books and checks pinned original SHA1 hashes.
3. Decodes selected pages at native resolution with DjVuLibre; preview JPEGs
   are never used for line extraction.
4. Runs Tesseract Polish single-column segmentation (`--psm 6`) with TSV boxes.
5. Keeps only unique, word-boundary, NFC/whitespace-only exact source anchors,
   at least 20 characters / 3 words, with every word confidence >=90.
6. Requires monotonic source order, nonoverlapping line boxes, line-like geometry
   and at least 40% page anchor coverage. No fuzzy substitutions, case folding,
   hyphen repair or long-s normalization.
7. Downloads `printed-replay-pilot-v1-evidence.zip` with crops, labels, provenance,
   exclusions, package checksums and the actual runtime versions.

## Interpretation

Exact matches select easier lines and can still inherit source transcription
errors or partial/incorrect crop boundaries. A source paragraph is not an
independently certified line annotation. **All candidates remain
`eligible_for_training=false` and `eligible_for_evaluation=false`.**
After the returned ZIP is audited, a separate protocol can authorize a bounded
replay experiment; the existing baseline and protected-domain selection gates
remain unchanged. Do not use the probe for training. This pilot covers only two
books, not newspapers, forms, tables, handwriting or SOTA full-page evaluation.
Upstream model pretraining exposure to these works is unknown.

## Validation

The source acquisition completed: 12/12 pages, 8 candidate / 4 probe, no rejected
sources. Local CPU unit tests cover glyph preservation, exact/ambiguous anchors,
confidence, geometry, ordering, coverage, HTML extraction and checksums.
The DjVuLibre/Tesseract mining stage requires the Colab/Linux runtime and was
not executed on this Windows laptop. No number of resulting lines is promised.
