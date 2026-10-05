# Recognizer v3: source split screening and six-case follow-up

## CPU Source Audit

The audit checks all 89 pages in the pinned source sample
`PiotrSty/impact-psnc-polish-ocr` at `c7cb156fb95d2880699c33725bbaf1fbc1008fea`:
29 permitted training-source pages, 36 geometry-holdout pages, 15 validation
pages and 9 source-test pages. Every original JPEG SHA-256 and dimension was
verified. The run made **2739 cross-split comparisons** and found **0 overlap
proposals** at its frozen thresholds. A second run rechecked cached bytes and
reproduced the signatures and empty proposal queue.

The protocol uses the existing 64-bit dHash implementation, Hamming distance
at most 4 and at most 10% aspect-ratio difference for image proposals. Text
screening uses case-preserving diplomatic 7-word shingles, at least 12 distinct
shingles in both texts, and Jaccard or containment at least 0.95. NFC and
whitespace normalization only; no long-s conversion or spelling modernization.
Short repeated headings do not become text-leakage findings.

These are **screening heuristics**, not proof that duplicates cannot exist.
The audit covers only this 89-page source sample, not external pretraining
or the separate 36-page IMPACT-v2 evaluation archive. Source transcriptions
are not adjudicated gold. dHash can miss rotations/crops and flag shared layouts.

All 89 digital `edition_id` values and all 89 `document_id` values are distinct.
Their one-per-page granularity does **not** establish bibliographic work or
edition independence. A catalog of 30 collections retains source XML paths and
digital IDs, with bibliographic work/edition fields explicitly unresolved.
`training_freeze_ready=false` remains in the report. No model was trained and
no new OCR accuracy or SOTA claim was made.

## Six Cases, Not Another 64-Line Review

The accepted 58 pairs remain untouched. The follow-up contains only:

- Three previous `proposed` transcriptions, copied verbatim with unchanged scans.
- Three rejected geometries, with NEW rectangle proposals from original regions.

The proposals restore full horizontal extent for the two truncated crops and
narrow the vertical extent for the Szturm line. Original crop coordinates were
located by exact decoded-pixel comparison, not OCR guesses. The new boundaries
are AI proposals and can still contain unwanted adjacent ink; they require human
review and must be rejected if a target glyph is clipped or another line is included.
The source PAGE XML files contain no `TextLine` geometry, so they cannot certify
these rectangles. Pixels inside each rectangle are unchanged; no resize,
whitening, deskew or linguistic cleanup is applied.

New pair IDs, crop hashes, parent IDs, rejected-crop reasons, original weak labels,
latest human text and prior event IDs are retained. Old raw teacher outputs are
explicitly labeled as belonging to the previous crop. No previous verified status
is transferred to a new image, and no simulated QA event is human evidence.

## What The Reviewer Does

Open the local six-case editor. For each entry inspect the SMALL LINE CROP first;
the full region is only context. Keep historical spelling exactly as seen.
If the crop contains the whole intended line, select `Cała jedna linia`, correct
or confirm the transcription, choose `Zweryfikowano wzrokowo` and save.
If it is cut or includes another line, choose `Odrzuć wycinek` and state the reason.
Do not confirm a full-context text against an incomplete small crop.

Export the history after all six decisions and return
`slayer-recognizer-line-review-36da6d5738a5.json`. No Colab or GPU is needed.
There is no need to reopen or repeat the accepted 58 pairs.

## Tooling And QA

- `training.audit_recognizer_source_splits`: pinned metadata, original JPEG
  verification, signatures, cross-split proposals and unresolved work catalog.
- `training.prepare_recognizer_line_remediation`: only pending/rejected cases,
  frozen recrop recipes, separate history and code-free data ZIP.
- Focused CPU regression suite: **199 tests passed** with `PYTHONUTF8=1` on Windows.
- Isolated Playwright/Edge QA: 1366x900 and 390x844, six crops and six contexts
  rendered, no console errors or horizontal mobile overflow, geometry gate,
  export hashes and local reload persistence checked.

Browser plugin unavailable; bundled Playwright with installed Edge was used.
The user's browser storage was not accessed. Only the scalar QA report is
published; simulated decisions are excluded from all datasets and evidence.

## Next Data Gate

Resolve bibliographic work/edition grouping, expand training-source lines outside
the already-reviewed IDs, and separately adjudicate frozen development labels.
Select a recognizer only on document-disjoint development data, then evaluate
the complete page pipeline without tuning on final test. This follow-up does not
replace the need for a much larger clean corpus.
