# Recognizer v3: imported human line review

## Scope and Result

The original human export contains 65 events for all 64 training-source lines.
Its SHA-256 is `2dca00305fd419cdd2cf9bf0760d3c007b364374ded2065d384950d41d66e8d2`.
The frozen review manifest is
`ae9a4dcb667b753bb6592a3ee86a1cd54faf5a2839348e4027d44dc4a7fbe0c0`.

- 57 original line crops are verified-text, complete-line training candidates.
- 3 proposed transcriptions remain pending, even though their geometry is complete.
- 3 verified transcriptions have rejected crops and are excluded from training.
- 1 original crop has a geometry conflict and remains excluded.
- 1 NEW pair replaces that conflicted crop using the full single-line source region,
  following the user's explicit confirmation of review scope in this chat.
- **58 single-human-reviewed training candidates**, from 24 pages and 10 collections.
- No independent gold labels, no new model training, no SOTA claim.

The four original-to-final text edits are retained. Three edits belong to pending
proposals and therefore do not enter the training manifest. Notes are preserved as
notes, not interpreted as replacement text. All raw teacher outputs remain separate.
Historical spelling and glyphs, including long s, accented a and e with stroke,
remain verbatim. No linguistic modernization or character substitution was applied.

## Context Replacement

`NA1_FT__433925__r011__line000` contains only a word fragment, whereas its full
context region contains the single line the user verified. The original export
marked it verified and complete-line; that decision and its original crop are
preserved. The user's subsequent answer confirmed that review concerned the full
context. The replacement has ID
`NA1_FT__433925__r011__line000__context-v1`, a new PNG hash, the original JPEG
context hash, the review event ID and a separate scope-confirmation record.
JPEG decoded pixels are converted to PNG without resize, crop, deskew or any
pixel modification. The importer verifies pixel equality. No old image or event
is overwritten, and the old pair is never selected for training.

## Validation and Files

`training.import_recognizer_line_review` checks source checksums, frozen manifest,
event identity, before/after history, strictly increasing per-line timestamps,
one reviewer, both text and geometry gates, and all pinned train/validation/test
region metadata. Forbidden collections, heldout page IDs and exact image hashes
are rejected. Empty text, replacement characters, private-use characters and
multi-line labels are quarantined. Proposed and rejected decisions are not upgraded.

The code-free HF bundle includes `training-candidates.jsonl`, `reviewed-lines.jsonl`,
pending/rejected/conflict manifests, unchanged original review and manifest,
scope confirmation, teacher diagnostics, source contexts, images and checksums.
Code and tests live on GitHub; datasets and experiment evidence live on public HF.
The focused CPU suite passes **174 tests**, including 32 importer tests. Run Python
with UTF-8 enabled on Windows (`PYTHONUTF8=1`); older full-page test fixtures use
locale-default reads and otherwise corrupt historical characters. A fresh import
with the final importer reproduces the candidate manifest and report exactly.

## Public Artifacts

- [Complete code-free dataset ZIP](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/76dad0d2f727f6b927030e8ec62c5861ef00fe9c/data/recognizer-line-reviewed-candidates-v1-20261005/recognizer-line-reviewed-candidates-v1-20261005.zip).
- [Dataset report and publication provenance](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/tree/76dad0d2f727f6b927030e8ec62c5861ef00fe9c/data/recognizer-line-reviewed-candidates-v1-20261005).
- [Original human decisions, scope confirmation and import evidence](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/1ec254d972b0d510420daff00ce2f6e34e43ca8f/experiments/2026-10-05/recognizer-line-review-result-v1).
- [Tested importer code](https://github.com/PiotrStyla/OCR_engine/blob/682d01e/training/import_recognizer_line_review.py).

All published files were downloaded at these immutable revisions and checked
against local SHA-256 digests. Every member of the downloaded dataset ZIP was
also checked. The evidence repository contains no scans or weights.

## Next Gate

This is a small clean-data pilot, not a sufficient corpus for a SOTA recognizer.
Expand verified training lines, repair the three rejected geometries, and resolve
the three proposed texts without repeating all 64 reviews. Complete work/edition
and near-duplicate split audits before a larger remote-GPU training run. Preserve
the existing heldout validation and final-test boundaries. Compare the resulting
recognizer against pinned baselines on adjudicated document-disjoint data and
full pages; training loss or same-set accuracy cannot establish SOTA.

Reviewer identity is self-reported, not independently authenticated. The current
firewall does not prove absence of near-duplicates or shared editions.
