# Recognizer v3: completed 64-pair reviewed pilot

## Result

The six-case follow-up export `slayer-recognizer-line-review-36da6d5738a5.json`
contains six `verified` / `complete-line` decisions. Their exact texts were
confirmed, not automatically modernized or replaced by model output.
The new pool contains **64 active single-human-reviewed training candidates**:

- 57 previously accepted original line crops.
- One full-context replacement confirmed separately by the user.
- Three unchanged images with their proposed texts now human-confirmed.
- Three new rectangle crops now human-confirmed for geometry and text.

These represent 64 distinct original line identities, 25 pages and 10
training-source collections. No validation, geometry holdout or final-test
collection is admitted. Historical spelling, spacing and glyphs such as
`ſ`, `á`, `ɇ` and `ʒ` remain verbatim; normalization applied: **none**.

The previous rejected or unresolved records are not retroactively edited.
The package retains 70 historical ledger rows, seven inactive historical rows,
68 unique crop PNGs and 35 original region JPEGs. Both raw JSON exports remain
byte-for-byte, as do the previous import reports, weak references, raw teacher
diagnostics, crop-rejection reasons and separate context-scope confirmation.
The old fragment labeled `Na nieſkończoną BOGU Chwałɇ.` remains inactive;
only its explicitly confirmed full-context replacement is active.

## Verification

`training.merge_recognizer_reviewed_pool` replays both imports from raw human
events using frozen train/validation/test region metadata. It rejects changed
checksums, non-reproducible imported rows, unknown or already-active parents,
incorrect parent review bindings and duplicate active roots, IDs or crop hashes.
The new rectangle PNGs reproduce the exact decoded source-region pixels at
their recorded coordinates, without resizing, whitening or cleanup.
This verifies the transformation, not pixel-perfect margins or independent
geometry adjudication.

The focused regression suite passed **215 CPU tests**, including 16 new merger
tests. A second fresh merge must reproduce the active manifest and pool report
before publication. Public downloads are checked at immutable revisions,
including every ZIP member, not just the archive filename.

Active manifest SHA-256:
`3b4184f45be39756e5e5f2068dd363c9a3c7a535502e953525690b62e5c85b77`.
Original follow-up export SHA-256:
`5944aae63765b0fd7484c02be21435004501fc0793261e88a79bf02467621612`.

## Package Layout

- `manifest.jsonl` / `training-candidates.jsonl`: identical active 64-pair pool.
- `images/{sha256}.png` / `contexts/{sha256}.jpg`: immutable assets.
- `review-history.jsonl`: historical rows with portable content-addressed paths.
- `inactive-history.jsonl`: old unresolved/rejected/conflicting records retained.
- `history/base/` / `history/followup/`: original import metadata and review JSONs.
- `source-audit/`: audit summary, settings, unresolved bibliographic catalog
  and original audit checksums; the separate audit's 89 scans are not duplicated.
- `pool-report.json` / `checksums.json`: result and complete package digests.

The raw manifests/checksum maps in `history/` describe their original imports;
their old asset paths are preserved for provenance, not rewritten. Portable
asset paths are in the merged manifest and review ledger. The top-level
`checksums.json` is the integrity map for this complete ZIP.
No model weights or application code are included. Source scans retain
CC-BY-3.0 attribution to IMPACT/PSNC via `PiotrSty/impact-psnc-polish-ocr`,
revision `c7cb156fb95d2880699c33725bbaf1fbc1008fea`.

## Reproduction

After reproducing/extracting the two input import packages and pinned region
metadata, use a NEW output directory:

```powershell
python -m training.merge_recognizer_reviewed_pool `
  --base-directory data/recognizer-line-reviewed-candidates-v1-20261005 `
  --followup-directory data/recognizer-line-followup-reviewed-v1-20261005 `
  --config experiments/2026-10-05/recognizer-data-v3-pilot-v1/config.json `
  --metadata-directory data/impact-historical-lines-source/regions `
  --audit-directory data/recognizer-source-split-audit-final-20261005 `
  --output data/recognizer-reviewed-pool-v1-20261005
```

## Next Gate Toward SOTA

No further review of these six cases and no repeat Colab run is required now.
The pilot is complete, but **64 pairs are not a sufficient SOTA training corpus**.
`eligible_for_training=true` denotes a reviewed candidate, not a final freeze:
`training_freeze_ready=false`, `gold_labels_created=0`,
`model_training_performed=false`, `eligible_for_evaluation=false`, `sota_claim=false`.

Next: resolve bibliographic work/edition grouping, expand the clean training
pool without recycling held-out documents, and independently adjudicate/freeze
development labels. Only then compare recognizer candidates on document-disjoint
development data and evaluate the full-page pipeline on untouched final test.
The prior 89-page audit found no proposals at its frozen thresholds; it does
not prove bibliographic independence, absence of near-duplicates or lack of
external pretraining contamination.

## Public Artifacts

The publication receipt records immutable dataset/evidence revisions and
download verification. Code is on GitHub; the complete scan/evidence package
is on Hugging Face under `data/recognizer-reviewed-pool-v1-20261005` and
`experiments/2026-10-05/recognizer-reviewed-pool-v1`.
