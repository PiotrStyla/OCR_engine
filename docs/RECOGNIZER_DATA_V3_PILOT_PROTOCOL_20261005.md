# DATA ENGINE: recognizer v3 training-source teacher pilot

## Why this step

Historical recognizer v2 improved the original line-level validation CER but
regressed in the 36-page evaluation. Its historical line targets were matched
by line count, not verified visually. More epochs on those labels would not
address that data-quality uncertainty. The completed 15-page validation review
remains validation-only, not training data.

This pilot creates teacher proposals on existing **training-source** crops.
It does not train v3, produce gold, promote a teacher, or establish SOTA.

## Frozen source selection

- Source: `PiotrSty/impact-psnc-polish-ocr` at
  `c7cb156fb95d2880699c33725bbaf1fbc1008fea` (CC-BY-3.0).
- Historical corpus manifest SHA256:
  `dbed37d35d77fe85af634dcdd353f628f0c318b37b02cff08520eb746f090314`.
- Select 64 line crops from 25 pages and 10 training collections, without
  predictions; deterministic round-robin with at most 8 per collection and
  4 per page, prioritizing source-label occurrences of historical glyphs.
- Target occurrences by line: `ſ` 47, `á` 52, `ɇ` 35. Counts overlap and
  describe unreviewed labels, not proven glyph presence in the scan.
- Exclude all collections, page IDs and exact hashes found in the reviewed
  validation manifest, original validation lines, frozen test metadata and
  the 12-collection geometry holdout. The report records 20 forbidden collections.
- Collection/page/exact-hash checks are complete for this pool. Work/edition
  grouping and near-duplicate audits are **not complete**. No claim of a
  comprehensively leakage-free benchmark is made.
- Existing private-use/replacement-character quarantines and count-match
  filtering mean this is a biased, bounded pilot, not the entire corpus.

## Two-family inference

[Run this one Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/b90ae82fb496d88320e8698cef8debd73785bece/training/colab_recognizer_data_v3_pilot.ipynb).
Select GPU, run all cells and return `recognizer-data-v3-teacher-evidence.zip`.
No manual ZIP uploads are required. The notebook checks out published code
`e24545b22b65d42937fd88714ff99c7b81e50f90`, fetches the branch rather than
requesting an unpublished object, and embeds the exact frozen configuration.
Its six code cells compile and the notebook passes nbformat validation.

One Colab downloads a content-addressed public HF ZIP without manual uploads.
Qwen3-VL-4B-Instruct and TrOCR mixed-v3 run sequentially on the same 64 crops.
Only ID, image path/hash/dimensions and empty source regions reach workers.
Source texts are consulted only after inference. No reference text is in
the prompt or worker inputs. Upstream pretraining overlap is not audited.

Qwen uses the existing NF4/SDPA loader and a new line transcription prompt,
with a 1 MP ceiling and 512 generated tokens. TrOCR uses its frozen checkpoint
hash, FP32, four beams and 128 generated tokens. This crop profile still
requires its first remote GPU run; earlier full-page runs do not validate it.
The notebook isolates dependencies from the kernel and does not use ensurepip.

Agreement uses only NFC and whitespace normalization. Both outputs must be
nonempty, successful, end with EOS and not hit a cap. Raw output is retained.
Any missing/error/empty/capped/non-EOS output is abstention, not a vote.
Disagreement stays explicit. All 64 inputs remain in the denominator.
Agreement is only a **review proposal**, never a training label or gold.

## Next Gates Toward A Specialized Model

1. Inspect all crops and their source-region context. Confirm complete line
   boundaries and diplomatic text; reject ambiguous or cut lines. Preserve
   original labels, teacher outputs, edits, reviewer identity and reasons.
2. Expand the clean training-source pool, resolve work/edition IDs and
   near-duplicates, then freeze document-disjoint train/dev splits. Keep the
   15 validation pages and test collections outside training. A 64-line pilot
   is a workflow check, not enough to establish SOTA.
3. Train a v3 historical recognizer with clean labels and replay, recording
   loss implementation, truncations, seeds, weights and source digests.
   Select only on frozen development data; track per-glyph substitution rates,
   overall CER/WER and replay regressions.
4. Evaluate the complete layout/order/recognition pipeline on separately
   adjudicated, held-out full pages and external collections against frozen
   baselines. Include failures and uncertainty; do not tune on final-test results.
5. Mine remaining training-source failures for the next DATA ENGINE cycle.

Code belongs on GitHub; input scans/manifests and run evidence belong on public
HF registries. The evidence ZIP contains predictions, failures, provenance and
checksums, but no weights, scans or code. Input scans are retrievable from the
frozen dataset revision. No GPU execution has been performed locally.

## Published Input And CPU Checks

[Frozen public HF input bundle](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/tree/4735f045cc791df1926940a1fa0cd78c2c82b48c/data/recognizer-data-v3-pilot-v1-20261005).
Archive SHA256:
`aa691ba4496709666842b6a44a29d8af08e816a83fbcf346fddf268fe777acbf`.
All uploaded files and ZIP members were downloaded and checked independently.
Actual staging verified all 64 crop dimensions and reference-free worker inputs.
97 focused CPU tests passed, including split-firewall rejection, unsafe archives,
raw historic spelling, abstentions, resume/error coverage and notebook failure
packaging. These tests do not verify GPU model loading, quality or T4 fit.
