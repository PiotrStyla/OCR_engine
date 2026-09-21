# Kraken Kaggle evidence review: 2026-09-21

Input: user-provided `polocrbench-kraken-evidence.zip`.
Code revision: `8c5cfd1a9be1f2eac6284ee77500dbdd748055ad`.
Configuration: Kraken 7.1.1, EHRI-finetuned recognizer and segmenter,
Tesla T4, Torch 2.10.0+cu128, Pillow 11.3.0.

## Verification

- All 13 checksums listed in the archive match.
- Completed 36/36 pages, zero execution errors, zero empty predictions.
- Full run: 06:19:50 to 06:32:51 UTC, approximately 13 minutes.
- Recomputed CER/WER locally from archived predictions and references using
  the repository's transcription-v1.1 normalization and jiwer: exact match.
- Reference texts and decoded image pixel hashes match the published CPU
  Tesseract baseline on all pages. PNG byte hashes differ across environments;
  this does not imply different decoded images.

## Results

Lower is better. These are micro-aggregated edit rates, not accuracy scores.

| Configuration | CER | WER |
| --- | ---: | ---: |
| Tesseract.js pol+eng, published PNG baseline | 34.94% | 81.23% |
| Kraken EHRI recognizer + EHRI segmenter, Kaggle | 79.56% | 107.61% |

Kraken CER is worse by 44.61 percentage points. WER can exceed 100% because
insertions count in the numerator. Successful execution does not imply
successful recognition.

## Limits and Next Experiment

The configuration is not suitable for this historical-print test as measured.
Domain mismatch, segmentation, line crops/order and recognition must be
diagnosed separately; this run does not isolate their contributions.
Do not generalize the result to every Kraken model.

References have previously identified Unicode issues. In particular,
`NA2_FT__434735` has only 4 normalized reference characters against 1598
prediction characters; review the scan and annotation coverage before claiming
the model alone explains that page's score. Do not silently drop this page.

Next: inspect segmentation overlays and line crops on representative development
pages, compare a default segmenter while retaining the same recognizer, and
evaluate an appropriate printed-text recognizer under the same protocol.
Do not tune on the frozen test or start another expensive training run yet.

Reproducibility gap: segmentation and recognition configurations were recorded
as object representations containing memory addresses rather than serialized
parameters. Model hashes and package versions are present, but explicit
configuration serialization should be fixed for future runs.
