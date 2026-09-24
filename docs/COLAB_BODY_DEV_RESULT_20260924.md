# Colab body-text diagnostic verification

Input result: user-provided `body-dev-diagnostic-20260924T055437275499Z.zip`.
Comparison: Kaggle `body-dev-diagnostic-20260921T153113386749Z.zip`.

## Verification

- All five payload checksums match; no unlisted payload files.
- Input manifest exactly matches the earlier Kaggle run.
- Reported runner hash matches the generated Colab notebook metadata.
- Both models: all 63 complete prediction records exactly match Kaggle, including
  text, token IDs, status and termination flags. No flagged truncations.
- All 63-line and 56-line metrics recomputed locally and match the report.
- All 126 predictions are nonempty, with no recorded execution errors.

## Metrics

| Model | All 63 CER | All 63 WER | Lowercase CER | CER on 56 without needs-review |
|---|---:|---:|---:|---:|
| Microsoft base-printed | 85.30% | 100.00% | 39.61% | 85.31% |
| PiotrSty mixed-v3 | 31.64% | 79.47% | 31.10% | 31.28% |

Both runs used Tesla T4, Transformers 4.57.6, Pillow 11.3.0, jiwer 4.0.0 and
huggingface_hub 0.36.0. Colab used Torch 2.11.0+cu128; Kaggle used 2.10.0+cu128.
The observed equality applies to this sample and these runs, not all environments.

This verifies the Colab migration, not an OCR quality improvement. The notebook
ran the original 63 draft crops, NOT a new geometry-corrected dataset. Historical
spelling remains preserved in references and primary scoring. The manual one-line
A/B improvement is separate evidence; it was not applied to these 63 inputs.

Next: continue development-region geometry work instead of repeating unchanged
recognition runs. Draft labels, seven uncertain lines, omitted region-alignment
failures and unaudited training overlap retain the previously documented limits.

## Colab A/B reproduction

Second user archive: `body-dev-diagnostic-20260924T060057708854Z.zip`.
All five payload hashes verified. Its two-record input manifest exactly matches
the Kaggle A/B run `body-dev-diagnostic-20260921T154245122854Z.zip`, and its runner
hash matches the Colab A/B notebook. All four complete prediction records are
identical to Kaggle; per-variant metrics were recomputed and matched exactly.

| Model | A original CER | B manual geometry CER |
|---|---:|---:|
| Microsoft base-printed | 172.22% | 77.78% |
| PiotrSty mixed-v3 | 122.22% | 5.56% |

Mixed-v3 B still emits `W świetne bławaty.` against the historical reference
`W świetne błáwaty.`. The missing acute is retained as a recognition error; the
reference is not modernized. No errors, empty outputs or flagged truncations.
This reproduces the single-line geometry effect on Colab, not an improvement
of the automatic segmenter or of the complete 63-line sample.
