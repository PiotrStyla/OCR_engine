# Line-band experiment: completed development diagnostic

Date: 2026-09-24. Model weights were not changed or retrained.
Result archive: `body-dev-diagnostic-20260924T104545380613Z.zip`.
Previous rectangular run: `body-dev-diagnostic-20260924T102032899332Z.zip`.
Machine-readable metrics and source hashes: [summary JSON](../experiments/2026-09-24/body-geometry-bands/result-summary.json).

## Verified result

Both pinned models completed 126 predictions: 63 original crops and 63
candidate/fallback crops. No inference errors or empty outputs. Five payload
checksums verified; metrics recomputed from predictions and draft references.
The mixed-v3 original-arm prediction records are identical to the previous run.

| Model | Original CER | Rectangle CER | Line-band CER |
| --- | ---: | ---: | ---: |
| PiotrSty/trocr-pl-mixed-v3 | 31.6441% | 29.4391% | 29.4004% |
| microsoft/trocr-base-printed | 85.2998% | 84.4874% | 84.4101% |

All columns use the same 63 IDs and draft reference strings. Primary CER
preserves case and historical spelling (including acute accents and long s);
only NFC and whitespace normalization is applied.

Mixed-v3 WER remains 77.5656% in both candidate variants, versus 79.4749%
on original crops. The new variant gains only **one character edit** over the
rectangle variant (0.0387 percentage points CER), and 58 over original crops.
Against rectangles, nine lines improve, ten regress and 44 tie in edit count.
This is not evidence of a meaningful general improvement.

The new variant changes 33 crops, not the previous 31: its changed-only CER
must not be directly compared to the old changed-only score. Within this run,
the changed-33 subset improves from 45.0947% to 40.1033% CER; 20 improve, nine
regress and four tie. All 30 fallback predictions are identical across arms.
The 56-line subset excluding needs-review references worsens slightly versus
rectangles: 29.0295% to 29.1161% CER.

For Microsoft, the lowercase-only diagnostic changes from 37.2147% (rectangle)
to 37.2921% (band), a slight regression despite the primary CER improvement.
Do not infer a clean quality win from its case-sensitive score alone.

## Specific observations

- `Wiesc_FT__436884__r001__line002`: terminal colon appears in the new OCR,
  but total character edits remain 18, versus 15 on the original crop.
- `Choragiew_FT__436804__r002__line000`: 22 to 17 edits after masking, versus
  19 on the original crop.
- `Slawna_wiktoria_FT__437103__r003__line003`: still one error, losing the
  historical acute accent. The reference is not modernized to hide that error.

## Scope and decision

These are draft user-corrected references from six pages, not approved gold
labels. Repeated review was skipped; some references remain uncertain or have
private-use glyphs. Geometry changes were developed after examining this same
sample. This is not independent full-page evaluation or proof of SOTA, and it
does not establish upstream training-set separation.

Do not promote band masking to the default or select per-line variants using
these scores. Stop iterative tuning on these 63 lines. The next quality check
should use separately selected development-validation pages, with a frozen
candidate and unchanged evaluation policy; keep the final frozen test untouched.

## Reproduction

- [Colab notebook](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/7e90d13/training/colab_geometry_bands.ipynb)
- [Input method and limitations](AUTO_GEOMETRY_BANDS_20260924.md)
- Model revisions: mixed-v3 `85d0c91c26f8e088849096dded7c9ba10b4cd9c9`;
  Microsoft `93450be3f1ed40a930690d951ef3932687cc1892`.
- Transformers 4.57.6, huggingface_hub 0.36.0, jiwer 4.0.0, Pillow 11.3.0,
  Torch 2.11.0+cu128; greedy FP32 decoding on Tesla T4.

Published artifacts contain aggregate metrics and provenance, not new weights
or raw OCR predictions. The result ZIPs remain local.
