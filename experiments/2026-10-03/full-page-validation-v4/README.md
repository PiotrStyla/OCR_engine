# Full-page validation v4

Status: prepared, GPU run pending. OvisOCR2 inference only; no training.

Run `training/colab_full_page_validation_v4.ipynb` with GPU and Run all.
Inputs and weights download automatically from pinned public sources. No uploads.
Expected output: `full-page-validation-v4-evidence.zip`.

Scope: all 15 existing validation pages, five collections. The 65 train pages and
the final 36-page historical test are not eligible for inference. Metadata for the
full non-test pool is retained for provenance. Selection is deterministic and
does not depend on OCR scores. Two selected pages were inspected in EOS v3.

The model, prompt, pixel budget, greedy decoding, EOS override and token limit
are unchanged from the successful two-page v3 run. No spelling modernization or
prediction cleanup. Preserve source labels as unreviewed, all failed/capped raw
outputs and per-collection metrics. Never promote this diagnostic to gold/SOTA.

Next decision: identify dominant error types and source-reference defects across
collections, then build reviewed targets for specialized-model training.
