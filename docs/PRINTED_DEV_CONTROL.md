# Printed-text development control

After the EHRI sanity check, compare `microsoft/trocr-base-printed` with
`PiotrSty/trocr-pl-mixed-v3` on identical historical development crops.

## Run on Kaggle

1. Import `training/kaggle_printed_dev_control.ipynb` into a new notebook.
2. Enable GPU and Internet. Do not reuse the Kraken session.
3. Run All. Download the ZIP using the final embedded download link or Output.
4. Return the ZIP for review of predictions, crop geometry and reference defects.

No tokens, training or upload to Hugging Face are required. Source/model revisions
are pinned in the script. The ZIP contains crops, raw predictions, references,
metadata, runtime versions, checksums and `review.html`; it excludes model weights.

## Scope and limits

- Validation regions only; frozen test collections are rejected, and exact crop
  hashes are checked against test-region metadata. No near-duplicate audit.
- Select all references without line breaks and with at least 20 characters.
  At the pinned revision this yields 15 candidates across three collections,
  mostly headings and publication details. This is not representative body text.
- A reference without a newline does not prove the image contains one line.
  Inspect `review.html` before interpreting the aggregate result.
- NFC and whitespace normalization only. Keep historical spelling, private-use
  characters and replacement characters; flag the latter two rather than quietly
  removing difficult references. Ground truth has not been manually audited.
- Model-card training descriptions do not establish absence of contamination
  across all upstream checkpoints. This is development evidence, not held-out SOTA.
- Failed predictions remain empty strings in CER/WER denominators. Check errors
  and truncation flags before comparing model quality.
- Do not compare these region metrics directly with the 36-page Kraken/Tesseract
  results. No segmentation or reading-order quality is measured here.

Local verification covers selection, split guards, scoring and notebook syntax.
GPU inference still needs the Kaggle run; no model-quality result is claimed yet.
