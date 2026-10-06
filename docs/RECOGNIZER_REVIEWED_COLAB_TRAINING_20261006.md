# Reviewed recognizer: one Colab training run

## Scope

The user requested model training on Colab after confirming the 31 corrected
complete-line transcriptions. This is a bounded exploratory training recipe,
not a promotion of the frozen review packages or an independent SOTA benchmark.
Original strict train/evaluation flags remain unchanged. A distinct
`experimental_training_eligible` field records this experiment's selection.

The two immutable reviewed packages contain 115 accepted annotations in total.
Exclude all 36 entries from `NA1_FT`, `NA3_FT` and `Wyprawa_1_FT`, following the
existing protective work-family audit. Of the remaining 79, reserve the nine
`O_cieplicach_FT` lines for development and train on the other 70. Source page IDs
and exact crop hashes do not overlap between training and development.
Bibliographic independence of the other collections is not certified; results
must not be presented as an independent benchmark or gold-label evaluation.

## Frozen Recipe

- Base: `PiotrSty/trocr-pl-mixed-v3`, revision
  `85d0c91c26f8e088849096dded7c9ba10b4cd9c9`.
- 70 human-reviewed training lines repeated four times, plus 500 deterministic
  synthetic replay lines from the pinned `PiotrSty/ocr-pl-lines` training split.
- Three epochs, batch size two, accumulation four, learning rate `1e-5`,
  decoder LoRA rank 16/alpha 32, fp16 without 4-bit quantization, seed 42.
- Nine historical development lines plus the existing 75-line ordinary-text
  development set. Combined development CER selects the checkpoint.
- Baseline/candidate CER, WER and raw predictions saved separately per domain.
- No final-test runs, no automatic production promotion, no automatic HF model
  upload. Candidate weights are retained even if development results regress.

Training uses the existing `training.train_trocr_pl` and its aligned token loss.
Token length and tokenizer round-trip checks run before training. Image hashes
are checked across replay, reviewed training and both development sets. Original
annotations and historical glyphs are not modernized; metrics use only NFC and
whitespace normalization. Selection and metric provenance remain explicit.

## Automatic Inputs

[Complete training data ZIP](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/3a301fd12a6272b28774d1888213ddf528b47f24/data/recognizer-reviewed-colab-input-v1-20261006/recognizer-reviewed-colab-input-v1-20261006.zip?download=true).

ZIP SHA-256:
`3e39b7d02b90a0a741adde506c0467698c46fe86c82846d3cc0e648eebaae32e`.
Both fresh local preparations reproduce every file byte. Public files and all
ZIP members were downloaded at the pinned HF revision and hash-checked.
The ZIP includes the 79 image/text pairs, source ledger, exclusions, configuration,
report and checksums. IMPACT/PSNC source attribution and CC-BY-3.0 are retained.

## Run

[Open the single training notebook in Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_recognizer_reviewed_training_v1.ipynb).
Its runtime code is pinned to the already-pushed full commit
`ae6f779d78cedcd15135c61b34175b56e389c061`.

Open the notebook, select a GPU T4 runtime and click **Run all**. Do not upload
any files. The notebook fetches the complete, already-pushed runtime revision,
creates an isolated interpreter without `ensurepip`, inherits Colab's CUDA Torch,
installs pinned OCR dependencies outside the notebook kernel and runs training.
No package restart or multi-notebook choreography is needed by design.

After success, it downloads `recognizer-reviewed-colab-v1-result.zip`, containing
the full model ZIP, evidence ZIP and their checksums. On failure it downloads
only diagnostic evidence and reports failure explicitly; no missing result
variable is used. Re-running training creates a new run directory.

## Validation Boundary

CPU checks cover selection, source hashes, work exclusions, page overlap,
unapproved/modified text rejection, exact historical text retention and notebook
structure/code syntax. Full model training requires the user's Colab GPU session;
preparing or opening the notebook is not evidence that training has run.
Focused validation: 69 tests passed. The existing Torch training-protocol test
module was skipped locally because Torch is not installed in this CPU tooling
environment; no GPU execution is claimed by these checks.
