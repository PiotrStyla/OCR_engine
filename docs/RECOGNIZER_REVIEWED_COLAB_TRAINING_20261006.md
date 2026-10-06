# Reviewed recognizer: one Colab training run

**Update: completed return audited.** Training succeeded and produced full
weights, but the candidate regresses on ordinary print and combined development
CER. [Result and decision](RECOGNIZER_REVIEWED_COLAB_RESULT_20261006.md).
Do not repeat this unchanged recipe as the next experiment.

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

## First Returned Run: Failed Before Training

The returned `recognizer-reviewed-colab-v1-evidence.zip` has SHA-256
`90d2450f11ddfb8b802079639699ca8b7eb530bf28a7da806549b15d5ed9bdea`.
All 13 unique members are present; the 12 checksummed payloads verify. The
baseline metrics were independently recomputed from all 84 raw predictions:

| Development domain | Lines | CER | WER |
| --- | ---: | ---: | ---: |
| Ordinary print | 75 | 5.3307% | 22.8800% |
| Historical text | 9 | 33.0317% | 81.0811% |

These are baseline results, not candidate results. Adapter creation failed:
Colab supplied `torchao==0.10.0`, whereas the pinned PEFT 0.19.1 checks for
TorchAO >=0.16.0 even when wrapping ordinary, non-quantized linear layers.
The archive contains no candidate metrics or training run metadata. No trained
candidate weights are available from this return.

[Original evidence ZIP and recomputed audit on HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/e3850b5d2004b3b84b76aa6951c25a6a048adc9d/experiments/2026-10-06/recognizer-reviewed-colab-torchao-failure-v1).
All published files were downloaded at this pinned revision and hash-checked.

The corrected setup installs TorchAO 0.17.0 for Torch 2.11, or 0.16.0 for
Torch 2.10, without replacing Colab's CUDA Torch. Other Torch minor versions
stop explicitly rather than silently choosing an unverified ABI combination.
See the [official TorchAO compatibility table](https://github.com/pytorch/ao/issues/2919)
and [PEFT 0.19.1 availability check](https://github.com/huggingface/peft/blob/v0.19.1/src/peft/import_utils.py).

The setup now exercises a tiny real ViT/TrOCR architecture: adapter attachment,
aligned loss, finite nonzero gradients, optimizer update, merge and generation.
The runner repeats this check on the actual pinned base model and one TRAINING
sample before baseline inference. Both checks use disposable models; their
updates are not carried into training or checkpoint selection. The shared LoRA
helper uses generic PEFT, preserving the `pixel_values` interface instead of
adding text-encoder `input_ids` through the text seq2seq wrapper.
TorchAO and the full-model preflight report are included in future evidence.

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
Its corrected runtime code is pinned to the already-pushed full commit
`fe074fee1da33a94ea60a0bcd658a8f6afbb7dc8` in the setup cell.
Do not reuse the earlier notebook pinned to `ae6f779d78cedcd15135c61b34175b56e389c061`.

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
Focused validation after the environment repair: 70 tests passed. The Torch training-protocol test
module was skipped locally because Torch is not installed in this CPU tooling
environment; the new tiny/full-model adapter checks are executed by Colab,
not claimed as locally validated GPU runs. At the time of this repair, the
corrected complete run still required execution on the user's Colab GPU. The
later successful return is audited in the result document linked above.
