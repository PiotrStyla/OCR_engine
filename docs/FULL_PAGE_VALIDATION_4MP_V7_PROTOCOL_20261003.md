# Full-page validation 4MP v7

Status: frozen protocol, first complete 15-page 4MP GPU run pending.
The three-page v6 result is not a result for this protocol.

[Public frozen protocol and CPU validation on HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/a66694ffcb39fbcd9c07a8b98d7d39aa18c7b498/experiments/2026-10-03/full-page-validation-4mp-v7-protocol).
All four files were downloaded and SHA-256 verified. This link is a later
publication receipt; the frozen snapshot remains unchanged.
Notebook code is pinned to `990146abb83005205e50dc28238c9767915a7efb`.
The published notebook at `9e1b98c2d429489609b64d4e0f115e535d72df32` was
independently downloaded, JSON-compared, nbformat-validated and syntax-checked.

## One Notebook

Open [the single v7 notebook in Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_full_page_validation_4mp_v7.ipynb), select GPU, then Run all.
No uploads, tokens, paid APIs or second notebook. Return
`full-page-validation-4mp-v7-evidence.zip`. If a later cell fails, use the final
download cell after setup completed; retain partial evidence.

## Frozen Inputs and Profile

All 15 pages of the existing v2 validation split are included. No training or
final-test pages, no outcome filtering. The public input ZIP at HF revision
`d7a0bed7eabb68ae1231ea68f2a4461dd4c190ef` is downloaded and checked against
the frozen archive, member, manifest and image hashes/dimensions.

Model: Qwen3-VL-4B-Instruct revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, Apache-2.0.
The model spec is exactly the 4MP arm from v6: min_pixels 262144,
max_pixels 4194304, NF4 double quantization, FP16 compute, SDPA, greedy decoding,
seed 42 and max_new_tokens 4096. Exact v5/v6 prompt and pinned packages retained.
No repetition penalty, adaptive retries, oracle crops, deskew or spelling cleanup.
Reference text, notes, PAGE XML and boxes do not enter the worker manifest.

This profile fitted three T4 pages in v6. Fit and reliable EOS on all 15 are
not yet verified; a failed/capped page remains in the primary score and ZIP.
Existing checkpoints, including errors, are retained on resume. Use a fresh
session for a new attempt, rather than deleting inconvenient results.

## Retained Baseline and Interpretation

Only 4MP is run on GPU. The exact raw 15-page Qwen 1MP predictions from v5 are
downloaded from the public evidence repository at revision
`5f30a4d0e32f3e4606c6a37690e3e21b59022d7b`. The original archive hash is
`bf6ddc8cb25c6041f034b2fff05a933d5f9197ba1025a9f2c71c6c63e25e62e7`;
raw-prediction hash is
`3aec77bd1e63ae736a8d6110a40fbb40bafa6cb7bc6547659797e95eff6795a1`.
The independent v5 auditor checks ZIP safety, all checksums, frozen inputs/code,
worker identity/packages, token/EOS traces and recomputes its scores before staging.
The baseline inference revision is `0d74a667f94901516dd9d3f75f9e478ca63502be`.

**These are separate runtime sessions**, not a same-session paired causal trial.
Only max_pixels changes in the model spec, but that does not make the runtime,
GPU environment or execution order controlled. Time and VRAM cannot establish
a speed/memory ranking across sessions. Actual grids/pixels are retained.
Ovis predictions in the input bundle are checked for bundle integrity only;
Ovis is neither run nor scored in v7.

## Metrics and Boundaries

Primary CER/WER includes all 15 pages, with errors/missing outputs as empty
text and all capped/looped output preserved. Raw predictions remain unchanged;
normalization is NFC and whitespace only, preserving `ſ`, `á`, accents and case.
Long-s/accent/`Poſłał` counts are diagnostics, not recall or glyph accuracy.
Actual input geometry and per-page scores are exported, with package/code/input
provenance, logs, a plot and SHA-256 checksums. The evidence ZIP excludes scans,
weights and program files; source artifacts are reconstructible from pinned HF.

The v2 reference texts remain single-review drafts with zero adjudicated gold
pages. Peripheral-text content scope and reading order are unadjudicated.
The protocol was chosen after observing v5/v6 on this validation split; it is
not an untouched final-test benchmark. A successful technical run does not
authorize an automatic teacher, clean training labels, production or SOTA.

CPU tests, source staging and notebook format/syntax checks are performed locally.
GPU execution and top-to-bottom Colab validation await the user's run.

The focused suite passes 134 CPU tests. All 15 source images/hashes and the
real published v5 download were checked; v5 CER 63.3478% / WER 132.7211%
were reproduced unchanged. All 15 absent candidate predictions were verified
to score as empty rather than be excluded. nbformat and all eight code cells validate.
