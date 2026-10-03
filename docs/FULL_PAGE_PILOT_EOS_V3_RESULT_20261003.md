# Full-page EOS v3: termination verified on two pages

Date: 2026-10-03. Development diagnostics, not an independent benchmark.
Archive SHA-256: `d69c45d244e5dea495f5a46357193337b018d7089f57af9dc00417bd4f47fee7`.

All 17 ZIP members passed SHA-256 coverage verification. The archived configuration,
model/input identities and runner provenance match the frozen experiment. The pinned
source metadata hashes, unchanged reference fields and deterministic page selection
also passed verification. Micro CER
and WER were independently recomputed from raw predictions and match the archive.
Generated token counts, effective EOS settings, stop positions and finish reasons
were checked against the retained token IDs.
The audit and validation-scope changes passed 81 focused CPU tests. The new
15-cell notebook passed nbformat validation and Python-cell syntax checks.
GPU execution of the expanded scope remains pending.

[Public HF evidence, audit and next frozen configuration](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/764cf402c80c9c0d64dd107302399421a58994fd/experiments/2026-10-03/full-page-pilot-eos-v3-result).
All 24 publication files exclude code, scans and weights. The published source
ZIP was downloaded again and its SHA-256 matches the user's original archive.
Publication provenance identifies code commit `c80e7628dfe48f72fa96137a4f8db742395147f2`.

## Observed Result

OvisOCR2 completed 2/2 pages on a Tesla T4 with no OOM or token-limit hits.
Both generations stopped on token 248046 (`<|im_end|>`); each contains exactly one
configured EOS, at its final position. No repeated output was removed.

| Page | Generated Tokens | Final EOS Position (Zero-Based) | Generation Time | Peak Allocated VRAM | CER | WER |
| --- | --- | --- | --- | --- | --- | --- |
| Slawna_wiktoria_FT__437103 | 336 | 335 | 47.32 s | 3.308 GiB | 22.3087% | 64.3478% |
| SLAWNA_VICTORIA_FT__437145 | 324 | 323 | 43.70 s | 3.281 GiB | 27.3292% | 77.1186% |

Micro CER: **24.5936%**. Micro WER: **70.8155%**. Macro CER: 24.8189%.
Model load: 42.91 seconds. Runtime: Python 3.13.15, Torch 2.11.0+cu130,
Transformers 5.18.0, Pillow 11.3.0. Per-page times exclude model loading/downloads.

## What Changed

V2 and v3 have byte-identical staged manifests and reference-free model inputs.
The pinned model, prompt, greedy decoding, 1,048,576-pixel budget, visual-token
budget and 4,096-token limit are unchanged. Effective v3 EOS IDs are
`[248044, 248046]`, with padding ID 248044. Each v3 raw text is an exact prefix
of its corresponding v2 response. This fixes continued generation after the
first transcription; it does not represent training or better first-pass OCR.

The v3 traces record model-default EOS 248044 and tokenizer EOS 248046.
The explicit override stopped correctly on 248046 in both pages. V2 did not
retain token IDs, so the claim that it ignored an emitted EOS remains an
inference, not directly observed v2 evidence.

| Variant on the Same Two Pages | Diagnostic CER | Diagnostic WER | Token-Limit Pages |
| --- | --- | --- | --- |
| mixed-v3 automatic row/column ordering (v2 archive) | 72.8622% | 92.2747% | 0/2 |
| mixed-v3 source-region-assisted (v2 archive) | 45.6537% | 85.4077% | 0/2 |
| Ovis v2 raw repeated generation | 860.4947% | 1388.8412% | 2/2 |
| Ovis v3 raw EOS-terminated generation | 24.5936% | 70.8155% | 0/2 |

The 48.2686 percentage-point CER gap versus automatic mixed-v3 is a small-sample
development observation. Source-region-assisted OCR is not automatic layout and
must remain a separate diagnostic. Markdown presentation is projected to visible
text for scoring; Unicode NFC and whitespace normalization do not modernize spelling.

## Remaining Problems

The source references are unreviewed. EOS termination does not prove complete
page coverage, correct reading order or correct historical characters. Residual
OCR errors remain substantial. Page two includes the Cyrillic word `моего` and
the English-looking `myselfi`; both remain in raw predictions rather than being
silently corrected. Historical `á` and `ſ` must not be normalized into modern letters.

Only two inspected historical development pages were used. No modern-document,
table or independent collection-held-out claim follows. Pretraining contamination
has not been excluded. No model promotion or SOTA claim.

## Next Gate: Validation v4

[Run the single 15-page validation notebook](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_full_page_validation_v4.ipynb).
GPU, Run all, no uploads or HF token. Download `full-page-validation-v4-evidence.zip`.

The new configuration restricts inference to all 15 frozen validation pages from
five collections. It excludes the 65 training pages and the 36-page historical
final test. Two validation pages have already been inspected in v3, so this is
an expanded diagnostic, not a new independent test. Do not select only easy pages
or drop failures. Raw outputs, token traces, source audits and per-collection
CER/WER are retained. Failed pages are scored as empty.

Use a fresh Colab session and the separate `/content/slayer-full-page-validation-v4`
directory. The same proven model/EOS/pixel profile is used without repetition
penalties, automatic spelling correction or post-hoc deduplication. The 15-page
GPU run has not been executed locally. After it returns, inspect worst pages
and source-reference defects before selecting teacher labels or fine-tuning.
An independent reviewed test is still required before any SOTA comparison.
