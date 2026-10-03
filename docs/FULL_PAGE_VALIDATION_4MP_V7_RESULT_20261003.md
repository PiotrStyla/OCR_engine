# Full-page validation 4MP v7: audited result

Status: all 15 validation pages completed on a reported Tesla T4, with 15 EOS,
no capped outputs, no runtime errors and no model-load error. This establishes
technical execution of the profile on this split, not historical OCR correctness.
The references remain single-review v2 drafts, with zero adjudicated gold pages.

## Integrity

Original user archive SHA-256:
`b6d85d9ac57366d09e93ce66c3a6466d3bca2a03d1aeef2f24318adf88286505`.
All 69 members and complete checksum coverage were checked. Frozen configuration,
source manifest, all image hashes/dimensions, reference-free inputs, worker identity,
code hashes, seven pinned import-preflight packages, recorded worker packages,
input grids and token/EOS termination were verified.

The original v5 evidence archive was independently audited again; the included
baseline audit report and nested original evidence match that source. v5 raw
predictions are byte-identical to their pinned SHA-256. All v7 metrics, per-page
diagnostics, scored projections and scoring configuration were recomputed on CPU
and match the returned evidence. Original files and ZIP remain unchanged.

Inference revision: `990146abb83005205e50dc28238c9767915a7efb`.
Dataset revision: `d7a0bed7eabb68ae1231ea68f2a4461dd4c190ef`.
Reported runtime: Tesla T4, Python 3.13.15, Torch 2.11.0+cu130, CUDA 13.0.
Worker exit status 0. No GPU inference was performed on the local laptop.
The focused regression suite passes 143 CPU tests, including nine new audit
tests covering recomputation and rejection of mutated provenance/data/metrics.

## Primary All-Page Results

| Profile | Pages | CER micro | WER micro | EOS | Caps | Errors/missing |
|---|---:|---:|---:|---:|---:|---:|
| Qwen v5 1MP, retained | 15 | 63.35% | 132.72% | 14 | 1 | 0 |
| Qwen v7 4MP | 15 | 17.52% | 61.63% | 15 | 0 | 0 |

All 15 pages remain in the primary score. Only max_pixels changes in the model
spec, but these runs use separate sessions: no controlled same-session causal
claim or speed ranking. The same provisional reference content scope/ordering
limitations apply to both. CER/WER normalize NFC and whitespace only, retaining
historical letters, accents and case. No spelling cleanup or output deduplication.

CER improves on 12 pages and worsens on 3, with no ties. Regressions:

| Page | CER 1MP | CER 4MP |
|---|---:|---:|
| Relacja_koronacji_FT__436547 | 10.18% | 16.29% |
| Relacja_koronacji_FT__436563 | 9.65% | 10.91% |
| SLAWNA_VICTORIA_FT__437118 | 9.03% | 13.89% |

The largest all-page improvement comes from removing v5's title-page `44.`
loop, which generated 4096 tokens. The same title in v7 ends after 105 tokens.
The profile does not improve every page merely by receiving more pixels.

For explanation only, excluding the old capped page symmetrically from both
profiles gives 14-page CER **19.51% -> 16.97%**, WER **64.30% -> 61.13%**.
This secondary subset is selected by v5 output behavior, not an independent
benchmark or a replacement for the primary score. No page is dropped from the
published main results. Peripheral text on the title page remains unadjudicated.

## Historical Characters

| Raw substring | Reference | v5 1MP | v7 4MP |
|---|---:|---:|---:|
| ſ | 266 | 6 | 14 |
| á | 291 | 112 | 139 |
| ɇ | 66 | 0 | 0 |
| Poſłał | 3 | 0 | 0 |

These are raw occurrence counts, not recall or calibrated glyph accuracy.
Some long-s characters appear, but source-spelling preservation remains unreliable.
On Choragiew_FT__436799, v7 still writes `Pofał` in the `Poſłał` contexts.
Two of the three draft instances were explicitly confirmed by the user; none
was modernized in the references or predictions during this analysis.

## Resources

Summed page-inference time: 744.80 s (~12 min 25 s); model load is separate,
175.29 s. Max generated output was 626 tokens; no response reached 4096.
Peak PyTorch allocated memory: 9,256,014,336 bytes (~8.62 GiB), not total GPU use.
Actual input grids and processed-pixel counts were checked; 4MP is a ceiling,
and native smaller pages use fewer pixels. Fit on these 15 pages is not a
guarantee for other dimensions or model configurations.

## Decision

Keep Qwen 4MP as the technically validated diagnostic baseline for this split.
Do not approve it as an automatic historical-text teacher, gold-label generator,
production OCR engine or SOTA model. Historical character errors and word-level
error rates remain substantial, and three pages regressed.

The next data-engine step is targeted hard-example mining for source spelling:
review scan/line crops around `ſ`, `á`, `ɇ` and recurrent substitutions, establish
diplomatic transcriptions with provenance, then train/evaluate a specialized
recognizer on document-disjoint splits. Unreviewed draft text or Qwen output
must not silently become clean training labels. No new training run is launched
as part of this evidence audit.

## Published Evidence

The [public HF result directory](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/23f96f88553e710b567e0bb699060ab3c748213c/experiments/2026-10-03/full-page-validation-4mp-v7-result)
contains the unchanged original ZIP, independent audit, recomputed metrics,
per-page CSV, input/glyph diagnostics, plot, frozen configuration, report and
publication/checksum manifests. All ten published files were downloaded at
this immutable revision and verified by SHA-256. No scans, weights or code
are included. Audit code revision: `7419e3eb98548d8de513e86393a48b57f34d330f`.
The HF report is the pre-publication snapshot; this link is a later receipt.
