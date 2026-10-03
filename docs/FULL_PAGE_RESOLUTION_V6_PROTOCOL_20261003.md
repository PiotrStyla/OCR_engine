# Full-page resolution v6: paired Qwen diagnostic

Status: frozen protocol, GPU execution pending. No v6 model result exists yet.

[Public frozen protocol and CPU validation on HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/6d92cebae68172c4acb7b21b1ecbd04a14444aa6/experiments/2026-10-03/full-page-resolution-v6-protocol).
All four published files were independently downloaded and SHA-256 verified.
The original protocol snapshot remains unchanged; this link is a later receipt.
The notebook pins published code commit `6a9896b1c5ebdb941b29dfd3d48c4cb2690658dd`.

## One Notebook

Open [the resolution notebook in Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_full_page_resolution_v6.ipynb), select GPU, then Run all.
No uploads, token, paid API or second notebook. Return
`full-page-resolution-v6-evidence.zip`. After a failure, use the final download
cell if setup completed; partial evidence is useful and must not be discarded.

## Selection and Controls

The public 15-page v2 bundle is downloaded at HF dataset revision
`d7a0bed7eabb68ae1231ea68f2a4461dd4c190ef`, with archive/member/image hashes
verified. Only three validation pages enter each worker, in this frozen order:

- `Slawna_wiktoria_FT__437089`: the v5 numeric loop on a title page.
- `Wiesc_FT__436884`: dense historical body and glyph errors.
- `Choragiew_FT__436799`: two user-confirmed `Poſłał` instances.

This is **post-hoc selection based on v5 errors**, not a held-out benchmark.
Both 1 MP and 4 MP are rerun in the same session; v5 is not reused as the
paired baseline. Separate subprocesses release model memory between arms.
Order is fixed: 1 MP first, 4 MP second. Timing is sequential diagnostic data,
not a randomized performance benchmark.

Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`,
the exact v5 prompt, packages, NF4 double quantization, FP16 compute, SDPA,
greedy decoding, seed 42 and 4096 output-token limit are unchanged.
The **only model-spec change** is `max_pixels`: 1,048,576 versus 4,194,304.
Both retain `min_pixels=262144`. No crops, oracle regions, deskew, cleanup,
repetition penalty or adaptive retries. Actual processed grids, dimensions,
visual-token counts and generated token IDs are retained.

The 1 MP profile completed a previous T4 run. **4 MP has not been validated
on T4** and may run out of memory. We do not lower its budget automatically.
A failed page remains in the evidence and in the all-page score.

## References and Claim Boundary

References are unchanged v2 single-review drafts: zero adjudicated gold pages.
Content scope, peripheral text and reading order are not fully adjudicated.
The sanitized worker manifest contains no reference text, notes or boxes.
Original full manifests, review history and v5 predictions remain unchanged.

CER/WER normalize only NFC and whitespace. Preserve `ſ`, `á`, accents,
capitalization and source spelling; no historical-to-modern substitutions.
Errors/missing outputs count as empty text. All loops and capped outputs remain.
Raw character counts and exact `Poſłał` substring counts are diagnostic aids,
not recall or validated glyph accuracy. Inspect grids and paired outputs before
attributing any change to greater input detail.

This test can guide the next experiment. It cannot establish SOTA, approve an
automatic teacher, create clean training labels or measure spatial reading-order,
omission or hallucination accuracy. A useful result must next be evaluated on
independently reviewed, document-disjoint full-page references.

## Evidence

The rerunnable ZIP contains raw predictions, arm configurations, logs, token/input
traces, metrics, per-page CSV, code/input provenance and SHA-256 checksums.
It excludes scans, weights and program files. Public inputs are independently
reconstructible from pinned HF sources. Rerunning preserves checkpointed errors;
use a fresh session for a new attempt rather than deleting evidence.

119 focused CPU tests passed, all 15 source images/hashes were verified, and
nbformat plus all eight Python cells validated. These checks do not substitute for an actual
Colab GPU execution. The first v6 GPU run is performed by the user.
