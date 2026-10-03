# Full-page comparison v5: Qwen and retained Ovis

Status: ready for a first GPU execution, not a completed model evaluation.

Follow-up: the returned v5 GPU evidence has now been audited.
[Result, remaining failures and next gate](FULL_PAGE_COMPARISON_V5_RESULT_20261003.md).
The original frozen protocol snapshot remains unchanged.

[Public frozen protocol on HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/4437258e02228c61d4018555ea857dea0b859d01/experiments/2026-10-03/full-page-comparison-v5-protocol).
The downloaded configuration was SHA-256 verified; no GPU results are published yet.

## One Notebook

Open [the single comparison notebook in Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_full_page_comparison_v5.ipynb), select a GPU runtime
(T4 is the target profile), then Run all. No file uploads, HF token, paid API or
second notebook is required. Send back `full-page-comparison-v5-evidence.zip`.
If a later cell fails, run the final download cell after setup has completed.
Do not upload the input-data ZIP as a notebook.

The notebook downloads the entire public input ZIP at HF revision
`d7a0bed7eabb68ae1231ea68f2a4461dd4c190ef`, verifies the archive SHA-256, every
member, the frozen manifest, baseline predictions, image hashes and dimensions.
All 15 validation pages are included. Training and final test pages are excluded.
Only image paths, hashes, dimensions and page IDs enter the worker, with empty
source-region lists: no references, notes, PAGE XML text or ground-truth boxes.

## Frozen Profiles

Candidate: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),
revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, Apache-2.0.
We reuse the model/revision and pinned dependency family already used by the
layout-teacher pilot, but this is a new transcription task and needs a new GPU run.
The loader uses `Qwen3VLForConditionalGeneration` from Transformers 4.57.6,
not the unavailable-in-that-release `AutoModelForMultimodalLM` alias.

The candidate uses 4-bit NF4 with double quantization, FP16 compute, SDPA,
greedy decoding and at most 4096 generated tokens. Image processing is bounded
by 262,144 to 1,048,576 pixels; actual grid, processed dimensions and visual-token
counts are saved per page. The full generated token sequence and EOS are retained.
No repetition penalty, adaptive retries, output deduplication or spelling cleanup.

Ovis is **not rerun**. Its exact saved projected predictions and unchanged raw
outputs from v4 are used. Their SHA-256 is
`b86afe2b427fab2cb15e9542c537b22a30508978fb5d02f293fc937dab9d3c19`.
The source v4 evidence archive SHA-256 is recorded in the experiment configuration.
This saves GPU time but means prompt, quantization, processor and runtime differ.
The comparison is between complete inference profiles, not a causal comparison
of model weights under identical conditions. Time/VRAM values cannot rank speed
or memory efficiency across different runtime executions.

## References and Scope

The v2 draft includes the two user-confirmed `Poſłał` corrections. Historical
spelling, `ſ`, `á`, accents and case are preserved. Metrics normalize only NFC
and whitespace. The raw candidate transcription is retained byte-for-byte as text;
plain outputs are not stripped of Markdown-like source punctuation.

The candidate prompt asks for all visible printed text, with peripheral text after
the main text. This is an explicit proposed output policy, **not** a declaration
that the reference content scope or reading order has been fully adjudicated.
Marginalia/catchword/signature comments remain notes, not appended references.
All metrics are provisional diagnostics against the exact v2 transcription fields.
Zero adjudicated gold pages; no training use, automatic teacher or SOTA promotion.

## Outputs and Failure Handling

- All-page CER/WER, macro CER, per-page diagnostics and CSV.
- Runtime errors and missing outputs scored as empty, without survivor filtering.
- Token-limit and EOS counts, generated tokens, elapsed time and peak allocated VRAM.
- Consecutive-character/word/line repetition flags, without changing output.
- Long-s/accent character counts as inspection aids, not spelling-accuracy metrics.
- Frozen configuration, input provenance, package versions, logs and a CER plot.
- Rerunnable evidence ZIP with SHA-256, no scans, model weights or program files.

No calibrated spatial reading-order metric, region omission accuracy or
hallucination accuracy is claimed. These require reviewed region/order annotations;
CER/WER and repetition heuristics are not substitutes.

Each page is checkpointed. Rerunning inference preserves completed records,
including errors. Resume is rejected if code, model or sanitized inputs change.
Use a fresh session for a new attempt rather than deleting evidence.
Environment creation avoids ensurepip and keeps model packages outside the kernel.
Installation and import preflight logs are retained even before inference begins.

## Local Validation

104 focused CPU tests passed. nbformat validated the generated notebook and all
eight code cells compile. Tests cover safe extraction, hashes, resuming,
reference-free inputs, historical normalization, raw token/EOS capture, error
inclusion, evidence filtering and compilation of every notebook Python cell.
The complete real v2 bundle was staged and checked locally. The retained Ovis
score on those references was recomputed: CER 237.2283%, WER 514.0136%, all 15
pages and no runtime-error exclusions. This is a reference-version diagnostic,
not an improved Ovis inference result.

No local CUDA inference or full top-to-bottom Colab execution was performed.
Qwen quality, T4 memory fit and runtime remain unverified until the evidence ZIP
returns from Colab. The next decision will use that output plus targeted visual
review of disagreements; these validation pages remain excluded from training.
