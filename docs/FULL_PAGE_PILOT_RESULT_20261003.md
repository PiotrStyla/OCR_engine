# Full-page pilot v1: T4 runtime evidence

Date: 2026-10-03. This is a two-page development diagnostic, not a benchmark result.

Source archive SHA-256:
`7f35013373b28bbee726e07ec6e20c7d24ee37d6b2b6f11d5bbbf2772b477655`.
All 26 ZIP members and checksum coverage passed validation. Model/input identities
match the frozen v1 configuration. CER/WER were independently recomputed from
the archived raw predictions and source references and match the supplied metrics.
This check does not independently verify reference completeness or source pixels.

Pages: `Slawna_wiktoria_FT__437103`, `SLAWNA_VICTORIA_FT__437145`.
Runtime: Colab Python 3.13.15, Tesla T4 (14.56 GiB), Torch 2.11.0+cu130.
The ensurepip-free bootstrap worked and both model weights loaded.

| Variant | Successful Pages | Diagnostic CER | Diagnostic WER |
| --- | --- | --- | --- |
| mixed-v3 row-major | 2/2 | 72.8622% | 92.2747% |
| mixed-v3 column-order | 2/2 | 72.8622% | 92.2747% |
| mixed-v3 source regions | 2/2 | 45.6537% | 85.4077% |
| OvisOCR2 full page | 0/2 | unavailable | unavailable |

Ovis raised CUDA OOM on both pages, attempting allocations of 36.31 and
24.13 GiB. The evaluator's 100% CER/WER is the empty-output failure penalty,
not a measured recognition error rate. Both v1 workers exited zero despite page
errors, so exit status alone must not be used as an inference success criterion.
The updated worker reports failed-page counts and exits 2 when errors are retained.

Annotation-assisted crops reduce diagnostic CER by 27.2085 percentage points on
these pages. This supports investigating segmentation; it does not isolate a
detector-only effect or prove generalization. Row/column outputs were identical.
Source references are unreviewed, have historic Unicode issues and may be
incomplete. No model promotion, independent test score, or SOTA claim.

## Next Run: Separate T4 Profile v2

[Open the T4 v2 notebook](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_full_page_pilot_t4_v2.ipynb).
Select GPU and Run all. No files or token need uploading.

- Same two pages, model revisions, prompt and normalization; original scans unchanged.
- Ovis processor budget: 1,048,576 pixels versus v1's 8,294,400.
- Generation cap: 4,096 new tokens versus 8,192; exhaustion remains recorded.
- At most 1,024 post-merge visual tokens for the pinned 16-pixel patch/2x2 merge processor.
- Use the current `processor_kwargs`/`images_kwargs.size` interface rather than deprecated pixel kwargs.
- Record actual processed dimensions, grid and visual/input token counts before GPU transfer.
- Refuse over-budget input before generation; no hidden OOM retries or model substitutions.
- Separate `/content/slayer-full-page-pilot-t4-v2` directory retains v1 failures unchanged.
- Download `full-page-pilot-t4-v2-evidence.zip` after the run, including after errors.

The smaller resolution can lose fine print. This is a constrained runtime smoke,
not the official upstream quality profile, and v1/v2 outputs must not be conflated.
The memory change has CPU regression coverage but still needs remote GPU execution.
Actual input geometry must confirm the requested budget; a configuration value
alone is not evidence that the processor honored it.

Official processor sources:
[Transformers chat-template argument handling](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/processing_utils.py),
[Qwen2-VL image resizing](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py).
