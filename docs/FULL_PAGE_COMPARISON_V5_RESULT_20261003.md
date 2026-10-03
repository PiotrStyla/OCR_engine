# Full-page comparison v5: audited GPU result

The returned evidence archive passed checksum/provenance validation and all-page
CPU rescoring. Qwen ran all 15 validation pages on the reported Tesla T4 runtime.
It has fewer capped outputs and lower provisional error than the retained Ovis
profile, but still fails reliable diplomatic transcription and cannot become an
automatic historical-text teacher or production OCR profile.

## Evidence Binding

- Archive SHA-256: `bf6ddc8cb25c6041f034b2fff05a933d5f9197ba1025a9f2c71c6c63e25e62e7`.
- All 30 members verified, with complete coverage of 29 file checksums.
- Code revision: `0d74a667f94901516dd9d3f75f9e478ca63502be`; runner and comparison
  hashes match Git source bytes at that revision.
- Config matches the frozen v5 protocol, including model/revision, prompt, NF4,
  FP16, SDPA, pixel budget, package pins and retained-baseline identity.
- Frozen v2 manifest and baseline SHA-256 match the published inputs exactly.
- All 15 unique validation pages are paired. No train/final-test pages enter.
- Worker inputs match the sanitized manifest projection exactly: no reference,
  notes, source PAGE text or annotation boxes. Model input regions are empty.
- Generation token counts, EOS/cap flags and checkpoint identity are consistent.
- Full metrics and candidate scoring projection reproduce exactly on CPU.

Runtime reported by evidence: Python 3.13.15, Torch 2.11.0+cu130, CUDA 13.0,
Transformers 4.57.6, bitsandbytes 0.48.1, accelerate 1.13.0, Tesla T4.
Worker exit status 0; model-load error null. Installation warnings about inherited
Gradio/diffusers requirements did not prevent this separate model environment run.
Checksum integrity and metadata consistency are not cryptographic runtime attestation.

## All-Page Results

| Profile | Pages | CER Micro | WER Micro | Runtime Errors | EOS | Capped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ovis retained v4 | 15 | 237.2283% | 514.0136% | 0 | 11 | 4 |
| Qwen3-VL-4B v5 | 15 | 63.3478% | 132.7211% | 0 | 14 | 1 |

Qwen has lower per-page CER on 11 pages, higher on 3 and an equal score on 1.
Primary metrics retain the entire raw output of every page, including loops.
No repetition removal, truncation repair, spelling normalization or survivor filtering.
The retained Ovis result is not a new GPU execution or a model improvement.
Prompts, quantization and runtime profiles differ, so this is not a controlled
model-weights-only comparison. References remain single-review drafts, not gold;
peripheral-text inclusion and spatial reading order are not fully adjudicated.

## Remaining Failures

`Slawna_wiktoria_FT__437089` reaches 4096 tokens and repeats `44.` after the title
and library-stamp text. It takes 489.9 seconds, more than half of candidate page
inference time. The scan has a short printed title, a stamp and handwritten marks,
not the extensive repeated numeric sequence. This is a real capped degeneration,
not merely a successful long transcription. Its output remains in primary metrics.

For diagnosis only, the 14 EOS pages have CER 19.5057% and WER 64.3005%.
This is a **post-hoc, outcome-selected subset**, not a replacement benchmark or
evidence that EOS certifies completeness. The all-15 score above remains primary.

Historical glyphs are not reliably retained despite the prompt:

| Character | Draft Reference Occurrences | Raw Qwen Occurrences |
| --- | ---: | ---: |
| `ſ` | 266 | 6 |
| `á` | 291 | 112 |
| `ɇ` | 66 | 0 |

These are aggregate counts, **not glyph recall or an alignment-based accuracy**.
They signal a review need; they cannot distinguish every substitution, deletion,
extra occurrence or reference-scope issue. Direct examples support the concern:
on `Choragiew_FT__436799`, the two user-confirmed `Poſłał` passages are still
rendered as `Połtał` in Qwen output. The title's `Woyſk` also becomes `Woyłk`.
No manual corrections were inserted into model predictions or gold labels.

## Resource Observations

Summed page inference: 936.9915 seconds (about 15 minutes 37 seconds).
Model loading: 154.2077 seconds as reported separately by the worker.
Peak PyTorch allocated CUDA memory: 4,010,851,840 bytes, about 3.7354 GiB.
This is not total device use measured by nvidia-smi. The frozen 4-bit profile
completed on the reported T4; its long-s fidelity and termination remain inadequate.
Images are reduced to roughly 864-928 pixels wide at the one-megapixel budget;
fine historical glyph detail may be lost. This is a hypothesis, not a demonstrated
cause: resolution, font familiarity and output scope have not been isolated yet.

## Decision and Next Experiment

Retain Qwen as a technically executable candidate for controlled development.
Do not promote this profile to production, automatic transcription teacher or SOTA.
Do not put these validation pages into the recognizer training pool.

Next: freeze a small **post-hoc development resolution diagnostic**, keeping
model, prompt, quantization and decoding fixed. Compare the one-megapixel profile
with a higher-resolution profile on named title/body cases; examine the `ſ`/accent
errors and numeric loop without rewriting outputs. Region/line recognition is a
separate subsequent experiment and must distinguish automatic from oracle boxes.
Only after this diagnosis should we pick clean training targets for the specialized
historical recognizer and resume the DATA ENGINE v2 teacher/review/distillation path.

The audit implementation and existing safeguards passed 108 focused CPU tests.
The original ZIP, configuration and protocol remain unchanged.
