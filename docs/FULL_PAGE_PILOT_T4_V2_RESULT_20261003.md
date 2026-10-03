# Full-page pilot T4 v2: memory fixed, termination unresolved

Date: 2026-10-03. Two historical development pages; source references remain unreviewed.
ZIP SHA-256: `aeb8312b595854aa3bcbc26ae14f489b924bbe7d8b88191835f283742f9fff0a`.
All 26 members, configuration and input/model identities passed verification.
Micro CER/WER were recomputed from raw archived outputs and match the supplied metrics.

## Observed Runtime

Ovis produced output on 2/2 pages with no CUDA OOM. This verifies the constrained
T4 memory profile, not correct termination or transcription completeness.

| Page | Processed Size | Visual Tokens | Peak Allocated VRAM | Generation Time |
| --- | --- | --- | --- | --- |
| Slawna_wiktoria_FT__437103 | 896 x 1120 | 980 | 3.308 GiB | 218.55 s |
| SLAWNA_VICTORIA_FT__437145 | 864 x 1152 | 972 | 3.281 GiB | 216.09 s |

Both actual input areas are below the configured 1,048,576 pixels.
Both responses reached exactly 4,096 new tokens. The first response repeatedly
regenerates the page; the second has a long sequence of repeated `1` lines.
Generated `<think>`/`</think>` fragments and stray non-source script characters
are also visible. No repeated output was removed or silently corrected.

| Variant | Diagnostic CER | Diagnostic WER | Token-Limit Pages |
| --- | --- | --- | --- |
| mixed-v3 row-major | 72.8622% | 92.2747% | 0/2 |
| mixed-v3 column-order | 72.8622% | 92.2747% | 0/2 |
| mixed-v3 source regions | 45.6537% | 85.4077% | 0/2 |
| OvisOCR2 raw full page | 860.4947% | 1388.8412% | 2/2 |

CER/WER can exceed 100% due to insertions. These raw Ovis scores preserve the
generation failure, not a cleaned first-pass transcription. Successful worker
exit/status does not establish a usable complete response. The audit now separates
execution success from token-limited output and blocks the complete comparison flag.
No model promotion or SOTA claim.

## Source Configuration Finding

At model revision `1fc9221b7823a371d6e97f92d527cc847e24e107`, the text configuration
sets EOS to 248044 (`<|endoftext|>`), while the tokenizer's EOS is `<|im_end|>`
(248046). The Transformers 5.18 GenerationConfig factory imports the text-config
EOS when the outer configuration does not supply one. This is a concrete stop-token
inconsistency and a plausible cause of continued assistant output.

V2 did not retain generated token IDs or the effective GenerationConfig, so we
cannot establish that an `<|im_end|>` token actually occurred and was ignored.
The next run must test this hypothesis, not assume it is proved.

Pinned primary sources:
[Ovis model config](https://huggingface.co/ATH-MaaS/OvisOCR2/resolve/1fc9221b7823a371d6e97f92d527cc847e24e107/config.json),
[Ovis tokenizer config](https://huggingface.co/ATH-MaaS/OvisOCR2/resolve/1fc9221b7823a371d6e97f92d527cc847e24e107/tokenizer_config.json),
[GenerationConfig factory](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/generation/configuration_utils.py).

The upstream vLLM example itself contains a repeat-tail cleanup step. We do not
apply this to primary predictions or metrics: retaining raw output exposes
termination problems instead of hiding them.
[Upstream inference example](https://huggingface.co/ATH-MaaS/OvisOCR2/resolve/1fc9221b7823a371d6e97f92d527cc847e24e107/README.md).

## Next Run: EOS v3

[Open the single EOS v3 notebook](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_full_page_pilot_eos_v3.ipynb).
GPU, Run all, no uploads. Only Ovis runs; do not spend time rerunning mixed-v3.

Same model, pages, prompt, greedy decoding, image budget and 4,096-token cap as v2.
The generation change is explicit EOS IDs `[248044, 248046]` and tokenizer padding.
Token names and IDs must round-trip through the pinned tokenizer before inference.
Raw generated IDs, decoding with special tokens, effective EOS settings, stop
positions and finish reason are saved. EOS reached exactly at the cap is not
mistakenly classified as truncation. No repetition penalties, n-gram suppression,
post-hoc deduplication or spelling modernization.

Uses a new `/content/slayer-full-page-pilot-eos-v3` directory.
Return `full-page-pilot-eos-v3-evidence.zip`, including after failures.
If truncation persists, keep the failed raw outputs and reconsider the backend
before expanding to the 80-page comparison. Passing EOS alone does not verify
reference quality or prove OCR accuracy.
