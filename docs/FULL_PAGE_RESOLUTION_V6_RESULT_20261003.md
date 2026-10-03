# Full-page resolution v6: audited result

Status: both arms completed on a reported Tesla T4. This is a three-page,
post-hoc development diagnostic, not a held-out benchmark or SOTA result.
The exact v2 references are single-review drafts, not adjudicated gold.

[Public evidence and independent audit on HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/bf2bf8372ea99c8fb3175fe12a8689f57f8e3fd9/experiments/2026-10-03/full-page-resolution-v6-result).
All nine published files were downloaded and SHA-256 verified. The snapshot
contains the unchanged user ZIP, recomputed scores, report and publication manifest;
no source code, scans or weights were uploaded to this evidence prefix.
Audit code revision: `a81bc2c394e931fe0f2cba828d190093dd3e1b14`.

## Integrity and Reproduction

Returned archive SHA-256:
`aeed9e2dcc5091d0f6114e306c955119349faeca53d9cb00278247ca4704c065`.
All 40 ZIP members were checked, including complete checksum coverage.
Frozen configurations, source/subset manifests, image hashes, sanitized model
inputs, code hashes, worker identities, recorded package versions, token/EOS
termination and actual input grids were verified. All metrics and per-page
diagnostics were independently recomputed on CPU and match the returned scores.
No model inference was performed locally. The original archive remains unchanged.

Inference code: `6a9896b1c5ebdb941b29dfd3d48c4cb2690658dd`.
Input dataset revision: `d7a0bed7eabb68ae1231ea68f2a4461dd4c190ef`.
Reported runtime: Tesla T4, Python 3.13.15, Torch 2.11.0+cu130, CUDA 13.0.
Both workers exited 0, with no model-load error and all 6 page inferences successful.
All seven pinned model packages match the recorded import preflight, including
sentencepiece 0.2.1. The six model packages recorded per worker also match their pins.

## Primary Paired Results

Same model/revision, prompt, greedy decoding, NF4/FP16, seed and output limit.
Only the model-spec `max_pixels` differs. Neither errors nor caps were removed.

| Budget | Pages | CER micro | WER micro | EOS | Capped |
|---|---:|---:|---:|---:|---:|
| 1 MP | 3 | 217.82% | 375.90% | 2 | 1 |
| 4 MP | 3 | 21.76% | 69.88% | 3 | 0 |

| Page | CER 1 MP | CER 4 MP | Actual pixels 1 MP / 4 MP |
|---|---:|---:|---:|
| Slawna_wiktoria_FT__437089 | 2985.29% | 54.41% | 1,032,192 / 4,128,768 |
| Wiesc_FT__436884 | 23.55% | 21.22% | 995,328 / 2,983,936 |
| Choragiew_FT__436799 | 20.43% | 17.40% | 1,039,360 / 4,157,440 |

The 1 MP title-page output repeats `44.` and reaches 4096 tokens; the 4 MP
output ends at EOS after 105 tokens. That loop disappearance dominates the
aggregate CER change. Both body pages also improve, but much less.

An explicitly secondary, explanatory score on the two body pages is
CER **22.10% -> 19.45%**, WER **73.48% -> 68.05%**. This does not replace
the three-page primary score and must not be presented as an independent benchmark.
The title reference does not cover every stamp/handwritten/peripheral element
requested by the prompt; content scope and reading order remain unadjudicated.

Actual processed pixels increased for all three pages. The 4 MP budget is a
ceiling, not a guarantee that every page contains four million processed pixels:
Wiesc is processed at 1504 x 1984, close to its native dimensions.
Visual-token counts rose from 1008/972/1015 to 4032/2914/4060.

## Historical Spelling Still Fails

Across the exact three reference texts there are 58 `ſ`, 73 `á` and 24 `ɇ`.
Raw outputs contain zero `ſ` and zero `ɇ` in either arm, and 25 / 21 `á`.
These counts are inspection aids, **not recall or calibrated character accuracy**.
Greater resolution does not establish preservation of historical spelling.

The Choragiew draft contains three `Poſłał` instances, two explicitly confirmed
by the user in v2. Neither arm emits that exact word: the 1 MP output uses
`Połtał`, whereas 4 MP uses `Pofał`. The title also still emits `Woyłk`
against the draft's `Woyſk`. No output was modernized or corrected after inference.

## Resource Measurements

Summed page-inference time: 584.38 s at 1 MP, 158.74 s at 4 MP.
Model-load time is separate: 133.08 s and 48.90 s.
The lower total at 4 MP is driven by avoiding the 4096-token loop. On both body
pages 4 MP took longer. This is not evidence that higher resolution is generally faster.

Peak PyTorch allocated memory: 4,010,851,840 bytes at 1 MP (~3.74 GiB),
9,226,438,144 bytes at 4 MP (~8.59 GiB). This is not total GPU memory usage.
4 MP fitted this T4 run; that does not guarantee fit for other pages or profiles.

## Decision and Next Gate

Keep 4 MP as a candidate inference profile, not an approved teacher or production
OCR engine. The three selected pages do not establish robustness across documents.
Next, test the unchanged 4 MP profile across all 15 validation pages, retaining
all failures/caps and comparing against the existing 1 MP v5 predictions.
Label that comparison as separate-runtime evidence, not a same-session paired trial.

Even a successful 15-page technical gate will not authorize automatic historical
labels. To improve source spelling, the data engine still needs reviewed diplomatic
transcriptions, glyph/error-targeted training and document-disjoint evaluation.
No clean-dataset generation, model promotion or SOTA claim follows from v6.

## Local Audit Tests

126 focused CPU tests passed. Audit regression tests cover immutable archive
preservation, exact score reproduction, checksum failures, injected reference
text, modified metrics, input geometry, code hashes and package-preflight versions.
