# PolOCRBench test A: pomiar SOTA GPU (Runpod) — wynik

## Run

Two systems measured on the frozen test A (36 pages, IMPACT `history_print`,
`PiotrSty/impact-print-v2@a2480fde…`, archive SHA-256 and all 36 image hashes
verified at staging) under the frozen zero-shot prompt where the model takes one
(`polocrbench-zero-shot-prompt-v1`, template A verbatim), temperature 0,
`max_tokens` 4096. Metric: `training/transcription_eval` v1.1.

Platform: Runpod pod `9z5bgz2idmq83h` (`polocrbench-sota`), **1× RTX 4090 24 GB**,
EU-RO-1, `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, CUDA host 13.0,
runtime code pinned at `607ae333a6f8c61457aa0372eb941679f2c86088`. Qwen3-VL runs
in the image's torch (4-bit NF4, fp16, sdpa, `max_pixels` 4 MP); PaddleOCR-VL in
an isolated Python environment (`paddlepaddle-gpu` 3.3.1, cu126) so its NVIDIA
libraries cannot break torch. Pod lifetime 11:32–12:44 UTC = 1.2 h × $0.89/h
≈ **$1.06 total**.

Evidence: `experiments/2026-10-10/runpod-sota/` (per-model `predictions.jsonl`,
`score.json`, `run.json`, `receipt.json`, `audit.json`,
`polocrbench-sota-measurement-v1-evidence.zip`, `checksums.sha256`). Local audit
recomputed both scores from the returned per-page predictions and verified every
pin; archive SHA-256 `dac9ef12216f79c28797b35407a62a04fe1e3c5d5b6a0a5ebeed75e6071b306a`.
Measurement only: no tuning, no promotion.

## Consolidated Results (all 36 pages)

| System | Class | CER micro | WER micro | Structure | Errors |
| --- | --- | ---: | ---: | ---: | ---: |
| Tesseract.js 7 (CPU baseline) | classic OCR | 34,94% | 81,23% | — | 0 |
| `gpt-4o-mini` (API) | small VLM | 58,11% | 84,66% | 0,246 | 0 |
| **PaddleOCR-VL-1.6** (RTX 4090) | OmniDocBench pipeline | 33,80% | 85,60% | 0,250 | 0 |
| **Qwen3-VL-4B 4-bit** (RTX 4090) | open-weight VLM | **20,03%** | **61,36%** | **0,588** | 0 |
| `qwen/qwen3-vl-235b` (API) | large open VLM | 20,48% | 59,56% | 0,557 | 1 |
| `openai/gpt-5.4` (API) | frontier | **16,87%** | **53,78%** | **0,763** | 0 |

Secondary (35 pages, the broken-GT errata page `NA2_FT__434735` excluded
symmetrically for every system):

| System | CER micro (35 pages) |
| --- | ---: |
| `gpt-4o-mini` | 56,18% |
| PaddleOCR-VL-1.6 | 30,83% |
| Qwen3-VL-4B 4-bit | 17,06% |
| Qwen3-VL-235B | 17,13% |
| `openai/gpt-5.4` | **13,46%** |

Per-collection CER micro (36 pages):

| System | NA2_FT (15) | Nowiny_z_Rakuz_FT (15) | Powodzenia_FT (6) |
| --- | ---: | ---: | ---: |
| Tesseract.js 7 | 27,35% | 43,93% | 34,83% |
| PaddleOCR-VL-1.6 | 20,22% | 37,73% | **78,79%** |
| Qwen3-VL-4B 4-bit | 17,61% | 22,28% | 22,29% |
| `openai/gpt-5.4` | 17,22% | 16,46% | 16,93% |

## Findings

1. **The frontier ceiling on the frozen test is 16,87% CER** (`gpt-5.4`; 13,46%
   on clean pages). That is the number our recognizer work must beat to claim
   SOTA territory for Polish historical print.
2. **A 4B open model matches a 235B endpoint.** Qwen3-VL-4B 4-bit on one RTX 4090
   scores 20,03% (17,06% clean) versus 20,48% (17,13%) for Qwen3-VL-235B through
   an API — at effectively zero inference cost per page. Local open-weight is a
   viable production path, not a compromise.
3. **OmniDocBench SOTA does not transfer to this domain.** PaddleOCR-VL-1.6, the
   #1 document-parsing pipeline on OmniDocBench, scores 33,80% CER here — level
   with the Tesseract.js CPU baseline (34,94%) and 14 points behind a 4B VLM.
   Its per-collection profile is uneven: 20,22% on NA2 but 78,79% on
   Powodzenia_FT (6 pages), i.e. it collapses on that collection. Leaderboard
   position on modern CN/EN documents says nothing about 18th-century Polish
   print.
4. **All six systems land between 13% and 58% CER** on this set; the frozen test
   is discriminating across system classes and the reference noise (one broken
   page, U+FFFD/PUA characters) is now quantified by two independent runs.

## Decision And Next

The SOTA question is now answered with measured numbers instead of estimates:
**the gap to frontier zero-shot is ~3 CER points for the best open-weight system
(20,03% vs 16,87%) and the best frontier result is 16,87%.** Our own recognizer
path (data engine + recognizer v3) has a concrete target: below 17% CER on the
frozen test to be competitive with frontier zero-shot, below 13,5% to match
frontier on clean pages.

Still open, in order of leverage: reference repair (the errata page + U+FFFD/PUA
review) so the leaderboard number is trustworthy; the per-line geometry
experiment to unlock more training data; recognizer v3 training against these
two new anchors. The GPU run requires no further action — the pod is terminated.

## Reproduce

```powershell
# skrypt jednokomorkowy dla poda GPU (RTX 4090), pin rewizji w srodku
python training/runpod_sota_testA.py
python -m training.audit_polocrbench_sota_evidence `
  --archive polocrbench-sota-measurement-v1-evidence.zip `
  --benchmark data/impact-test-a-staged --output runs/runpod-sota/audit
```

Limitations: single frozen subset of historical print; the metric keeps case,
diacritics and historical spelling; one reference is known-broken (primary score
retains it); PaddleOCR-VL used its default generation settings while Qwen3-VL
used greedy decoding — both recorded in `run.json`; package versions recorded,
GPU wheel builds not independently re-verified.
