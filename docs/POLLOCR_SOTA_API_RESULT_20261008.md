# PolOCRBench test A: pomiar API zero-shot — wynik

## Run

Three vision models measured on the frozen test A (36 pages, IMPACT
`history_print`, `PiotrSty/impact-print-v2@a2480fde…`, archive SHA-256 verified,
36/36 image hashes verified at staging) under the frozen zero-shot prompt
(`polocrbench-zero-shot-prompt-v1`, template A verbatim), temperature 0,
`max_tokens` 8192. Metric: `training/transcription_eval` v1.1.

Transport note: the frozen pages are TIFF files under `.jpg` names and chat
endpoints reject that format (`invalid_image_format`). `ocr/page_parser.image_message`
now transcodes them to PNG data URLs **without changing decoded pixels**; the
source files and their frozen hashes are untouched. Runtime code revision:
`0c9322ec61f213d8f96273bbcf2c0f0599b9b6d6` (commit "Transport IMPACT TIFF pages
as lossless PNG to chat endpoints").

Evidence: `experiments/2026-10-08/api-testA/` (per-model `predictions.jsonl`
with per-page `usage`, `run.json` with cost/tokens, `receipt.json`,
`checksums.sha256`). Measurement only: no tuning, no promotion.

## Primary Results (all 36 pages)

| System | CER micro | WER micro | Structure | Errors | Tokens in/out | Mean s/page |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tesseract.js 7 (CPU baseline) | 34,94% | 81,23% | — | 0 | — | 10,9 |
| `openai/gpt-4o-mini` | 58,11% | 84,66% | 0,246 | 0 | 966 288 / 13 067 | 10,3 |
| `qwen/qwen3-vl-235b-a22b-instruct` | 20,48% | 59,56% | 0,557 | 1 | 161 125 / 22 467 | 39,7 |
| `openai/gpt-5.4` | **16,87%** | **53,78%** | **0,763** | 0 | 108 480 / 23 206 | 21,6 |

Per-collection CER micro:

| System | NA2_FT (15) | Nowiny_z_Rakuz_FT (15) | Powodzenia_FT (6) |
| --- | ---: | ---: | ---: |
| Tesseract.js 7 | 27,35% | 43,93% | 34,83% |
| `openai/gpt-4o-mini` | 52,40% | 70,34% | 37,64% |
| `qwen/qwen3-vl-235b` | 22,05% | 19,12% | 18,61% |
| `openai/gpt-5.4` | 17,22% | 16,46% | 16,93% |

## Secondary Results (35 pages, broken-GT page excluded symmetrically)

Page `NA2_FT__434735` is an **errata table whose frozen reference is 4
characters** (`'Tam‑'`). All three models transcribe the actual page (1,013 to
1,790 characters), so its CER (252–447) measures the reference, not the models.
Primary scores keep every page as required; this row excludes that one page for
all models equally.

| System | CER micro (35 pages) |
| --- | ---: |
| `openai/gpt-4o-mini` | 56,18% |
| `qwen/qwen3-vl-235b-a22b-instruct` | 17,13% |
| `openai/gpt-5.4` | **13,46%** |

## Findings

1. **First measured external anchor on the frozen test: frontier VLM CER ≈ 17%.**
   `gpt-5.4` is about 2× better than the CPU baseline (16,87% vs 34,94%); the
   open-weight Qwen3-VL-235B follows at 20,48%. The frozen historical print set
   remains hard for everyone: no system is below 13% CER even excluding the
   broken page.
2. **`gpt-4o-mini` is worse than Tesseract here (58,11%)**, two failure modes
   visible in its predictions: it modernizes historical spelling and glyphs
   (`Ja rożne` → `Na różne`, `MȺDRYM` → `MĄDRYM`), and it refuses two pages with
   `"I'm sorry, but I can't assist with that."` (pages 433931, 433935), which
   score as full errors. Historical-spelling preservation in the metric is doing
   exactly its job.
3. **Reference defect confirmed on real predictions**, not just in annotation
   review: the errata page proves at least one frozen GT is truncated. This adds
   evidence to the pending reference review (U+FFFD/PUA characters already
   flagged in `docs/POLLOCR_SPLIT_POLICY.md`).
4. **Generation caps are a failure mode of their own**: Qwen3-VL hit
   `max_tokens=8192` on `NA2_FT__433928` (`finish_reason` length → error row,
   scored zero). Its reported 20,48% is therefore a lower bound.
5. Token accounting differs sharply by provider image handling (966k vs 108–161k
   prompt tokens for the same 36 pages); total cost of all three runs is below
   1 USD at published prices.

## Decision And Next

These are zero-shot/API-track measurements, not training and not a leaderboard
claim. They set the reference point for the project's own target: our recognizer
work has to beat ~17% CER (frontier zero-shot) and then ~13% (frontier, clean
pages) on the frozen test to be in SOTA territory for Polish historical print.

Still pending at the time of writing: the CPU Surya 2 run (in progress) and the
GPU Colab measurement (PaddleOCR-VL-1.6, Qwen3-VL-4B). Both land in separate
documents and will be compared against this table.

## Reproduce

```powershell
python -m training.run_vision_baseline --subtask A `
  --manifest data/impact-test-a-staged/manifest.jsonl `
  --output runs/api-a-testA/<name> --model <model-id> --max-tokens 8192 --execute
python -m training.transcription_eval --manifest data/impact-test-a-staged/manifest.jsonl `
  --predictions runs/api-a-testA/<name>/predictions.jsonl --output <name>-score.json
```

Limitations: single frozen subset of historical print; the metric keeps case,
diacritics and historical spelling; one reference is known-broken (primary score
retains it); refusals and truncations are scored as errors by protocol, not
filtered.
