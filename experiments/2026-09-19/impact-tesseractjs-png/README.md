# IMPACT historical-print CPU baseline, 2026-09-19

Tesseract.js 7.0.0 / core 7.0.0, pol+eng, OEM 1, PSM 3, one CPU worker.
This is a new baseline, not a reproduction of native Tesseract pol tessdata_best.

| Scope | Pages | CER micro | WER micro |
| --- | ---: | ---: | ---: |
| All collections | 36 | 34.9419% | 81.2336% |
| NA2_FT | 15 | 27.3493% | 73.2172% |
| Nowiny_z_Rakuz_FT | 15 | 43.9271% | 87.7816% |
| Powodzenia_FT | 6 | 34.8273% | 87.9365% |

All 36 predictions returned nonempty text; no execution errors or missing pages.
This does not imply correct transcription. Recognition took 392.10 seconds,
excluding 2.30 seconds of worker startup and image preparation. Hardware and
weight hashes are in `run.json`. Reference text was not passed to the recognizer.

The legacy NFC/whitespace scorer and the v1.1 quote-normalizing scorer produced
the same CER and WER. Approximate Markdown structure similarity was 0.568348.
The references were derived from PAGE text, not independently annotated Markdown;
this structural score is diagnostic, not a validated layout-quality benchmark.

## Inputs and reproducibility

- HF source: `PiotrSty/impact-print-v2`, revision
  `a2480fde6f15284701458ff370b81cce50dc5c2d`.
- Frozen corpus: 36 pages from three held-out collection IDs.
- Archive and original image hashes passed verification.
- Files named `.jpg` actually contain pyramidal TIFFs. Copies were re-encoded
  to PNG from frame zero with equal mode, dimensions and decoded pixel bytes.
- `conversion.json` links each PNG to its original and records pixel hashes.
- `input-manifest.jsonl` preserves the exact scoring manifest bytes. Its image
  paths resolve in the staged PNG directory, not in this evidence directory.
- Images and language weights remain in ignored local data/cache directories.
  Recreate images using `docs/POLOCRBENCH_REPRODUCTION.md` and score the saved
  `predictions.jsonl` against the staged PNG manifest.
- `checksums.sha256` covers the archived results and metadata.

Code base was `bb72058243570ca1603f35c69e1d1d9d589a5014` plus local evaluator,
staging, conversion and runner fixes on `codex/polocrbench-evaluator-correctness`.
The exact runner hash is in `run.json`; source hashes are in `code-hashes.json`.

## Limitations and next step

Frozen references contain 86 replacement characters (U+FFFD) on 23 pages and
624 private-use characters on 33 pages. No references were edited after viewing
predictions. Audit those annotations against images before using this subset as
a definitive leaderboard. Collection and exact-hash separation do not establish
absence of near-duplicates or pretraining contamination.

The old native Tesseract summary reports CER 33.11%, but lacks raw predictions
in the checked artifacts. Differences in languages, weights and implementations
prevent attributing the score difference to a single cause.

Next comparison should reuse these images and references with a separately pinned
backend and retain all page predictions. No SOTA claim is supported by this run.

## Source attribution

Reference transcriptions derive from IMPACT Polish ground truth, Poznan
Supercomputing and Networking Center (PSNC), via the IMPACT Centre of Competence,
CC BY 3.0: https://creativecommons.org/licenses/by/3.0/ .
Original source: https://github.com/impactcentre/groundtruth-pol .
Retain this attribution with reference-bearing artifacts.
