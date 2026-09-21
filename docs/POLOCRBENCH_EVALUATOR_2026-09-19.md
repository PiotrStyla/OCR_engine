# PolOCRBench evaluator verification

Base commit: `bb72058243570ca1603f35c69e1d1d9d589a5014`.

## Protocol v1.1

`training.transcription_eval` reports protocol version
`polocrbench-transcription-v1.1`. Structure similarity is a macro average over
every manifest page. Missing predictions and explicit errors score zero,
including on blank reference pages. Successful blank predictions on blank
reference pages still score one. Error responses cannot contribute their text.

Previously, failed pages were excluded from the structure average. A submission
with one perfect page and one missing page could score 1.0; it now scores 0.5.
Historical structure scores must be recomputed from predictions before comparison.
The frozen `training.benchmark_pages` evaluator and data manifests are unchanged.

JSONL records are separated by LF, not by arbitrary Unicode line separators.
Literal U+2028 and U+2029 inside JSON strings are accepted.

The existing Markdown skeleton remains an approximate metric: it treats each
nonempty source line as an element and does not implement a full Markdown parser.
This patch does not validate it as a final leaderboard metric.

## Frozen manifest inspection

- Test A: 36 records.
- Training pool: 2531 records (regions, not full pages).
- Both file hashes match `benchmarks/polocrbench/MANIFEST.sha256`.
- No overlap in collection identifiers derived from IDs before `__`.
- Images available in the fresh checkout: 0 of 36.
- Example image reference: `..\impact-corpus\pages\images\NA2_FT__433927.jpg`.

Collection ID separation alone does not prove absence of duplicate images or
near-duplicate documents. Pixel hashes and provenance need checking once the
source images are restored. The current image references also use Windows path
separators; portable staging needs verification before a Linux/Kaggle run.

## Verification

Eight tests passed in `tests/test_transcription_eval.py` and
`tests/test_benchmark_pages.py`, covering missing/error outputs, blank pages,
Unicode JSONL, normalization, disabling structure scoring, and existing fixtures.
No model inference or training was run. No full-corpus CER was recomputed.

Next: restore pinned benchmark images in a portable layout, verify image hashes
and split provenance, then reproduce existing baselines with the versioned scorer.
