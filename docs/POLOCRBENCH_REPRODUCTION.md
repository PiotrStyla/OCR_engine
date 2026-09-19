# Reproduce the historical test subset

The frozen PolOCRBench manifests remain unchanged. Stage their images into a
separate directory with relative POSIX paths, usable on Windows and Linux.
Python 3.10+ is sufficient; staging does not need ML packages or a GPU.

## Download pinned inputs

Download `impact-print-v2-test.tar.gz` from:

https://huggingface.co/datasets/PiotrSty/impact-print-v2/blob/a2480fde6f15284701458ff370b81cce50dc5c2d/impact-print-v2-test.tar.gz

Save it under `data/impact-print-v2/impact-print-v2-test.tar.gz`.
The archive is 222506690 bytes. Expected SHA-256:

```text
0a9ffa126029703726fc5883a8279ddccf15763c4fdd483ed9b8cc034bdb42f0
```

## Stage and verify

From the repository root:

```bash
python -m training.stage_impact_benchmark --archive data/impact-print-v2/impact-print-v2-test.tar.gz --output data/polocrbench-history-v1
```

The command checks the archive hash, every selected image hash, collection
overlap and exact image-hash overlap with the training manifest. It writes only
explicitly selected regular image members, not archive-controlled paths or links.
It refuses to overwrite an existing output directory. Output:

- `images/`: 36 verified source files (TIFF bytes despite `.jpg` names).
- `manifest.jsonl`: original IDs, text, hashes and subset, with portable paths.
- `verification.json`: source revision and input/output hashes.

The frozen manifests are not rewritten. A changed staging manifest hash is
expected because image paths changed; each image hash and reference is retained.
Near-duplicate detection and correctness of the original annotations are separate
checks, not established by this command.

## Decode images for the CPU baseline

The published archive uses `.jpg` names for tiled, pyramidal TIFF images.
Tesseract.js cannot decode that TIFF encoding directly. Preserve those original
files and produce pixel-verified PNG derivatives with Pillow:

```bash
python -m training.prepare_ocr_images --manifest data/polocrbench-history-v1/manifest.jsonl --output data/polocrbench-history-png-v1
```

The converter selects frame zero and verifies that no later frame is larger.
It preserves image mode, dimensions and decoded pixel bytes, checks the PNG
roundtrip, and records original and derivative hashes in `conversion.json`.
It performs no resizing, thresholding or text modification. The PNG manifest
retains `source_sha256` for every original image.

Run the existing single-worker CPU baseline:

```bash
pnpm --dir tools/cpu-baseline install --frozen-lockfile --ignore-scripts
mkdir -p validation
node tools/cpu-baseline/run.mjs data/polocrbench-history-png-v1/manifest.jsonl validation/impact-tesseractjs-png
```

On PowerShell, create the parent with `New-Item -ItemType Directory -Force validation`.
This is Tesseract.js 7 with `pol+eng`, OEM 1, PSM 3 and its default language
weights. It is not a reproduction of the historical native `tessdata_best pol`
configuration. Images remain local; first startup downloads language weights.

## Score predictions

Supply actual model outputs as UTF-8 JSONL records with `id`, `status` (`ok` or
`error`) and `text`. Missing pages remain in the denominator. Install `jiwer`
in the evaluation environment, then run:

```bash
python -m training.transcription_eval --manifest data/polocrbench-history-v1/manifest.jsonl --predictions predictions.jsonl --output validation/transcription-v1.1.json
```

For comparison with the historical NFC/whitespace-only metric, use
`training.benchmark_pages` with the same arguments. Do not equate its scores
with the newer quote-normalizing protocol without recomputing predictions.

## Verification on 2026-09-19

All 36 page hashes matched the frozen references. The training pool contains
2531 region records. No collection ID or image hash overlaps were found.
Fourteen focused staging/scoring tests passed; no GPU or OCR inference ran.
Native Linux execution is not yet tested.

Frozen reference manifest SHA-256:
`382b4aba318e94722c3e1e47be324473b375fd02a6c41271f06c527e739fb962`

Training pool SHA-256:
`9ad700e949b3974c8c950a5e8617d14a7a35581a56878942d6e6feb05c51d97b`

Portable manifest SHA-256:
`b7ed83788ae887565bb2e305d760a869d120fbaa8546949ad4b4dbfe6d0db211`

The pinned HF repository contains aggregate Tesseract and Kraken result JSONs,
but no per-page prediction files; neither does the test archive. Those summary
scores cannot be independently recomputed from these assets. The next baseline
run must retain raw predictions, model/weight revisions, decoding settings,
dependency versions, input hashes, and page failures.

## Reference quality flag

Inspection of the frozen 36-page references found 86 U+FFFD replacement
characters across 23 pages and 624 private-use characters across 33 pages.
Private-use glyphs need a documented transcription convention; replacement
characters require inspection against the source. References are unchanged in
this baseline. These counts flag annotation work, not permission to silently
normalize or repair ground truth after viewing predictions.
