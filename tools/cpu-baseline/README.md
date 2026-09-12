# Optional CPU baseline

Tesseract.js 7.0.0 wraps Tesseract in WASM. This is an independent baseline, not
the output of the Python CRAFT/TrOCR pipeline. No web server or GPU is required.
One worker recognizes the bundled page with PSM 3 and user-supplied line crops
with PSM 7; these tracks must be reported separately.

```powershell
cd tools/cpu-baseline
pnpm install --frozen-lockfile --ignore-scripts
cd ../..
node tools/cpu-baseline/run.mjs C:/path/to/pl_lines_sample validation/new-run
python tools/cpu-baseline/summarize.py validation/new-run
```

The output directory must not already exist; its parent must exist. Node.js and
Python must be available. The first run may download language weights; images
are processed locally. Outputs record predictions, source hashes, model hashes,
package lock hash, hardware, timing and segmentation mode. The language cache is
local and excluded from Git. For exact weight replay, retain or provide the
recorded cached files; a future upstream download is not an immutable pin.

The package postinstall only displays a funding message; install scripts are
disabled. All measurements are from a tiny synthetic fixture, not a production
quality or speed benchmark. No train/test independence is claimed for the lines.

Upstream: https://github.com/naptha/tesseract.js
