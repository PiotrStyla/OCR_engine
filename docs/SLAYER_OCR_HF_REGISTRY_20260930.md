# SLAYER-OCR public Hugging Face registry

Created: 2026-09-30

Last verified update: 2026-10-06

Status: **uploaded, public, and manifest-verified**

## Scope

The registry preserves non-code artifacts recovered from the full fetched GitHub
history and the local OCR experiment workspace. Source code, notebooks, browser
review applications, dependency caches, executables, pytest work directories,
and temporary test fixtures are excluded. Historical OCR spelling is preserved.

Source repository: `https://github.com/PiotrStyla/OCR_engine`

Latest artifact source commit: `4ae4eb9f2d4d5dcf3f94d052d739602fd2989d27`

## Public repositories

| Artifact class | Repository | HF revision | Manifest SHA-256 | Manifest files | Bytes |
| --- | --- | --- | --- | ---: | ---: |
| Models | `PiotrSty/slayer-ocr-models` | `12132885beb68a579b77ffc5c7c9c5b7497476e1` | `98e8a2863308163e1410db234a565f2a0894b98dc2d0a9e081378bb5306e64c4` | 23 | 1,586,825,060 |
| Datasets | `PiotrSty/slayer-ocr-datasets` | `8fda75dd9a7612b8c64bedbade6a29bce579b739` | `7fd1778332ba5950e59a4b07785c10649979608b421c41b9bc9d4466ad2c3b7f` | 2,361 | 1,486,536,176 |
| Evidence | `PiotrSty/slayer-ocr-experiment-evidence` | `48af34860e6ed48a251b3cec3c459797775d93f8` | `ec397763cddf163dbf53be69b93b695feca2a1d3fccd63d00b4606fb18ad7032` | 1,124 | 1,438,242,821 |

Collection: `PiotrSty/ocr-experiment-registry-6abca642387af872378a7fee`

The collection also references the existing public
`PiotrSty/slayer-vision-onnx` model. Its audit evidence is retained in the
evidence repository instead of duplicating the ONNX payload.

## Verification

All three repositories and the collection were confirmed as public through
anonymous Hub API requests. Their
`MANIFEST.jsonl` and `MANIFEST.sha256` files were downloaded back from the exact
revisions above. Each downloaded manifest hash matched its local source exactly.

The 2026-10-02 incremental update adds the complete corrected RF-DETR dataset
v2, its deterministic ZIP, the corrected-GT audit ZIP, extracted predictions
and overlays, and the result report. Critical dataset provenance and audit JSON
files were downloaded from the exact revisions in the table and matched the
local sources byte-for-byte. The published archive hashes are:

- corrected dataset ZIP: `1cfb920217c15e1c2b4f0de50a972ca30bc6cbf975022dbfc9597fdbee83732d`;
- corrected-GT audit ZIP: `84efbe41837a42f94eea27c3176b6c6af78e837d7184d3356ceae311eaa87e51`.

The staging tree was scanned for executable/code extensions and common token
patterns before upload. No matching code files or credential patterns were
found. The registry is an open research archive. Upstream provenance and
license limitations remain attached to their source artifacts.

## Reviewed Recognizer Result: 2026-10-06

The table above preserves the earlier bulk-registry snapshots. The completed
reviewed-recognizer run is now appended under
`experiments/2026-10-06/recognizer-reviewed-colab-v1-result`:

- Models: `PiotrSty/slayer-ocr-models`, revision
  `43583c3932fb0cddd3e6a92333360f1d67ca56b3`: complete original result ZIP
  (912,673,777 bytes), SHA-256
  `7c97b6c99687863368c31bbd40822d9bd3d69493c2fd2c7c1ab02860426efe2c`.
  The pinned Hub LFS digest and size were verified; the full bundle was not
  downloaded again after upload.
- Evidence: `PiotrSty/slayer-ocr-experiment-evidence`, revision
  `43b749b453f563f9589ab3d6a418e72079cd3214`: original evidence ZIP, audit,
  paired predictions, trainer state, checksums and publication/code provenance.
  All smaller published payloads were downloaded and hash-checked.
- Candidate retained for research only: historical CER improves slightly,
  ordinary-print and combined CER regress. No production or SOTA promotion.

[Result details and downloads](RECOGNIZER_REVIEWED_COLAB_RESULT_20261006.md).

## Reproduction

`tools/prepare_hf_experiment_registry.py` rebuilds the three code-free staging
trees, records the fetched Git history, creates source and content manifests,
and preserves failed as well as successful experiment evidence.
