# SLAYER-OCR public Hugging Face registry

Date: 2026-09-30

Status: **uploaded, public, and manifest-verified**

## Scope

The registry preserves non-code artifacts recovered from the full fetched GitHub
history and the local OCR experiment workspace. Source code, notebooks, browser
review applications, dependency caches, executables, pytest work directories,
and temporary test fixtures are excluded. Historical OCR spelling is preserved.

Source repository: `https://github.com/PiotrStyla/OCR_engine`

Frozen artifact source commit: `4ce2215d45814d2fd1e9749c394b95fde6a240af`

## Public repositories

| Artifact class | Repository | HF revision | Manifest SHA-256 | Manifest files | Bytes |
| --- | --- | --- | --- | ---: | ---: |
| Models | `PiotrSty/slayer-ocr-models` | `12132885beb68a579b77ffc5c7c9c5b7497476e1` | `98e8a2863308163e1410db234a565f2a0894b98dc2d0a9e081378bb5306e64c4` | 23 | 1,586,825,060 |
| Datasets | `PiotrSty/slayer-ocr-datasets` | `09a4978402b0508795a16a8cc0350b5a7ee1e024` | `eb8c6f2661d546b47b6b7cc9644304707935bccadf2653b7c6795ded1c4e4e29` | 2,292 | 1,266,214,905 |
| Evidence | `PiotrSty/slayer-ocr-experiment-evidence` | `e99e06045824daa577d7d80104663b416420a299` | `0ca2aeb4d62a3653db3381c3131c0be476c37bd49681daa2f9eb9fbc1073d097` | 1,104 | 1,426,381,205 |

Collection: `PiotrSty/slayer-ocr-experiment-registry-6abca642387af872378a7fee`

The collection also references the existing public
`PiotrSty/slayer-vision-onnx` model. Its audit evidence is retained in the
evidence repository instead of duplicating the ONNX payload.

## Verification

All three repositories and the collection were confirmed as public through
anonymous Hub API requests. Their
`MANIFEST.jsonl` and `MANIFEST.sha256` files were downloaded back from the exact
revisions above. Each downloaded manifest hash matched its local source exactly.

The staging tree was scanned for executable/code extensions and common token
patterns before upload. No matching code files or credential patterns were
found. The registry is an open research archive. Upstream provenance and
license limitations remain attached to their source artifacts.

## Reproduction

`tools/prepare_hf_experiment_registry.py` rebuilds the three code-free staging
trees, records the fetched Git history, creates source and content manifests,
and preserves failed as well as successful experiment evidence.
