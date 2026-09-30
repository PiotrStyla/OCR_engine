# SLAYER-OCR private Hugging Face registry

Date: 2026-09-30

Status: **uploaded, private, and manifest-verified**

## Scope

The registry preserves non-code artifacts recovered from the full fetched GitHub
history and the local OCR experiment workspace. Source code, notebooks, browser
review applications, dependency caches, executables, pytest work directories,
and temporary test fixtures are excluded. Historical OCR spelling is preserved.

Source repository: `https://github.com/PiotrStyla/OCR_engine`

Frozen artifact source commit: `4ce2215d45814d2fd1e9749c394b95fde6a240af`

## Private repositories

| Artifact class | Repository | HF revision | Manifest SHA-256 | Manifest files | Bytes |
| --- | --- | --- | --- | ---: | ---: |
| Models | `PiotrSty/slayer-ocr-private-models` | `c425d7ccf44a4adfb8113c5584ffbf940437a173` | `4e088256a4cbfb74de084227c3bd64a420962b73b0305bf1ac0d4b4787fce9eb` | 23 | 1,586,825,086 |
| Datasets | `PiotrSty/slayer-ocr-private-datasets` | `e7c902d5e942780f7491896664eb84d4b38b3398` | `1da737a49ee409812f2e2498d6079257d10eaec55ca9f6cab2254cadbaae2b48` | 2,292 | 1,266,214,923 |
| Evidence | `PiotrSty/slayer-ocr-experiment-evidence` | `055133d776209b0841a95f638db60eef89ffe1d9` | `5f13f23fe0e72b77f5101255e35c73e8116285b282ca9fc8a9d56d549f66494f` | 1,103 | 1,426,377,679 |

Collection: `PiotrSty/slayer-ocr-experiment-registry-6abca642387af872378a7fee`

The collection also references the existing public
`PiotrSty/slayer-vision-onnx` model. Its audit evidence is retained in the
private evidence repository instead of duplicating the ONNX payload.

## Verification

All three repositories were confirmed as private through the Hub API. Their
`MANIFEST.jsonl` and `MANIFEST.sha256` files were downloaded back from the exact
revisions above. Each downloaded manifest hash matched its local source exactly.

The staging tree was scanned for executable/code extensions and common token
patterns before upload. No matching code files or credential patterns were
found. The registry is a private research archive, not a statement that every
source artifact is cleared for public redistribution.

## Reproduction

`tools/prepare_hf_experiment_registry.py` rebuilds the three code-free staging
trees, records the fetched Git history, creates source and content manifests,
and preserves failed as well as successful experiment evidence.
