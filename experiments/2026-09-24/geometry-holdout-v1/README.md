# Geometry holdout v1

Frozen before OCR on 2026-09-24. This is a development holdout for comparing
two image-only line geometries, not a model benchmark or proof of SOTA.

- Source: `PiotrSty/impact-psnc-polish-ocr`, pinned revision
  `c7cb156fb95d2880699c33725bbaf1fbc1008fea`.
- Split: upstream `train`; 12 regions from 12 collections not used in the prior
  geometry work and disjoint from the frozen test collections.
- Selection: deterministic hash rank, one paragraph region per collection,
  3-12 nonblank reference lines, no U+FFFD replacement character.
- Historical spelling and private-use glyphs remain unchanged. Their counts are
  recorded per region and a no-private-use subset will be reported separately.
- References are upstream and unreviewed. The model card says mixed-v3 used
  synthetic Polish plus EHRI lines, but no independent leakage audit was done.

The 12 regions contain 67 reference lines. A local pre-OCR smoke check found 68
lines with both rectangle and line-band variants: 11 regions matched the
reference line count and one detected seven instead of six. Every region remains
in the denominator; reference line counts are not used by segmentation.

No source images or OCR predictions are stored here. The Colab notebook will
download the 12 CC-BY-3.0 region images directly from the pinned Hugging Face
revision and verify each SHA256. The result ZIP stays with the person running it.

Files:

- `manifest.json`: frozen selection, references, source paths and hashes.
- `exclusions.json`: deterministic exclusion reasons before OCR.
- `checksums.json`: hashes for both JSON artifacts.

Primary normalization is NFC plus whitespace only. Case and historical spelling
are retained. This sample may guide a geometry decision, but the final frozen
PolOCRBench test remains untouched.

## Completed result

The Colab run completed all four model/geometry combinations on 2026-09-24.
For mixed-v3, rectangle CER was 37.9231% and line-band CER was 38.0092%.
The paired region-bootstrap 95% interval for the difference crossed zero, so
rectangle remains the default and this holdout is closed for geometry tuning.

- `result-summary.json`: independently recomputed metrics and provenance.
- `recognition-error-summary.json`: aggregate mixed-v3 rectangle error profile;
  raw OCR text is excluded.
- [Full result and decision](../../../docs/GEOMETRY_HOLDOUT_RESULT_20260924.md).
