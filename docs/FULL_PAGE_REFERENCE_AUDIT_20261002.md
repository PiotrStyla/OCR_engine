# Full-page reference audit: 2026-10-02

This is a source-data audit, not an OCR inference result or a gold release.

Pinned source: `PiotrSty/impact-psnc-polish-ocr` at
`c7cb156fb95d2880699c33725bbaf1fbc1008fea`, CC-BY-3.0. The original source
URLs and image/PAGE XML hashes are preserved in each manifest record.

## Verified observations

| Observation | Value |
| --- | ---: |
| Non-test pages staged | 80 |
| Original train / validation | 65 / 15 |
| Collections | 27 |
| Duplicate source image hashes within this pool | 0 |
| Valid source region boxes containing annotation text | 532 |
| Source line boxes | 0 |
| Pages with U+FFFD | 38 |
| Pages with private-use Unicode | 71 |
| Pages flagged for either Unicode issue | 72 |
| Flagged Unicode occurrences | 1050 |
| Pages with incomplete content-region reading-order references | 31 |
| References shorter than 20 non-whitespace characters | 2 |
| Independently verified complete gold pages | 0 |

Manifest SHA-256:
`6fa9af89fede8472ced5d98ea36633f65d675fd95523ad2dc14169da8f8c47a3`.

Reference-free inference input SHA-256:
`09e7aeccf8a13e9fd49e216443b8568b913947e3a08c3d89c1246f8bcd76f31d`.

Every source image and XML passed its pinned hash. PNG frame-zero conversions
passed mode/dimension/pixel round-trip checks. The PNGs total 350,716,499 bytes.
The offline transcription editor was generated successfully for all 80 pages.
No human review decisions have been fabricated or imported into this new pool.

## Interpretation

Private-use glyphs can represent legitimate historical typography. They must
be resolved against the scan and an explicit glyph policy; they are not grounds
for automatic replacement, modernization or data exclusion. Likewise, absent
Unicode flags do not establish complete transcription.

The PAGE documents contain region geometry, not text-line geometry. Therefore
the new pilot's annotation-assisted diagnostic segments lines automatically
inside source regions. It does not invent line coordinates or claim perfect
oracle segmentation. Incomplete reading-order references remain visible in
the audit.

Exact duplicate checking here is only internal to the candidate pool. Near
duplicates and pretrained-model contamination are not ruled out. These pages
have prior development/training exposure and cannot establish final SOTA.

## Next action

Run the two-page `colab_full_page_pilot_v1.ipynb` GPU smoke and inspect raw model
outputs. Independently review complete transcriptions, add at least 20 pages
from additional sources, and freeze a new document-disjoint final test.
The contemporary Polish document track remains to be sourced separately.
