# Geometry holdout: verified result

The frozen 12-region development holdout does not support replacing the
rectangle geometry with the experimental line-band variant. This is a geometry
development result, not a benchmark score or evidence of SOTA.

## Verified execution

- Evidence ZIP SHA-256:
  `6f786d9f01b17b6c19c52e6003b75454b6518bae99f30a9c81b2032d70c84d21`.
- Frozen manifest SHA-256:
  `0b5459ea56d352de532b2cdc2284bd701b1c98458c67afd9f2220b6f0a6a7b18`.
- 12 regions from 12 collections, 67 reference lines and 3,486 normalized
  reference characters.
- Both models completed both variants for all 12 regions. Each variant detected
  68 lines; no failed, empty or apparently truncated outputs were recorded.
- The reported runner hash matches the source embedded in the published Colab
  notebook. Payload membership and every evidence checksum were independently
  verified by `training/audit_geometry_holdout.py`.

## Results

Lower is better. The primary normalization is NFC plus whitespace only; case,
historical spelling, `á`, `ſ` and all other source glyphs remain significant.

| Model | Rectangle CER | Line-band CER | Delta | Region wins / losses / ties |
| --- | ---: | ---: | ---: | ---: |
| `microsoft/trocr-base-printed` | 83.7349% | 83.5055% | -0.2295 pp | 6 / 5 / 1 |
| `PiotrSty/trocr-pl-mixed-v3` | **37.9231%** | 38.0092% | +0.0861 pp | 5 / 7 / 0 |

For mixed-v3, paired bootstrap resampling over regions gives a 95% percentile
interval of **[-1.2055, +1.3597] percentage points** for line-band minus
rectangle CER. The interval crosses zero, and line-band adds three character
edits overall. It improves WER (84.4444% to 83.0769%), but the primary CER
regresses slightly. Rectangle therefore remains the default geometry.

The Microsoft control has a small numerical line-band gain, but its 95% interval
also crosses zero: **[-0.9416, +0.5355] percentage points**. This is not a stable
geometry effect.

## Recognition error profile

The mixed-v3 rectangle output contains 1,322 character edits: 1,027
substitutions, 180 insertions and 115 deletions. Diagnostic categories are not
alternative scoring rules; every item remains an error in primary CER.

| Category | Edits |
| --- | ---: |
| Other glyph/recognition errors | 943 |
| Diacritic-related | 84 |
| Long `ſ` / `s` | 83 |
| Whitespace | 71 |
| Punctuation | 70 |
| Private-use reference character | 45 |
| Case only | 26 |

The most frequent reference characters involved in errors include `á` (118),
`e` (80), `d` (55), `ſ` (49), `n` (48), `i` (45) and `ɇ` (29). This supports
working on the historical-print recognizer and its alphabet rather than further
tuning this line-band geometry.

## Reference limitation

Eleven of the 12 regions contain 45 private-use Unicode characters. The clean
subset has only one region and must not be treated as a representative metric.
The upstream references are not manually adjudicated. A local scan-grounded
review workspace can be rebuilt with:

```powershell
python -m training.stage_holdout_reference_review `
  --manifest experiments/2026-09-24/geometry-holdout-v1/manifest.json `
  --output data/geometry-holdout-reference-review-v1
```

Review must preserve historical spelling. It may map a private-use code point
only after checking the printed glyph and must retain the original reference,
the proposed transcription and provenance. No scan or raw OCR prediction from
this holdout is published in the repository.

## Decision

1. Keep rectangle geometry as the default.
2. Stop tuning geometry on this holdout.
3. Adjudicate the 45 private-use reference characters against the scans.
4. Build a historical-character training inventory, prioritizing `á`, `ſ`,
   `ɇ`, ligatures and recurrent blackletter confusions.
5. Train a new recognizer on separate development/training data, then evaluate
   it once on a new document-disjoint validation set.

Machine-readable summaries are stored with the frozen experiment as
`result-summary.json` and `recognition-error-summary.json`.
