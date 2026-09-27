# Historical layout development v1

Frozen development comparison on 15 public validation pages. Every detected
line was recognized once with `PiotrSty/trocr-pl-mixed-v3`; the three variants
changed only the permutation of the same decoded lines.

## Decision

`column_aware_v1` failed the frozen gates. Its overall CER and WER are exactly
equal to the current row-major order. The PAGE XML region-order diagnostic did
not improve either metric. Keep row-major as the default and move development
to detection and crop geometry without touching the frozen 36-page test.

The original verified archive is available as
[historical-layout-dev-v1-evidence.zip](historical-layout-dev-v1-evidence.zip).
See the [full result](../../../docs/HISTORICAL_LAYOUT_DEV_V1_RESULT_20260927.md),
[audit](result-audit.json) and extracted [evidence](evidence/).
