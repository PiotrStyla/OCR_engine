# Historical full-page A/B v1

Frozen 36-page comparison of `PiotrSty/trocr-pl-mixed-v3` and private
`PiotrSty/trocr-pl-historical-replay-v2`. Both recognizers received the same
OpenCV detections in the same order. The evidence contains no scans, reference
texts, raw predictions, weights or token.

## Decision

The candidate failed the frozen promotion gates. It improved micro WER from
97.8318% to 97.1838%, but worsened the primary micro CER from 56.1126% to
56.7479%. It also exceeded the 1 pp collection CER regression limit on
`NA2_FT` and `Powodzenia_FT`.

Keep `mixed-v3` as the default full-page recognizer. Retain v2 as a private
research artifact; do not claim SOTA or tune against this frozen test.

See the [full result](../../../docs/HISTORICAL_FULL_PAGE_AB_V1_RESULT_20260927.md)
and the exact extracted [evidence](evidence/). The original verified archive is
available as [historical-full-page-ab-v1-evidence.zip](historical-full-page-ab-v1-evidence.zip).
