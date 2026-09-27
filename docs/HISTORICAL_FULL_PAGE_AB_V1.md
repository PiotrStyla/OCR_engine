# Historical full-page A/B v1

Status: frozen before notebook implementation and evaluation on 2026-09-27.

## Question

Does the accepted historical replay v2 line recognizer improve end-to-end
transcription of unseen historical pages when the detector, crops and reading
order are held fixed against `trocr-pl-mixed-v3`?

This is an evaluation of the current practical pipeline, not a SOTA claim.

## Inputs

- 36 frozen full pages from `PiotrSty/impact-print-v2` at
  `a2480fde6f15284701458ff370b81cce50dc5c2d`;
- baseline `PiotrSty/trocr-pl-mixed-v3` at
  `85d0c91c26f8e088849096dded7c9ba10b4cd9c9`;
- private candidate `PiotrSty/trocr-pl-historical-replay-v2` at
  `35d521d65fc59bf04bb7083322d1a6b7dc75da81`.

The test pages are collection-disjoint from the historical training and
validation material used for v2. No test text enters detection, cropping,
recognition or model selection.

## Fixed pipeline

Each page is deskewed once. `OpenCVDetector` produces one ordered list of line
boxes, and both recognizers receive exactly the same crops in the same order.
The output text is the whitespace-normalized concatenation of recognized
lines. No language-model correction or modernization is allowed.

The current reading-order heuristic is row-major and is not column-aware.
Consequently, this experiment measures the complete current pipeline and also
exposes its segmentation and ordering ceiling. A line-count comparison is only
a diagnostic; it is not detector precision or recall because line coordinates
are unavailable in the page references.

## Metrics and gates

The primary metric is full-page micro CER over all 36 pages. The report also
records WER, per-page and per-collection metrics, paired page wins, errors,
empty outputs and detected versus nonempty reference line counts.

The candidate passes this pipeline gate only if:

1. overall CER and WER both improve;
2. no collection CER regresses by more than 1 percentage point;
3. all 36 candidate pages complete successfully.

Evidence contains metrics, counts, environment versions, provenance and
checksums. It excludes page images, references, raw predictions and weights.
