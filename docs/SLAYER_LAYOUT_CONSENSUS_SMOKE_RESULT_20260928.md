# SLAYER layout consensus: two-page smoke result

Date: 2026-09-28

## Scope and integrity

Three pinned teacher runs completed on the same two training pages with zero
reported page errors. The combined evidence ZIP has SHA-256
`219ef28f2d8a7e23fc7788cbe7cbd3e62cc3ded25075057556bec9aa885ad79b`.
All embedded teacher and consensus checksums were verified before review.

The evidence contains proposals, manifests and checksums, but no images or
reference transcriptions. Images and the interactive review output remain local;
this report does not authorize publication of either.

| Teacher | Evidence SHA-256 | Pages | Errors |
| --- | --- | ---: | ---: |
| DocLayout-YOLO | `35c3c355a7f214db12237362d3aa707211f93c9b2ae1e1d6a2d6b6ddea8e9589` | 2 | 0 |
| Qwen3-VL 4B, canonical-label repair | `b957594304741b922fb1b845489ed7bcf4bc4ef25fd4025f39cbb82ef1f04467` | 2 | 0 |
| Surya Layout | `3aeef82a23f054bbf2e0262e39fcab578d06e500fdb09d415063b650bb0cc4c8` | 2 | 0 |

## Machine consensus

- teacher rows: 6;
- teacher detections: 18;
- accepted objects: 5;
- review objects: 7;
- hard-example pages: 2 of 2.

The five accepted objects are visually plausible: one page-level text region,
one stamp/figure and three body-text regions. This is a smoke observation, not an
accuracy estimate.

## Visual review findings

The review queue exposed three useful corrections and two systematic issues:

- a central ornament should be `figure`, not `text_region`;
- the title on the second page should remain `heading`;
- the lower body block on the second page is a valid `text_region` missed by the
  second detector;
- Qwen's `page_number` proposal points to an ornament and should be rejected;
- Qwen's single body-sized `text_region` conflicts in granularity with the
  paragraph/block regions returned by the detector teachers and should not add
  votes to multiple smaller objects.

The source text was not transcribed or normalized during this review. Historical
spelling remains outside the layout labeling operation and is unchanged.

## Decision

The execution and provenance smoke passes. The data-quality gate does not yet
pass for a 60-page run. Before scaling:

1. freeze a short visual definition for `heading`, `figure` and `text_region`;
2. detect cross-scale containment as `granularity-conflict` without duplicating
   one teacher vote across child boxes;
3. preserve majority boxes but keep every page with a cross-scale conflict in
   the human-review queue;
4. rerun the same two pages and require zero unexplained accepted objects.

Only after that repeated smoke should the three 60-page teacher runs begin.

## Policy v2 rerun

The same 18 detections were recombined without new inference using the frozen
`slayer-layout-consensus-policy-v2.json`. Counts intentionally remain unchanged:
5 accepted objects, 7 review objects and 2 hard pages. Three review objects now
carry the additional `granularity-conflict` reason. Majority-supported smaller
regions remain accepted; a weaker enclosing region cannot lend its vote to each
child. Equal-vote cross-scale clusters both abstain.

All five accepted objects remain visually plausible under the frozen
`slayer-layout-ontology-v2.json`. The remaining queue still requires explicit,
versioned adjudication before it can become a clean training dataset.

The returned v2 evidence ZIP has SHA-256
`d9c04ea2e893aed7b63364f8f29337f7298e9e7937e568aaf8436816aa7b6138`, pins
code revision `f851c6fc38efef3a222616c7b5d04cc37bc0e120`, and contains the expected
three `granularity-conflict` records. Its policy hash and all embedded teacher
and consensus checksums were verified locally.
