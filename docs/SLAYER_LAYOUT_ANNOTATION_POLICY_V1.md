# SLAYER-OCR layout annotation policy v1

Status: proposed frozen policy for the 12-page RF-DETR development review

This policy resolves the ambiguities found by the private best-checkpoint audit.
It applies to layout geometry only. It does not alter OCR text, historical
spelling or transcription provenance.

## Classes and geometry

- `text_region`: one coherent paragraph or block in the main reading flow. Do
  not split one block line by line. Split only where columns, whitespace,
  typography or reading order create genuinely separate blocks.
- `heading`: a title or section heading functioning as a heading. Draw a tight
  box and exclude adjacent body text.
- `table`: the complete tabular structure. Do not add an overlapping
  `text_region` for the same table content. Cells are not annotated separately
  in v1.
- `figure`: a standalone illustration or ornament. A decorative initial
  embedded in a paragraph remains inside `text_region`, not a separate figure.
- `caption`: text semantically attached to a figure and outside its figure box.
- `marginalia`: semantic text outside the main reading flow. Random marks,
  handwritten shelf marks without transcription intent and scan artifacts are
  not marginalia.
- `header` and `footer`: repeated running elements, not arbitrary first or last
  lines of body text.
- `page_number`: a tight box around the printed page number only.

## Ignore policy

Do not annotate library stamps, fingers, page frames, shadows, tears, stains,
bleed-through or other acquisition artifacts. They are neither `figure` nor
`marginalia` unless a later, separately versioned policy introduces an artifact
class.

## Overlap policy

Classes are exclusive for v1. Avoid boxes describing the same pixels as two
different semantic objects. Incidental geometric overlap is acceptable only
when distinct objects physically overlap in the scan. Record uncertain cases as
`needs-review`; do not guess.

## Review rule

Every development page must be marked `verified` when the frozen GT already
matches this policy, `edited` when the exported annotations were changed, or
`needs-review` when uncertainty remains. A final candidate may contain only
`verified` and `edited` pages. The source COCO file is never overwritten.
