# Validation v4: single-review reference draft v1

Date: 2026-10-03. This is a reference-data intervention, not a new OCR run.

The supplied `polocrbench-review-e35f3ef6037a.json` passed validation against the
exact 15-page review manifest, image hashes, original text hashes, before/after
history, timestamps and reviewer identity. Export SHA-256:
`5b00ead5cd57866b7c56f46fabdfed8d96dc169c12181d6200250c45ffaf0331`.
Source manifest SHA-256:
`e35f3ef6037a53015d9ffb09d87c3be558f9a45cab2521d7d3bb3e7be170c2fe`.

## Imported Decisions

15 events cover all 15 validation pages from one reviewer. Fourteen decisions are
`proposed` and one is `verified`; the verified page has unchanged text. Fourteen
page texts changed. All decisions, original source texts and notes were retained.
No approval event was synthesized, and no reviewer was invented.

In the proposed/current draft transcriptions, private-use/replacement-character
issues decrease from 117 to zero. The original manifest and unchanged source
PAGE XML still preserve the old encoded references. Notes are not promoted into
transcriptions. Historical `á`, `ſ` and other source spelling are not modernized.

The draft has **zero adjudicated gold pages**. One verified single-review page is
not independent adjudication. Formal benchmark/training eligibility and page
completeness remain false. The existing two-review adjudicator is unchanged.

## Ligature Findings

Character-level edit alignment finds repeated proposals for `U+EADA -> ſt`
(55 observations), `U+EBA2 -> ſi` (22) and `U+EBA6 -> ſſ` (2).
These are observed edits with page/event provenance, not a universal font mapping.
Reordered text or contextual edits can affect alignment; no replacements are
automatically applied to other pages or datasets.

`U+F51E` has mixed proposed expansions: `ſł` (12), `ſl` (1), `ſt` (1), `łt` (1).
Two words on `Choragiew_FT__436799` deserve a targeted visual recheck:
the imported draft has `Poſtał` and `Połtał`, while the scan appears to read
`Poſłał` at both positions. This is an assistant diagnostic/proposal, not an
approved user correction. The imported draft remains verbatim pending confirmation.

The notes also contain marginalia and catchword/signature comments. These remain
separate annotation evidence. Absence of invalid Unicode does not establish that
the full-page inclusion policy, all marginal text or reading order is adjudicated.

## Same-Prediction Diagnostic

| Reference Version | All-Page CER | All-Page WER |
| --- | --- | --- |
| Frozen source reference | 243.7391% | 515.8644% |
| Single-review draft | 237.2065% | 514.0136% |

Exactly the same 15 saved projected predictions were used, with unchanged raw
output retained in their `raw_text` fields. There was no GPU inference, training,
repeat removal or model change. Four capped outputs remain; EOS still ends 11/15.
The changed scores measure a draft-reference intervention, including possible
content-scope changes, not an improvement of OvisOCR2. The original v4 score and
evidence remain immutable. No deployment, teacher-label or SOTA promotion.

## Artifacts and Next Gate

`training.prepare_single_review_draft` creates a versioned candidate, complete
image/XML ZIP, original review and manifest, glyph proposals, comparison and hashes.
Draft manifest SHA-256:
`54460fccdc0f082b55c59cf9acde1ab06f7464fb2aa34d63fc9f3576e04bb6e3`.
Complete ZIP SHA-256:
`6640f7b24971e8a9e5cc44392090e6a204cbeeb22d2bd4f09a885c6ac9965a95`.
The ZIP contains all 15 scans, all 15 unchanged source XML files and the complete
review evidence. No program files or weights. Preparation and existing safeguards
passed **90 focused CPU tests**.

Next: resolve the two word proposals and inclusion-policy notes, then compare
another supported full-page backend using these explicitly provisional references.
Keep a separate independent, reviewed test before selecting training targets or
claiming SOTA. The existing validation pages must not be repurposed as training data.
