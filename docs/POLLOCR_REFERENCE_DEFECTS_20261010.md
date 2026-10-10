# PolOCRBench test A: defekty referencji — audyt źródeł

## Question

How much of the measured CER on the frozen test A is model error and how much
is reference defect? This audit traces every known reference problem to its
source (upstream IMPACT ground truth versus our extraction) so the repair path
is evidence-based.

## Method

For each candidate defect the frozen manifest text was compared against the
**source PAGE XML** in `impactcentre/groundtruth-pol`
(`https://raw.githubusercontent.com/impactcentre/groundtruth-pol/master/{collection}/{0-padded doc}.xml`).
Counts are of `<Unicode>` elements; our freeze extracts them verbatim
(`training/polocrbench_freeze.py`), so identical counts mean identical
provenance.

## Finding 1 — errata page: upstream annotation is truncated

`NA2_FT__434735` has a frozen reference of **4 characters** (`'Tam‑'`). All six
measured systems transcribe 1,013–1,790 characters of what the scan shows to be
an errata table.

Source check: `NA2_FT/00434735.xml` (6,941 bytes) contains **exactly one
`<Unicode>` element with the same 4 characters**. Our extraction is faithful;
the **upstream annotation itself is truncated**. Therefore:

- this is not repairable from the source XML;
- per project policy the frozen reference is not repaired from model
  predictions, so the fix requires a **new human transcription** of the page
  through the two-reviewer consensus workflow (`tools/annotation-review`),
  or a documented exclusion of the page from the benchmark.

Score impact: retaining the page inflates every system's CER by a page-level
catastrophic term (CER 252–447% on that page); excluding it symmetrically is
already reported as the secondary score in the measurement docs.

## Finding 2 — U+FFFD and private-use characters: upstream glyph mapping

| Defect | Pages | Occurrences | Distinct |
| --- | ---: | ---: | ---: |
| U+FFFD replacement char | 23 | 86 | 1 |
| Private Use Area chars | 33 | 624 | **10** |

PUA inventory: U+EADA (416), U+EBA2 (113), U+F51E (51), U+EBA6 (16), U+EEC5
(13), U+EBA7 (8), U+E5DC (3), U+F516 (2), U+F50E (1), U+EBA3 (1).

Provenance: on 8 sampled pages the frozen counts equal the source XML counts
exactly (e.g. `NA2_FT__433928`: GT uffd=29/pua=15, XML uffd=29/pua=15). These
characters are **upstream annotation artifacts** — historical glyphs the IMPACT
annotators encoded in a private mapping — faithfully reproduced by our pipeline.

Score impact: 710 of 52,291 reference characters (624 PUA + 86 U+FFFD) are
characters a system reading historical print as modern Polish will almost never
match: **≥1,36% CER of built-in noise**, independent of model quality. This is
a lower bound; it affects every system equally.

## Decision gates (require a human decision, not a code fix)

1. **Errata page (`NA2_FT__434735`)**: commission a two-reviewer transcription
   (`tools/annotation-review` candidate workflow) *or* exclude the page with a
   documented reason. Until then, reports carry both primary (36 pages) and
   secondary (35 pages) scores.
2. **PUA mapping policy** (10 codepoints — small enough for a hand-built table):
   - keep as-is (current; noise floor stays ≥1,36% CER), or
   - publish a PUA→historical-glyph mapping table and apply it to references,
     with the metric documenting the mapping as a normalization step, or
   - map to modern equivalents and accept that the metric loses the
     historical-spelling distinction the project has preserved so far.
   The choice changes every system's score by roughly the same amount and must
   be applied symmetrically; it is a benchmark-policy decision.
3. **U+FFFD (86 chars)**: these are unrecoverable in the source annotation;
   they can only be transcribed anew from the scans (same two-reviewer path) or
   accepted as permanent noise.

## Recommendation

Gate 1 first (one page, high score leverage, cheap to review). Gate 2 second:
build the 10-row mapping table from the scans — with the historical-glyph
preservation rule kept, i.e. map PUA to the *actual historical glyph*, not to a
modern letter. Gate 3 last: fold into the same review pass as Gate 1.

No frozen reference is modified by this audit; it produces evidence and review
candidates only.
