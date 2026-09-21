# Historical body-text development review

This step prepares annotation work, not another model-quality score. It uses
paragraph regions (including verse) from the pinned validation split of
PiotrSty/impact-psnc-polish-ocr, excluding the three previous heading-control pages.
Choose up to four regions per collection, round-robin across pages, in stable ID
order. Selection does not depend on predictions or detector success.

```bash
python -m training.prepare_body_dev_review --output data/body-dev-review-v1
```

Requires Pillow, numpy and OpenCV; no GPU or model weights. Original JPG crops,
source metadata and SHA-256 hashes are retained. PNG conversion preserves RGB
pixels; line crops use the original RGB image without deskew or binarization.
OpenCV normalization is used only to propose geometry, not alter saved pixels.
Proposals preserve the full region width and divide vertical gaps at their
midpoints; the first and last crop extend to the region edges. This avoids tight
ink-band clipping but can retain neighbouring fragments, requiring visual review.

## Review

1. Open `region-review/index.html` to inspect original paragraphs and references.
2. Open `line-review/index.html` to inspect proposed individual line crops.
3. Check the entire line is present, no adjacent text is included, and the
   reference really belongs to that crop. Preserve historical spelling.
4. Mark doubtful geometry or glyphs as requiring review. Do not silently replace
   private-use symbols or illegible text with a model prediction.
5. Export the JSON review history. Use independent reviewers/profiles before
   treating agreed labels as a candidate reference set.

The existing panel calls records pages; here each record is a region or line.
All labels begin unverified. The line/reference alignment is a count-only
proposal, not a certified correspondence. Models are not run or shown to reviewers.
Count mismatches and overlapping boxes are logged in `report.json`, with their
source regions still available for review; no reference strings are assigned to
these line proposals. Do not discard these failures from a future full-page score.

## Boundaries

`candidate-lines.jsonl` is NOT an evaluation or training release. Every record
has `eligible_for_evaluation: false`; the generic review tool does not turn that
flag on. Review/adjudication and a separate audited release step are required.
Do not feed this file straight into an evaluator that ignores those fields.

Frozen test collection IDs and exact crop hashes are excluded. Near-duplicates,
shared editions and upstream training overlap are not ruled out. In particular,
similar collection names must not be assumed to identify independent works.
These are historical paragraphs/verse, not a representative sample of modern
Polish documents. Check typeface coverage manually before expanding or training.

Raw images, draft references and review histories remain local. Only preparation
code, tests and methodology belong in the repository until publication is approved.

## Local preparation, 2026-09-21

The `data/body-dev-review-v2` run selected 19 paragraph regions across 10 pages
and five collection IDs, with 96 reference lines. Eleven regions produced 63
count-matched line proposals across four collection IDs. Eight regions had line
count mismatches, including every selected SLAWNA_VICTORIA_FT region. Therefore
the 63 proposals are not representative of the entire selected sample; manual
geometry work is needed before evaluating it as a whole.

The initial v1 run used tight detector boxes. Visual spot checks showed clipped
letter tops; v2 uses full-width, inter-band midpoint crops. The original v1
artifacts were retained, not overwritten. Two v2 crops were visually spot-checked;
neighbouring fragments and uncertain line boundaries still require human review.
No reference was corrected and no review decision was fabricated.

Validation: 25 focused tests passed; all 200 output checksums verified; all 63
saved line images matched the pixels of the corresponding region crop exactly.
Runtime: OpenCV 5.0.0, numpy 2.5.3. No model inference or training was performed.
The existing review assets were reused unchanged; browser UI validation of this
generated packet was blocked by the browser tool's local-file security policy.

## Transcriptions entered as notes

Only after the reviewer explicitly confirms that notes contain replacement
transcriptions, prepare a separate draft with:

```bash
python -m training.migrate_review_notes --manifest data/body-dev-review-v2/candidate-lines.jsonl --review reviewer-export.json --output data/body-dev-corrected-draft-v1 --notes-are-transcriptions
```

This preserves note text verbatim, the original manifest/export, prior decisions
and per-record provenance. Blank notes leave the latest actual text edit intact.
No verified events are synthesized: all new records remain drafts, ineligible
for evaluation. The new review panel starts a separate confirmation history;
old statuses remain in `source_review_decision` and `original-review.json`.

On 2026-09-21, the supplied export validated against all 63 records. Its 68 events
ended with 10 verified, 46 proposed and 7 needs-review UI decisions. The stricter
multi-reviewer audit found 9 single-review and 54 pending records, zero agreed.
After explicit user confirmation, 47 nonempty notes were promoted into the new
draft. These are not 47 independently approved ground-truth corrections. The
original history and all geometry concerns remain available for review.

## Diagnostic run without repeat review

The user explicitly chose to skip another review. Draft status is retained;
this does not create approved benchmark labels. Build a PRIVATE Kaggle notebook:

```bash
python -m training.build_private_body_notebook --manifest data/body-dev-corrected-draft-v1/manifest.jsonl --output data/body-dev-kaggle-v2
```

Import `kaggle_body_dev_PRIVATE.ipynb` from the output directory into Kaggle,
keep it private, enable GPU and Internet, and Run All. Images and corrected
draft text are embedded; no separate dataset upload is needed. Do NOT commit
this generated notebook to GitHub. The builder and runner contain no images.

Both pinned TrOCR models use the same 63 inputs. Report all-draft metrics and
the 56-line subset excluding the seven `needs-review` decisions separately.
The latter is still provisional, not an independently verified subset. Failed
predictions stay in denominators. Report case-sensitive NFC/whitespace CER/WER
and an additional lowercase CER, without historical spelling modernization.
The result ZIP contains predictions, draft references, environment and hashes,
not weights or reviewer names. No inference quality is claimed until Kaggle runs.
