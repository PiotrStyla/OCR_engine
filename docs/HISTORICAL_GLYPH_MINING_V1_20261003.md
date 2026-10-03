# Historical glyph mining v1

The independently audited v7 run now has a CPU-only, reproducible diagnostic
queue for historical `ſ`, `á`, `ɇ`. No normalization, spelling correction,
reference replacement or GPU inference is performed.

## Output

- 15 validation pages retained; 14 have targeted mismatches.
- 479 candidate edit spans, including 16 in contexts containing `Poſłał`.
- Original draft manifest, raw v5/v7 predictions and scans preserved by hash.
- A review manifest, complete JSONL queue, alignment diagnostics and provenance.
- Offline review shows the scan, draft, both raw OCR outputs and clickable
  mismatch contexts. Source offsets are UTF-16 for browser selections.
- Zero training examples and zero gold pages created.

RapidFuzz 3.14.6 Levenshtein opcodes propose one minimum-edit alignment.
Repeated text, different reading order and draft-reference errors can change
the alignment. These 479 spans are not 479 proven glyph errors and are not
recall. No word/line bounding boxes are inferred from text alignment.
The user's prior `Poſłał` confirmations remain unchanged; this queue does not
invalidate them or require the same word to be re-confirmed.

## Reproduce

```powershell
python -m training.mine_historical_ocr_errors --audit <verified-v7-audit-dir> --output <new-mining-dir>
```

To rebuild the review UI from the data-only HF bundle and the GH code:

```powershell
python -m training.build_annotation_review --manifest <mining-dir>/review-manifest.jsonl --diagnostics <mining-dir>/diagnostics.json --output <new-review-dir>
```

Sources must match the unchanged v7 ZIP; modified manifest/predictions/images,
duplicate/missing predictions and training/final-test page scope are rejected.
Existing output directories are not overwritten.

## Review Decisions

The panel shows draft text on the left of each arrow and raw v7 OCR on the
right. Neither is automatically true. Check the scan before proposing a change.
Keep historical spelling; do not replace `ſ` by `s`, `á` by `a`, or expand
an ambiguous abbreviation silently. Click a difference to select its source
span in the draft. After editing, source offsets are stale and selection is
blocked until the original draft is restored.

For a correction limited to the highlighted glyphs, choose **Propozycja korekty**
and record the scope in the note. **Zweryfikowano wzrokowo** means the entire
page text/content scope/order was checked, not just one highlighted word.
Uncertain readings remain **Do wyjaśnienia**. Export the JSON history after
saving decisions. Raw OCR remains a separate immutable variant.

## Split Firewall And Next Training Gate

These pages are validation diagnostics, never recognizer training examples.
Review here can improve the reference and describe error families, but it
must not supply crops/text to training or become an untouched final test.
Training data must come from a separate source pool. Before a GPU notebook:

1. Inventory collection/work/page identities and exact image hashes against
   validation plus the frozen historical test; audit near duplicates separately.
2. Select training-only examples of the same glyph families without taking
   content from this queue. Preserve raw XML/transcriptions and source licenses.
3. Verify line crops, diplomatic text, reading order and annotator provenance;
   quarantine uncertain/unmatched crops rather than generate labels by count.
4. Freeze a document-disjoint development protocol and a separate untouched
   final test before training; record CER/WER and glyph diagnostics without
   historical-letter modernization.

Only after this gate should teacher proposals enter human-reviewed clean
training data for a specialized recognizer. No new GPU run is required to
inspect the current queue; no SOTA or automatic-teacher claim is made.

## Verification

The related regression suite passes 156 CPU tests; a focused 16-test rerun
also verifies unchanged original image paths/hashes. Browser plugin was not
available; local Playwright used installed Edge, with no browser installation.
At 1440x1000 and 390x844 the scan loads, content is nonblank, no page errors
occur and mobile has no horizontal overflow. Clicking an edit selects the
corresponding draft span; the flagged filter returns 14 pages. Text mutation
blocks stale offsets. A temporary proposed decision updates progress and
exports a valid-named JSON history; that browser-test storage was cleared.
No human review was performed by these tests.
