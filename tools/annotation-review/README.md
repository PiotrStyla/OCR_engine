# Offline annotation review

Generate a static review workspace from a verified PNG manifest:

```bash
python -m training.build_annotation_review --manifest data/polocrbench-history-png-v1/manifest.jsonl --output data/annotation-review-v1
```

Pillow is required only for workspace generation. Open the generated `index.html`
in a current Edge or Chrome browser. No server, model, API or network request is
required. Source images are copied into the workspace, and manifest hashes are
verified before generation. Existing output directories are never overwritten.

The workspace shows each scan beside an editable reference, flags U+FFFD and
Unicode private-use characters, and filters pages by ID, source flags or saved
decisions. Clicking a flag selects its position in the editor. Original text
remains available separately. Private-use glyphs may be intentional conventions;
a flag is a review request, not an instruction to remove or modernize a character.

## Review records

Every saved decision appends a record with page ID, reviewer, time, decision,
note, original text hash, image hash, previous text and proposed text. Saving a
restoration also appends a record. Decisions are self-reported annotations, not
independent certification or an authenticated signature.

Browser localStorage keeps saved decisions under the manifest hash. Its behavior
for file URLs depends on browser/profile and location. Unsaved editor changes are
not persisted. Export the JSON history before closing or moving the workspace;
localStorage is not an archival backup. Storage failures produce a visible warning.

Exports use schema `polocrbench-review-patch-v1`. They never overwrite the input
manifest or constitute a new benchmark release. Imports verify the manifest,
page/image/original-text identity, unique event IDs, field types, and text-change
continuity. Import may extend an identical history prefix; divergent histories
are rejected rather than silently merged. Independent reviewers should export
separate files for later adjudication with the command below.

The tool does not show model predictions, so reviewers are not prompted to
match baseline outputs. No automatic transcription correction is performed.

## Verification

Generator tests cover hash mismatch, immutable inputs, safe embedding of text
containing HTML/script tags, and UTF-16 offsets for supplementary characters.
Browser QA exercised a real 36-page workspace in Edge with Playwright:
navigation, issue selection, search/filtering, save/reload, export/import,
rejection of a foreign manifest, and append-only restoration history.
Desktop 1440x1000 and mobile 390x844 had no JS errors or horizontal overflow.
These checks validate the tool, not the transcription accuracy of any page.

## Build a reviewed candidate

Two reviewers should independently inspect the same original review workspace
and export their decisions. Use separate browser profiles or computers; sharing
one profile shares localStorage. Do not import the first review into the second
reviewer's workspace before independent review.

```bash
python -m training.adjudicate_reviews --manifest data/polocrbench-history-png-v1/manifest.jsonl --reviews reviewer-a.json reviewer-b.json --output data/polocrbench-history-candidate-v2
```

Only pages with at least two distinct reviewer names and identical latest
`verified` texts are applied. All latest reviewer decisions for that page must
be verified; a pending decision or differing text prevents application. Names
are normalized with NFC, case folding and whitespace normalization so aliases
such as `Anna` and ` ANNA ` do not count twice. Names are self-reported, not
authenticated identities or proof of independent review.

The command validates manifest/image/original-text identity and the complete
text-edit chain in each export. Duplicate exports are deduplicated by event ID.
A later withdrawal supersedes approval, backdated events within a reviewer's
page history are rejected, and conflicting records sharing an ID are rejected.
Different decisions at an identical latest timestamp are reported as ambiguous.

Output is a **candidate**, never an automatic publication:

- `manifest.jsonl` and `images/`: portable dataset; only agreed texts change.
- `adjudication.json`: source/output hashes, per-page decisions, change counts,
  remaining suspicious-character counts and review-input hashes.
- `review-evidence.json`: deduplicated full event records, including before/after.
- `checksums.sha256`: checksums of the manifest, report and evidence. Image hashes
  are retained in the manifest.

Pages marked `unreviewed`, `pending`, `single-review`, `conflict` or
`ambiguous-reviewer-history` retain their original text. `agreed` does not imply
that all suspicious glyphs are resolved. Review glyph conventions and outstanding
flags before releasing a benchmark. Existing output directories are not overwritten.
Corrected references define a new benchmark version; recompute baseline scores
against that exact manifest and do not compare them as if the ground truth were
unchanged.

A run without `--reviews` is a preflight: every page remains unreviewed. On the
36-page historical set this preserved all texts and reported 710 flags, zero
agreed pages and zero changes. The combined focused test suite has 36 passing
tests, including agreement, conflict, duplicate names/exports, withdrawn approval,
invalid provenance and immutable source checks. No human reviews were fabricated.
