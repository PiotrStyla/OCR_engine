# Recognizer expansion: explicit visual confirmation

## Human Confirmation

The user answered `tak` on 2026-10-06 to the question asking whether the corrected
texts of the 31 proposed complete-line entries had also been checked visually.
This refers only to those 31 entries. It does not approve rejected crops,
unchecked geometry, missing decisions, training eligibility or a SOTA claim.

The new `visual-confirmation.json` records the exact question/answer and binds
every approved entry to its latest event ID, crop hash, context hash and exact
UTF-8 transcription hash. It also binds the raw export, source manifest and
complete raw import checksum manifest. No reviewer name or timestamp is invented.
This is follow-up by the same user, not a second independent reviewer.

Confirmation SHA-256:
`943dbcf3e335db2cb16297610c1bc278fe748282330c1ec115ef3688cd470caf`.

## Result

- 51 accepted single-human annotations: 20 original verified entries plus 31
  explicitly confirmed proposals.
- 13 follow-up entries: nine rejected crops, two unchecked geometries and two
  lines without a decision.
- 17 accepted lines are quarantined by the protective cross-split work families.
- 34 accepted lines still have unresolved bibliographic independence.
- Zero training/evaluation-eligible rows, zero gold labels, no training run.

Original `review_status`, `import_status`, reviewer fields, text, notes and all
71 raw review events are unchanged. The new effective `annotation_status` and
`annotation_verification_source` describe the separate confirmation. Raw
`source_statuses` therefore intentionally retain the original proposal counts.
Historical glyphs, combining marks, punctuation and spelling are not normalized.

The top-level manifest hash is
`9dc013724d56ebf13460d5b66bf052103738aa4ec3adcba8043f5ac7a5fa3e41`.
The old v1 public package and earlier 64-pair pool remain immutable. This is not
a claim that 115 annotations are independently validated training data.

## Validation And Contents

Both fresh builds replay the original import and reproduce all output file bytes.
The focused CPU regression suite passes 295 tests, including 39 gate tests.
The gate rejects missing/extra/duplicate confirmations, stale events, changed
crop/context/text hashes, non-user evidence, rejected/unchecked geometry and
unsafe or multiline text. Training gates remain closed after confirmation.

The complete code-free ZIP contains all 64 crops and 34 source contexts, the
original weak labels/teacher diagnostics, raw import history, the separate
confirmation, the 51-entry gated manifest, all 64 ledger rows, the 13-entry
follow-up queue, bound work-family policy and checksums. Images retain the
IMPACT/PSNC attribution and CC-BY-3.0 source license at dataset revision
`c7cb156fb95d2880699c33725bbaf1fbc1008fea`.

No new Colab inference or full review is required from the user for this version.
The next research work is bibliographic split independence and crop remediation,
not another blanket request to inspect the same 64 lines. No further review
task is requested in this handoff.
