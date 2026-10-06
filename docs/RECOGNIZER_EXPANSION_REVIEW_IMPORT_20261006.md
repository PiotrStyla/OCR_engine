# Recognizer expansion: imported review with a closed training gate

**Historical v1 snapshot:** the user subsequently confirmed visual review of the
31 proposed complete-line texts. See the
[new v2 confirmation report](RECOGNIZER_EXPANSION_VISUAL_CONFIRMATION_20261006.md).
The v1 artifacts below remain immutable; the pending confirmation described
here is resolved in v2, not by rewriting the original export.

## Exact Export, Not Automatic Approval

Imported `slayer-recognizer-line-review-31debfc142b5.json`:

- Export SHA-256: `07b527f7a0886142960fd05f617983cbe7b6f8eb360eafcfbd82914869baaf75`.
- Bound review manifest: `31debfc142b523bf7786b42ee9d53c8fcd3218cf09fe99fa97967e448fb4627c`.
- 71 events covering 62 of the 64 new line crops.
- Latest decisions: **20 verified complete-line annotations**, **9 rejected crops**,
  **31 proposed complete-line transcriptions**, **2 proposals without checked
  geometry**, and **2 lines without any decision**.
- 41 latest transcriptions differ from their source label, across all statuses.

The latest event wins; an earlier verified event does not override a later
`proposed` correction. A modified text does not automatically become verified.
Rejected crops remain rejected even when their text was marked verified.
The 31 complete-line proposals remain proposals until explicit human confirmation.
No new confirmation, reviewer identity or annotation event is invented.

All text is retained verbatim, including historical glyphs and combining marks:
no modernization, NFC rewriting, punctuation repair or conversion of the new
`ezh` forms into modern z. Notes, original labels, raw teacher outputs and all
event history remain separate. The self-reported reviewer spelling in the JSON
is preserved rather than silently corrected.

## Annotation Acceptance Is Not Training Eligibility

Among the 20 verified lines, **13** belong to the protective Nowe Ateny/Wyprawa
families and **7** have unresolved bibliographic identity outside those families.
The new top-level manifest therefore contains 20 accepted annotations but
**zero training-eligible rows**. All top-level ledger/follow-up rows also have
training/evaluation gates closed. `training_freeze_ready=false`.

The previous immutable 64-pair reviewed pool is unchanged. This is a versioned
import of the additional batch, not a replacement of the old pool and not a
claim that 84 pairs are ready for model training. No model or GPU was run.
Single-human review is not independently adjudicated gold or a SOTA result.

`history/import/` retains the full original import unchanged. Its historical
`training-candidates.jsonl` describes the pre-work-gate annotation decision and
is **archival only**. Never use its old eligibility flags as the training input.
Use the top-level manifest and its gate report; no training selection is released.

## Validation And Package

`training.import_recognizer_line_review` checks event chronology, raw text history,
reviewer/decision fields, exact manifest/crop/context hashes, image formats and
dimensions, pinned train/validation/test metadata and the frozen holdout firewall.

`training.gate_recognizer_review_import` replays the import from raw events,
checks the frozen work-audit package and exact expansion-manifest binding,
reproduces the protective groups, and writes a separate gated annotation ledger.
It does not rewrite the original JSON or the user editor/localStorage.

Two fresh gated builds reproduce all output file bytes. The focused CPU suite
passes **271 tests**, including 15 new gate tests. The package contains:

- `manifest.jsonl`: 20 accepted annotations, all training/evaluation flags false.
- `review-ledger.jsonl`: all 64 lines with original statuses and work-gate reasons.
- `followup-queue.jsonl`: 44 unresolved/rejected/unreviewed rows, without promotion.
- `history/import/`: original 71-event export, 64 crops, 34 contexts, weak labels,
  raw teacher diagnostics, frozen configuration and raw import checksums.
- `work-audit/`: bound risk report/policy/source-access subset with its provenance.
- `gate-report.json`, `checksums.json`: explicit claims, limitations and hashes.

Images are unchanged IMPACT/PSNC source pixels via
`PiotrSty/impact-psnc-polish-ocr` at
`c7cb156fb95d2880699c33725bbaf1fbc1008fea`, CC-BY-3.0.
The public ZIP contains data, not code, weights or simulated review decisions.

## Public Artifacts

- [Complete gated data ZIP: crops, contexts, review and history](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/98033ad0348f18148db9a3d35192cf5c5ced350e/data/recognizer-expansion-gated-review-v1-20261006/recognizer-expansion-gated-review-v1-20261006.zip?download=true).
- [Public data publication and gate report](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/tree/98033ad0348f18148db9a3d35192cf5c5ced350e/data/recognizer-expansion-gated-review-v1-20261006).
- [Raw review export, ledger, follow-up queue and import evidence](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/020f26a98e839c837d9ac411da883a58a5047c21/experiments/2026-10-06/recognizer-expansion-review-import-v1).

Published against code revision `91c8d8c`. All published files were downloaded
again at the immutable HF revisions above and checked against local SHA-256;
all 125 unique ZIP members were checked individually. The complete package
checksum manifest has SHA-256
`87da9067ef6466d754f2f2e80d9985f964156c90c94bd3030fd663c4b1f591ac`.
The published README is the pre-link report snapshot. Later confirmations must
create a new version rather than altering this immutable publication.

```powershell
python -m training.gate_recognizer_review_import --import-directory RAW_IMPORT --work-audit WORK_AUDIT --config experiments/2026-10-05/recognizer-data-v3-expansion-v1/config.json --metadata-directory REGIONS_METADATA --output FRESH_GATED_OUTPUT
```

## Next Action

Do not repeat the 64-line review or Colab inference. First answer the explicit
question about the **31 proposed, complete-line** entries: was their corrected
text checked visually against the small crop? This question excludes the nine
rejected crops and both unchecked geometries. No answer has been applied by this
import; any later confirmation must have separate provenance and a new version.

The two entries with no saved decision are:

- `Discurs_FT__427010__r001__line007`.
- `O_cieplicach_FT__436233__r002__line015`.

The two proposals with unchecked geometry are:

- `Nowiny_z_Torunia_FT__436758__r002__line004`.
- `Nowiny_z_Torunia_FT__436761__r001__line003`.

Nine rejected geometries require new crops or exclusion, never automatic transfer
of an old verified status. Work-family quarantine remains independent of text
review. Bibliographic independence and an independently reviewed development
set must be resolved before a training run.
