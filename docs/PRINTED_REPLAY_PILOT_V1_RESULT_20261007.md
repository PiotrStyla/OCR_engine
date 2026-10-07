# Printed replay V1: empty result and repair

## Audited Result

Returned ZIP: `printed-replay-pilot-v1-evidence.zip`, 4,450,745 bytes.
SHA256: `f89523ca33d86afe62990ed398fac969633918eba3ca1388dfcde8145a14e8fe`.
All 85 payload checksums and complete ZIP coverage verified. The nested original
source package matches the frozen source checksum manifest, and the runner hash
matches commit `794f48feba605da4664d09610c735eb84c04e9c7`.

Runtime: Tesseract 5.3.4, DjVuLibre 3.5.28, Pillow 11.3.0. All 12 pages completed;
396 returned line records, **zero accepted pairs**. The manifest is empty.

| Rejection reason | Lines |
| --- | ---: |
| Some word confidence below 90 | 306 |
| No unique exact source anchor | 53 |
| Too short | 10 |
| Whole-page exact coverage below 40% | 26 |
| Non-line-like geometry | 1 |

All counts, page coverage values, exact source spans and exclusion reasons were
recomputed from the returned raw line records and frozen transcriptions. No model
was trained; the baseline remains unchanged.

## Protocol Issue

The whole-page coverage gate discarded 26 lines that had already passed exact
matching, length, word confidence, order and geometry rules. It is unsuitable as
a mandatory gate for an explicitly partial line-candidate pool. It was our policy,
not a runtime failure and not proof of a defective recognizer.

Removing **only that gate** yields 25 diagnostic line proposals from the candidate
work and one from the probe work, across six pages. No character substitutions,
confidence relaxation or hyphen repair were used. These are proposals, **not
recovered crops or approved training data**. The other 370 lines remain excluded.

Low confidence and the very small surviving pool remain real limitations. This
run does not establish that Tesseract confidence is calibrated, that source text
is error-free, or that the proposed rectangles contain complete lines.

## V2 Repair

The miner supports `--protocol v2`; V1 remains the default for reproduction.
V2 reports page coverage but does not discard an exact line solely for low page
coverage. All other line gates remain unchanged. All eligibility flags stay false.
V2 evidence also contains native-resolution PNG pages, complete Tesseract TSV and
stderr, so crop boundaries and excluded lines can be inspected without rerunning
the original books. Candidate records include explicit line IDs and TSV keys.

V1 evidence did not contain those native page images, TSV or original DjVu files.
Pixel/word-level geometry was therefore **not independently verified** in this
audit. A fresh V2 extraction is needed before recovered crops can be assessed.
Do not train on the 26 text-only proposals or use the probe for training.

## Local Audit

```bash
python -m training.audit_printed_replay_evidence \
  --archive printed-replay-pilot-v1-evidence.zip \
  --output data/printed-replay-pilot-v1-audit-20261007
```

Preserves the original ZIP, verified extracted evidence, `audit.json`, and
`exact-anchor-proposals.jsonl`. Use a new output directory. The V2 repair is
prepared as a separate [V2 CPU protocol](PRINTED_REPLAY_PILOT_V2_20261007.md);
this report does not claim it has been executed on Colab.
