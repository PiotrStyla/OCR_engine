# Printed replay pilot V2: verified crops, insufficient replay scale

## Returned Run

The CPU Colab run completed. The original `printed-replay-pilot-v2-evidence.zip`
has 35,135,620 bytes and SHA256
`7108b55c9fa1ca254864d1d074545e177e1d355e6d9fe0996c115e97b2122f8c`.
All 173 payload checksums, complete checksum coverage, safe archive paths,
the pinned source package and the runner at
`8b88ee5c8dd6cdbb429dbf6980077eaf35f32676` were verified.

[Public original ZIP, machine audit and separate visual-review receipt](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/fbcff86826be1e06773095bb433d44a9bb05e46f/experiments/2026-10-07/printed-replay-pilot-v2-result).

Environment receipts report Pillow 11.3.0, Tesseract 5.3.4 and DjVuLibre 3.5.28.
These are recorded versions, not an independently reproduced local OCR run.

## Recomputed Results

| Check | Result |
| --- | ---: |
| Complete native pages and word TSV files | 12 |
| Lines reconstructed from word TSV | 396 |
| Candidate image/text pairs | 26 |
| Replay-candidate: Zeromski, one work family | 25 |
| Replay-probe: Prus, a separate work family | 1 |
| Crops pixel-identical to returned native page regions | 26/26 |
| Quarantined lines with independently reproduced records | 370 |
| Training or evaluation promotion | None |

The audit recomputes word grouping, bounding boxes, minimum word confidence,
exact unique source spans, line order, exclusion reasons and page counts.
It verifies crop padding, RGB pixels, text bytes, file hashes, source revisions,
licenses, split assignments and complete pair/native-file coverage.

Quarantine: 306 low-confidence lines, 53 without a unique exact source anchor,
10 short anchors and one non-line-like region. Historical spelling is untouched:
only NFC and whitespace normalization participate in exact matching.

An AI-agent inspected all 26 crops and their surrounding-line context in six
contact sheets. No obvious incomplete line, neighboring-line intrusion or
text/image mismatch was observed. This is **not human annotation or certified
ground truth**. The original `not-human-reviewed` records remain unchanged.
Margins are tight and the Zeromski native pages are only about 0.8 megapixels.

## Decision

Keep the 25 candidate pairs and the separate single probe as a small,
provenance-preserving data-engine pilot. **Do not start a recognizer training run
on this result alone. Keep the baseline unchanged.**

These pairs are selected for exact Tesseract/source agreement, so they favor
easy lines. Two literary works are neither a representative ordinary-document
corpus nor a SOTA benchmark. Upstream Wikisource pagequality=4 is not project
human review; prior model exposure to these works is unknown. Original DjVu
decoding and OCR execution were not rerun locally: the pixel check establishes
derivation from returned native images, not independent decoding correctness.

As a diagnostic only, bypassing the confidence gate while retaining the exact
source, boundary, length, geometry and order rules yields 38 anchors instead of
26. The extra 12 were not exported as training pairs. Merely relaxing confidence
cannot provide the replay scale we need.

## Next Stage

1. Expand source diversity at work/document level, excluding protected families
   and all current development/evaluation material. Add ordinary Polish printed
   documents beyond literature; record licenses and immutable source revisions.
2. Pilot line segmentation and independent teacher agreement against frozen
   source text. Preserve raw predictions; do not treat approximate alignment or
   a teacher's confident correction as verified transcription.
3. Measure usable lines per source, boundary failures, duplicate rates and
   domain coverage before choosing a training budget. Review disagreements and
   a random sample of agreements, rather than assigning the whole pool to the user.
4. Train only after a sufficiently diverse replay set exists. Require paired
   improvement against the unchanged baseline on historical and ordinary print,
   then evaluate complete held-out documents. Keep the single Prus probe out of
   training; do not present it as a statistically meaningful test set.

The V2 notebook is now archival. Rerunning the unchanged 12-page pilot is not
the next project step.

## Reproduce the Machine Audit

With Pillow and the repository available, run:

```powershell
python -m training.audit_printed_replay_v2_evidence `
  --archive "C:/Users/Hipek/OneDrive/Pulpit/printed-replay-pilot-v2-evidence.zip" `
  --output data/printed-replay-pilot-v2-audit-reproduced
```

The output directory must be new. The command preserves the original ZIP,
extracts verified evidence, emits `audit.json`, six contact/context sheets and
checksums. Visual inspection is a separate step; the machine command deliberately
reports it as pending. The completed local audit also contains the separately
recorded `visual-review.json` and diagnostic `confidence-sensitivity.json`.

Focused regression suite: **59 tests passed**. Tests cover metadata/TSV drift,
rehashed modified crop pixels, missing candidates, quarantine drift, V1 archive
validation, V2 protocol selection and recognizer baseline protections.
