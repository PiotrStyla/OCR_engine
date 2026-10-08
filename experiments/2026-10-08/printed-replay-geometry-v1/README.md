# Printed replay line geometry V1 (2026-10-08)

Compares unchanged native segmentation with single-angle page deskew on the
eight printed-replay conflict pages and three controls. Result: 43 of 99 exact
anchors recovered, 56 still gated, controls untouched, no promotion.

- Protocol: `docs/PRINTED_REPLAY_GEOMETRY_V1_20261008.md`
- Result and decision: `docs/PRINTED_REPLAY_GEOMETRY_V1_RESULT_20261008.md`
- Complete evidence ZIP, sheets and receipts:
  https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/089ff368ec0f5b49230940c09a7f20b3d62859c4/experiments/2026-10-08/printed-replay-geometry-v1

This directory keeps `report.json` and the machine receipts and visual sheets
under `evidence/`. The line crops, normalized texts and the evidence ZIP are
published on Hugging Face only (`evidence/checksums.json` covers them by hash);
`pairs.jsonl` references crop paths inside the ZIP. Nothing here is eligible for
training or evaluation.
