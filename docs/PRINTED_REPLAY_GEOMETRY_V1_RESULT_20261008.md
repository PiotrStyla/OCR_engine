# Printed replay line geometry V1: deskew recovers 43 of 99 anchors

## Returned Run

The CPU experiment ran locally against the reproduced expansion audit root and
finished in 23 s. It selected the eight conflict pages recorded in
`geometry-diagnostic.json` plus three control pages
(`andersen-basnie-1929-0224`, `andersen-basnie-1929-0225`,
`andersen-basnie-1929-0234`) and wrote 71 pairs, 22 crop sheets, 8 geometry
sheets and 160 verified checksums.

Frozen inputs are pinned in `report.json`
(`input_geometry_diagnostic_sha256`, `input_audit_sha256`,
`input_evidence_checksums_sha256`) with per-page `native_page_sha256` and
`tesseract_tsv_sha256`. Recorded environment: Pillow 12.2.0, Python 3.11.9;
`environment.json` pins `module_sha256` and the code revision.

[Complete ZIP, sheets and receipts on HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/089ff368ec0f5b49230940c09a7f20b3d62859c4/experiments/2026-10-08/printed-replay-geometry-v1).
The evidence ZIP is 11,010,924 bytes, SHA256
`5ae8216f39b520d462ae0a153ef9fedf70f72c51aec1dac4fea22499927bad76`.

## Recomputed Results

| Variant | Accepted anchors (8 conflict pages) | Rejected by overlap gate | Extent conflicts | Source-order conflicts |
| --- | ---: | ---: | ---: | ---: |
| native | 0 | 99 | 27 | 0 |
| page-deskew | **43** | 56 | 9 | 0 |

The native variant reproduces the frozen diagnostic exactly: 27 extent
conflicts, 99 rejected exact anchors, zero source-order conflicts.

| Page | Skew | Conflicts native -> deskew | Accepted deskew | Intruding crops |
| --- | ---: | ---: | ---: | ---: |
| may-nad-rio-de-la-plata-0523 | -1.204 | 11 -> 0 | 17 | 7 |
| may-nad-rio-de-la-plata-0542 | +0.741 | 1 -> 0 | 12 | 0 |
| may-nad-rio-de-la-plata-0549 | -0.824 | 1 -> 0 | 14 | 0 |
| may-nad-rio-de-la-plata-0557 | -0.734 | 2 -> 2 | 0 | 0 |
| may-nad-rio-de-la-plata-0563 | -0.786 | 9 -> 1 | 0 | 0 |
| witkiewicz-teatr-0014 | -1.071 | 1 -> 1 | 0 | 0 |
| witkiewicz-teatr-0022 | +0.705 | 1 -> 1 | 0 | 0 |
| zawadzki-kurjer-1904-0004 | -0.019 | 1 -> 4 | 0 | 0 |

Control pages are identical in both frames (1, 13 and 0 accepted anchors, zero
conflicts): geometry handling does not disturb conflict-free pages.

## Validations

- Word TSV re-parsing must equal the frozen `parse_tsv` records or the run
  aborts; it matched on all 11 pages.
- All text, anchor and confidence gates are the frozen miner's, executed inside
  the comparison frame. Rejection reasons outside the overlap gate are identical
  in both variants.
- Coordinate mappings are recorded per pair (`affine_native_to_crop`,
  `native_corners`, `roundtrip_error_px`); the maximum round-trip error is
  5.2e-13 px. A regression test checks `rotate_point`/`affine` against real
  `PIL.Image.rotate` pixels and checks crop pixels against the rotated frame.
- All 160 evidence checksums were recomputed from the published ZIP content
  without mismatches.

## Findings

1. **Single-angle deskew is safe and useful where tilt dominates.** Three May
   pages recover all 43 of their exact anchors with zero residual conflicts.
   Extents shrink by up to 15 px once lines are measured in a level frame.
2. **Tight-leading backmatter resists one page angle.** Pages 0557 and 0563
   (publisher backmatter in the May volume) keep conflicts of 2-3 px and 3 px;
   0563 drops from 9 conflicts to 1 but the all-or-nothing gate still discards
   all 19 anchors.
3. **A stubborn single pair stays on each Witkiewicz page** (13 px -> 4 px and
   11 px -> 3 px). Rotation shrinks but does not separate them.
4. **The nearly flat newspaper page gets worse** (1 -> 4 conflicts). Verified
   cause: at a measured skew of -0.019 degrees the framing grows every word-box
   extent by 0-2 px purely from axis-aligned re-bounding and outward rounding,
   which is enough to create 1-2 px conflicts. Rectangle framing is
   noise-sensitive near zero skew.
5. **7 of 43 recovered crops overlap a neighboring line's word boxes**, all on
   the steepest page (0523), 46-554 px^2 per crop on strips 46-53 px tall. These
   are candidate crops for boundary review, not verified pairs.

## Decision And Next Experiment

Keep the baseline and the frozen miner unchanged. All 71 pairs stay
`eligible_for_training: False`, `promoted_pairs` is 0 and no label changed.
Usable yield measured by this experiment: **43 of 99 anchors (43%) are
recoverable by page-level deskew alone**; the remaining 56 need per-line
geometry, not a better page angle.

Next bounded experiment: replace axis-aligned line extents with per-line
geometry — rotated word quads or word polygons with per-line angles — on the
same 11 frozen pages, keeping the exact text, anchor and confidence gates. It
must resolve the tight-leading backmatter pairs and must not create conflicts on
the flat newspaper page (case 4 is a hard regression gate). Column-aware
grouping is a separate step for the multi-column newspaper. Before any replay
training dataset is built, prose, contents and publisher backmatter rows must be
distinguished; 0557 and 0563 are backmatter per the previous audit.

No human review and no model training is required at this stage.

## Reproduce

```powershell
python -m training.printed_replay_geometry `
  --audit-root data/printed-replay-expansion-v1-audit-reproduced `
  --output experiments/2026-10-08/printed-replay-geometry-v1
```

Use a new output directory. Focused regression suite: `tests/test_printed_replay_geometry.py`
(8 CPU tests). Intrusion is rectangular word-box overlap, not ink collision;
exact-anchor selection still favors easy lines; crops are not human reviewed.
