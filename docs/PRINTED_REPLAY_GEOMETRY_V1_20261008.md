# Printed replay line geometry V1: protocol

## Goal

Compare unchanged native segmentation with deskew geometry handling on the eight
pages whose exact anchors were discarded by the all-or-nothing overlap gate, plus
three control pages without conflicts. The experiment decides whether geometry
handling can increase usable replay yield; it does not promote any pair to
training or evaluation and does not change the frozen miner, its gates or the
baseline.

## Frozen inputs

- Evidence source: the reproduced audit root
  `data/printed-replay-expansion-v1-audit-reproduced`, produced by
  `python -m training.audit_printed_replay_expansion_evidence` from the immutable
  HF archive `printed-replay-expansion-v1-evidence.zip` (94,678,979 bytes).
- Input hashes are recorded in `report.json`
  (`input_geometry_diagnostic_sha256`, `input_audit_sha256`,
  `input_evidence_checksums_sha256`) and per page
  (`native_page_sha256`, `tesseract_tsv_sha256`).
- Page selection is derived from the reproduced `geometry-diagnostic.json`:
  the eight conflict pages plus the first three remaining page IDs as controls.
- Word TSV is re-parsed locally and must reproduce the frozen `parse_tsv`
  records exactly (`strip_words(parse_words(tsv)) == parse_tsv(tsv)`), otherwise
  the run aborts.

## Variants

| Variant | Comparison frame | Crop source |
| --- | --- | --- |
| `native` | 0 degrees, unchanged boxes | native page pixels |
| `page-deskew` | page rotation by the median word-center slope of all lines with at least two words, clamped to +/-5 degrees (0 degrees outside the clamp) | page pixels rotated by the same angle and center |

In both variants a line extent is the union of its transformed **word boxes**.
Rotating an already-collapsed axis-aligned line rectangle would inflate it and is
not used. Word centers drive the slope estimate; words are never moved on their
own.

## Gates

Every text, anchor and confidence gate of the frozen miner
(`training.mine_printed_replay.exact_anchors`, runner pinned at
`c8ae68ae741d2c30d95317c2c0bb84d347d79a16`) is reused unchanged inside the
comparison frame: minimum word confidence 90, minimum length 20 characters and 3
words, line-like extent, valid glyphs, unique exact NFC anchor with whitespace
boundaries, source-order and vertical-extent conflict rule, and page coverage
treated diagnostically (`page_min_coverage=0`, protocol v2). Only the geometry
the gate sees changes between variants.

## Recorded outputs

- `report.json`: per page and per variant accepted anchors, rejected reasons,
  extent conflicts, page skew, aggregate comparisons.
- `pairs.jsonl`: one record per accepted pair with native bbox, frame bbox, crop
  bbox, the 2x3 `affine_native_to_crop` matrix, the four crop corners mapped back
  to native coordinates and the measured round-trip error of that mapping.
- `crops/<variant>/<page_id>/*.png` and `texts/<variant>/<page_id>/*.txt`:
  rendered pairs; text bytes are the normalized anchor text.
- `geometry-sheet-<page_id>.png`: native and deskew line boxes drawn over page
  pixels around the first native conflict.
- `crop-sheet-<variant>-<page_id>.png`: accepted crops with every intruding
  neighbor word-box rectangle drawn inside the crop.
- `environment.json`, `checksums.json`, and `printed-replay-geometry-v1-evidence.zip`
  containing all of the above.

Intrusion is measured as the rectangular overlap of the crop with **all** other
lines' word boxes on the page, in the same comparison frame.

## Non-goals

- No pair is promoted; `eligible_for_training` and `eligible_for_evaluation`
  stay `False`, `promoted_pairs` is 0.
- No transcription label is changed, no rejected pair is relabeled.
- No ink-collision analysis: rectangle overlap is a proxy, not proof of touching
  ink.
- No per-line slanted quads or curved-baseline handling; a single page angle
  cannot model warped lines. If V1 leaves residual conflicts this is the next
  candidate.
- No prose/contents/publisher-backmatter classification; that gate belongs to the
  replay training dataset build.

## Reproduce

```powershell
python -m training.printed_replay_geometry `
  --audit-root data/printed-replay-expansion-v1-audit-reproduced `
  --output experiments/2026-10-08/printed-replay-geometry-v1
```

The output directory must not exist. CPU only; Pillow is the only imaging
dependency. Focused tests: `tests/test_printed_replay_geometry.py`.
