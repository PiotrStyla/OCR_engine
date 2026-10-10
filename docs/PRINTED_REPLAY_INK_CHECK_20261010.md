# Printed replay: ink-level collision check — wynik

## Question

The geometry V2 gate rejected 56 exact anchors behind 6 word-box overlaps of
2–3 px. Rectangles are not ink: this check measures whether the printed ink of
the two lines actually collides, and whether neighbor ink enters accepted
crops. It produces review evidence only — no crop is accepted, rejected or
relabeled here.

## Method

Frozen native pages (reproduced expansion audit root), per-case adaptive ink
threshold inside the union region (median vs darkest), ink = thresholded pixels
inside each line's word boxes. Pair measure: whitespace rows between the two
lines' ink, column by column over columns where both lines have ink. Crop
measure: neighbor ink pixels inside the crop rectangle.

## Results — the 6 blocking pairs (all ink-separated)

| Page | Pair | Box overlap | **Ink gap** | Shared ink columns | Columns in contact |
| --- | --- | ---: | ---: | ---: | ---: |
| may-…-0557 | [4, 5] | 2.105 px | **14 px** | 243 | 0 |
| may-…-0557 | [10, 11] | 2.358 px | **13 px** | 255 | 0 |
| may-…-0563 | [6, 7] | 2.135 px | **17 px** | 375 | 0 |
| witkiewicz-…-0014 | [21, 22] | 3.010 px | **14 px** | 669 | 0 |
| witkiewicz-…-0022 | [16, 17] | 2.135 px | **13 px** | 707 | 0 |
| zawadzki-…-0004 | [7, 8] | 2.000 px | **7 px** | 350 | 0 |

Zero of 6 pairs have ink contact. The rectangle overlaps are word-box padding
(ascender/descender slack around glyphs), not colliding print. **The 56 gated
anchors are blocked by boxes, not by ink.**

## Results — the 11 crops with box intrusion

| Verdict | Crops | Neighbor ink pixels in crop |
| --- | ---: | --- |
| boxes-only (0 ink inside) | 5 | 0 |
| ink-inside-crop | 6 | 1, 1, 8, 9, 15, 30 |

The six ink intrusions are tiny (1–30 px on crops of ~30,000 px) and sit at the
crop boundary; they are the exact cases a human boundary review must decide
(trim the crop, accept the trace, or reject the pair).

## Findings

1. **Ink collision is not the blocker; box padding is.** All 6 blocking pairs
   are ink-separated by 7–17 px. A crop cut between the ink bands is physically
   clean even where the frozen rectangle gate rejects the pair.
2. **Most box intrusions are also empty of ink** (5 of 11 crops), so the
   rectangle intrusion metric over-states risk as well.
3. Together with geometry V2 (43 accepted, 6 box conflicts, regression gate
   passed), the path to the remaining 56 candidates is now a **reviewable
   decision with machine evidence**, not an open geometry problem.

## Decision Gates (human)

For the 6 ink-separated pairs: allow crops cut at the ink gap (candidate
promotion after review) or keep the conservative rectangle gate — either is
defensible; the evidence supports the former.
For the 6 ink-inside crops: per-crop call on the 1–30 px traces.
All 56 anchors and 43 accepted pairs stay `eligible_for_training: False` until
that review passes; nothing here changes a frozen reference or a gate.

## Reproduce

```powershell
python -m pytest -q tests/test_printed_replay_ink_check.py
python -m training.printed_replay_ink_check `
  --ink-root data/printed-replay-expansion-v1-audit-reproduced `
  --geometry-root experiments/2026-10-10/printed-replay-geometry-v2 `
  --output experiments/2026-10-10/printed-replay-ink-check
```

Limitations: ink threshold is adaptive and measured, not calibrated; per-column
band gaps approximate distance along the line normal; no transcription label
changed.
