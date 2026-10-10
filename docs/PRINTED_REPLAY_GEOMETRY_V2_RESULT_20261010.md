# Printed replay geometry V2: exact projection, regression gate passed

## Run

Per-line geometry V2 measured on the same 11 frozen pages as V1 (eight conflict
pages plus three controls, from the reproduced expansion audit root), with the
frozen miner's text, anchor and confidence gates reused verbatim through
`training.mine_printed_replay.exact_anchors` and the all-or-nothing page rule
unchanged. Only the geometry predicate differs.

**V2 predicate:** line extents are the projections of word-box corners onto the
normal of the page's median baseline angle — exact real arithmetic, no
axis-aligned re-bounding (which inflates a 700 px line by up to 15 px) and no
outward rounding (up to 1 px per edge). Two noise rules, both derived from
measurement rather than tuning:

- axes below `0.1°` snap to flat: a slope fitted from 2–3 words per line cannot
  resolve 0.1° (which is 1.2 px of x-coupling on a 700 px line), and a tilted
  axis turns exactly-touching boxes into false overlaps;
- overlaps at or below `1e-6` px count as zero: touching boxes project with
  ~1e-14 float residue and the frozen gate allows touching (`bottom > top`).

Per-line angles are used only to render upright crops.

## Results (frozen pages)

| Page | Role | V2 accepted / candidates | V2 conflicts | native conflicts | V1 deskew accepted |
| --- | --- | ---: | ---: | ---: | ---: |
| andersen-basnie-1929-0224 | control | 1 / 1 | 0 | 0 | 1 |
| andersen-basnie-1929-0225 | control | 13 / 13 | 0 | 0 | 13 |
| andersen-basnie-1929-0234 | control | 0 / 0 | 0 | 0 | 0 |
| may-nad-rio-de-la-plata-0523 | conflict | **17 / 17** | 0 | 11 | 17 |
| may-nad-rio-de-la-plata-0542 | conflict | **12 / 12** | 0 | 1 | 12 |
| may-nad-rio-de-la-plata-0549 | conflict | **14 / 14** | 0 | 1 | 14 |
| may-nad-rio-de-la-plata-0557 | conflict | 0 / 16 | 2 | 2 | 0 |
| may-nad-rio-de-la-plata-0563 | conflict | 0 / 19 | 1 | 9 | 0 |
| witkiewicz-teatr-0014 | conflict | 0 / 5 | 1 | 1 | 0 |
| witkiewicz-teatr-0022 | conflict | 0 / 6 | 1 | 1 | 0 |
| zawadzki-kurjer-1904-0004 | conflict (gate) | 0 / 10 | **1** | **1** | 0 |

| Totals (8 conflict pages) | native | V1 page-deskew | **V2** |
| --- | ---: | ---: | ---: |
| Accepted anchors | 0 | 43 | **43** |
| Extent conflicts | 27 | 9 | **6** |
| Source-order conflicts | 0 | 0 | 0 |

**Regression gate: PASSED** — the flat newspaper page keeps exactly its native
conflict count (1; V1 page-deskew inflated it to 4). Controls are unchanged (14
anchors accepted, zero conflicts). Coordinate mappings round-trip within
9.3e-13 px; 11 crops show a neighboring word box inside the crop rectangle.

## Residual conflicts (all measured, none artificial)

| Page | Pair | Exact overlap |
| --- | --- | ---: |
| may-…-0557 | [4, 5] | 2.105 px |
| may-…-0557 | [10, 11] | 2.358 px |
| may-…-0563 | [6, 7] | 2.135 px |
| witkiewicz-…-0014 | [21, 22] | 3.010 px |
| witkiewicz-…-0022 | [16, 17] | 2.135 px |
| zawadzki-…-0004 | [7, 8] | 2.000 px (= the native conflict) |

These are genuine 2–3 px extent overlaps on tight-leading pairs. **No rectangle
geometry can separate them**; the crop of one line necessarily contains part of
the neighbor's word boxes.

## Findings

1. **V2 equals V1's recovery and fixes its regression.** Same 43 anchors, but
   the newspaper page no longer gains conflicts (4 → 1, native parity).
2. **The naive per-line-frames variant was falsified by the regression gate.**
   Testing each pair in both lines' own frames couples x into the separation
   (a 0.3–1.3 px false overlap from slope-fit noise alone) and invented
   conflicts even on a control page. The shared-axis design and its regression
   test (`test_shared_axis_prevents_frame_mismatch_noise`) record this.
3. **At this precision, arithmetic and tie semantics are part of the
   geometry**: float dust (1e-14) and the frozen touch-allowed rule each needed
   explicit handling (`PROJECTION_EPS_PX`, `FLAT_AXIS_EPS_DEGREES`), with tests.
4. **The geometry approach has reached its boundary**: the last 56 anchors are
   gated by 6 true 2–3 px overlaps. Clearing them requires crop-level ink
   analysis (does neighbor ink actually enter the crop?) or a reviewed policy
   accepting crops with measured neighbor intrusion — 11 of 43 accepted crops
   already show a neighbor word box inside the crop rectangle and are marked
   for boundary review.

## Decision And Next

V2 is the geometry path for the replay corpus: exact gate, upright per-line
crops, honest residuals. All pairs remain `eligible_for_training: False` and
nothing is promoted. The 43 accepted anchors become candidates together with
V1's; the 56 gated ones wait for the crop-level ink review. Next gates, in
order: human boundary review of the 11 intrusion crops and the 6 residual
pairs; content-type classification (prose / contents / publisher backmatter);
only then a replay training dataset.

[Kompletny ZIP i dowody na HF (rewizja pinned)](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/faabf819f252ff9ba6d3c1e01319eef3e42d42b5/experiments/2026-10-10).

## Reproduce

```powershell
python -m pytest -q tests/test_printed_replay_geometry_v2.py
python -m training.printed_replay_geometry_v2 `
  --audit-root data/printed-replay-expansion-v1-audit-reproduced `
  --output experiments/2026-10-10/printed-replay-geometry-v2
```

Limitations: intrusion is rectangular word-box overlap, not ink; the page axis
is a median of word-center slope fits and cannot model warped pages; crops are
not human reviewed; no transcription label was changed.
