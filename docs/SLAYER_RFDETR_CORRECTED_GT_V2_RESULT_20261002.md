# SLAYER-OCR RF-DETR corrected-GT reanalysis v2

Date: 2026-10-02

Status: **reanalysis completed; checkpoint lineage verified; quality gate failed**

## Integrity

- Audit ZIP SHA-256: `84efbe41837a42f94eea27c3176b6c6af78e837d7184d3356ceae311eaa87e51`
- Audit code revision: `964691bd1ea1db7d51abfee30ec50e41628f83ca`
- Corrected dataset ZIP SHA-256: `1cfb920217c15e1c2b4f0de50a972ca30bc6cbf975022dbfc9597fdbee83732d`
- Frozen model ZIP SHA-256: `68578a6e008788fd41b6b9f64d31b3a9def82f25c615ead8fc97da5a9962fb0b`
- Training evidence ZIP SHA-256: `ce5e68096866ab903d7722a75da1b0b5cf44215d597da49c7a8137859773fbde`
- Checkpoint SHA-256: `eda8e385179fa9df51baafcfd1accb3591d83ba88a73c620ba3436bcdf13b93c`
- Internal audit checksums verified: 17/17 files
- Audit mode: `corrected-gt-reanalysis`

The corrected dataset passed its complete checksum manifest and lineage checks.
It descends from the exact dataset used to train the frozen checkpoint and
points to the same checkpoint shown during human review.

## Ground-truth changes

The 48-page training split is unchanged. Human review modified the 12-page
development split as follows:

| Change | Count |
| --- | ---: |
| Added objects | 18 |
| Deleted objects | 3 |
| Relabeled objects | 4 |
| Moved/resized boxes | 13 |
| Validation objects before | 60 |
| Validation objects after | 75 |

All 12 pages remain development data. The review is one completed human pass,
not an independently double-adjudicated benchmark release.

## COCO evaluation

The same epoch-15 checkpoint was evaluated without retraining.

| Metric | Original GT v1 | Corrected GT v2 | Delta |
| --- | ---: | ---: | ---: |
| mAP@50:95 | 0.3273 | 0.2570 | -0.0703 |
| mAP@50 | 0.4055 | 0.3420 | -0.0634 |
| mAP@75 | 0.3589 | 0.2839 | -0.0751 |
| mAR | 0.6698 | 0.5527 | -0.1171 |
| RF-DETR evaluator F1 | 0.3883 | 0.3339 | -0.0544 |

Corrected-GT mAP@50:95 is 21.5% lower relative to the original-GT result. The
original `0.3273` remains a valid reproduction result for v1 labels, but it is
not the accepted quality estimate after review.

Interpretation correction: the class support also changed. The v1 mean covers
six classes; v2 covers eight after adding header/footer support. The v2 mean
over the original six supported classes is approximately `0.2726`. Therefore
the overall 21.5% difference is not an isolated measure of label correction
or model degradation. Ground truth changed and the checkpoint did not train
again. A clean counterfactual requires rescoring identical frozen predictions
with both label versions and reporting common-class as well as full means.

The single review pass is still provisional. Visual inspection found a library
inventory mark on `Diariusz_FT__436661` labeled as `figure`, contrary to the
annotation policy for non-content library artifacts. This finding is recorded
for targeted adjudication; the published GT v2 has not been silently modified.

Corrected-GT per-class AP is:

| Class | AP@50:95 |
| --- | ---: |
| text_region | 0.3790 |
| heading | 0.2004 |
| table | 0.0498 |
| figure | 0.6400 |
| marginalia | 0.0064 |
| header | 0.3500 |
| footer | 0.0707 |
| page_number | 0.3598 |

`caption` has no development support. Values for rare classes are unstable and
must not be presented as reliable dataset-wide estimates.

## Fixed-threshold diagnostic

At score threshold 0.25 and IoU 0.5:

| Measure | Original GT v1 | Corrected GT v2 |
| --- | ---: | ---: |
| True positives | 33 | 38 |
| False positives | 51 | 46 |
| False negatives | 27 | 37 |
| Precision | 0.3929 | 0.4524 |
| Recall | 0.5500 | 0.5067 |
| F1 | 0.4583 | 0.4780 |
| Hard-example pages | 12/12 | 12/12 |

This diagnostic improves slightly because corrected boxes convert some former
false positives into true positives. COCO mAP still falls because the reviewed
GT adds difficult objects and evaluates ranking and localization over the full
threshold range. The two measurements answer different questions.

The four highest corrected-GT error counts are:

| Page | TP | FP | FN | FP + FN |
| --- | ---: | ---: | ---: | ---: |
| `Diariusz_FT__436661` | 5 | 3 | 11 | 14 |
| `List_FT__436642` | 4 | 6 | 5 | 11 |
| `Nowe_nowiny_z_Czech_FT__436778` | 1 | 7 | 4 | 11 |
| `Relacja_spraw_FT__437057` | 5 | 5 | 4 | 9 |

## Decision

The checkpoint fails the quality gate. Do not promote it as a production layout
model and do not tune only the inference threshold against these 12 pages.

The next training cycle should first expand and rebalance reviewed layout data,
especially `table`, `marginalia`, `header`, `footer`, `page_number` and
`caption`. A new model run should use a separately frozen holdout and report
both COCO metrics and a fixed-threshold operating point. The corrected 12-page
split remains a development diagnostic and should not become the final test set.
