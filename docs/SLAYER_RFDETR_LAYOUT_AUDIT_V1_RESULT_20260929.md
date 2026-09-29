# SLAYER-OCR RF-DETR best-checkpoint audit v1

Date: 2026-09-29

Status: **reproducible audit completed; quality gate failed; label-policy review required**

## Integrity and reproduction

- Audit ZIP SHA-256: `7fbab49a5970b1f00ecc55ead04ebc1e6cbabe00139f574870691a602a85a663`
- Audit code revision: `4e49211f1bafd35b0f05ade8f8d94888363b351c`
- Checkpoint SHA-256: `eda8e385179fa9df51baafcfd1accb3591d83ba88a73c620ba3436bcdf13b93c`
- Dataset ZIP SHA-256: `f0a157c79756276e9a82e4f8010521190bf72dc8b459c81c3806e840681a0c33`
- Model ZIP SHA-256: `68578a6e008788fd41b6b9f64d31b3a9def82f25c615ead8fc97da5a9962fb0b`
- Training evidence ZIP SHA-256: `ce5e68096866ab903d7722a75da1b0b5cf44215d597da49c7a8137859773fbde`
- Internal checksums verified: 17/17 files

The saved checkpoint reproduces the logged best result exactly on the frozen
12-page development split: `mAP@50:95 = 0.3272939026`. This confirms the model
archive and selected epoch-15 EMA checkpoint. The audit discarded one explicit
RF-DETR background sentinel and did not treat it as a document object.

The audit ZIP contains private page overlays and predictions. Those files are
preserved locally and are not published in this repository.

## Fixed-threshold diagnostic

This is a class-aware diagnostic at score threshold 0.25 and IoU 0.5, not a
replacement for COCO mAP:

| Measure | Value |
| --- | ---: |
| True positives | 33 |
| False positives | 51 |
| False negatives | 27 |
| Precision | 0.3929 |
| Recall | 0.5500 |
| F1 | 0.4583 |
| Hard-example pages | 12/12 |

`figure` is the only reliable class in this small diagnostic (5 TP, 2 FP,
2 FN). `text_region` reaches 27 TP but also produces 42 FP and 16 FN.
`heading` has 1 TP, 4 FP and 4 FN. No table, marginalia or page-number object
is matched at this threshold. Caption, header and footer have no validation
support.

The four highest error counts are:

| Page | TP | FP | FN | FP + FN |
| --- | ---: | ---: | ---: | ---: |
| `Diariusz_FT__436661` | 5 | 3 | 12 | 15 |
| `Nowe_nowiny_z_Czech_FT__436778` | 0 | 8 | 3 | 11 |
| `List_FT__436642` | 2 | 8 | 1 | 9 |
| `Nowe_nowiny_z_Czech_FT__436793` | 3 | 7 | 1 | 8 |

## Visual diagnosis

Inspection of the private overlays shows that the fixed-threshold error count
mixes model errors with annotation-policy disagreements:

- the model often predicts coherent paragraph-sized regions where ground truth
  splits the same text into many short or line-sized boxes;
- one table page has an overlapping page-sized `text_region` and `table` ground
  truth, while the model fragments the content and misses `table`;
- a decorative initial inside text is absorbed into a larger `figure`
  prediction, showing that the figure/text boundary is not yet operationally
  stable;
- one page contains a visible stamp predicted as `figure` but omitted from the
  ground truth, indicating at least one plausible annotation omission.

These observations are review hypotheses from 12 development pages, not proof
of dataset-wide error rates. They do show that another training run on the same
labels would confound model quality with label granularity.

## Decision and next gate

Do not claim layout quality or proceed directly to a larger retraining run.
First adjudicate all 12 development pages under a written policy that fixes:

1. paragraph/block versus line-level `text_region` granularity;
2. whether `table` is exclusive or may overlap `text_region`;
3. treatment of decorative initials embedded in text;
4. treatment of library stamps, scan artifacts and other non-content marks;
5. minimum box size and rules for headings, marginalia and page numbers.

After the labels are corrected, re-run the same frozen checkpoint audit. Only
then should the project decide whether to expand annotations, tune thresholds or
retrain RF-DETR.
