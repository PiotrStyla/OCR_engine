# SLAYER-OCR RF-DETR layout pilot v1

Date: 2026-09-29

Status: **technically valid private development checkpoint; quality gate remains open**

## Verified artifacts

- Evidence ZIP SHA-256: `ce5e68096866ab903d7722a75da1b0b5cf44215d597da49c7a8137859773fbde`
- Model ZIP SHA-256: `68578a6e008788fd41b6b9f64d31b3a9def82f25c615ead8fc97da5a9962fb0b`
- Best checkpoint SHA-256: `eda8e385179fa9df51baafcfd1accb3591d83ba88a73c620ba3436bcdf13b93c`
- Dataset ZIP SHA-256: `f0a157c79756276e9a82e4f8010521190bf72dc8b459c81c3806e840681a0c33`
- Runtime: RF-DETR 1.11.0, PyTorch 2.11.0+cu128, Tesla T4

Both archives have unique, traversal-free members and complete internal
SHA-256 manifests. The model archive contains the 127.6 MB
`checkpoint_best_total.pth`, training configuration, run manifest and metrics.
No archive or derived page image is published in this repository.

## Result

Training used 48 pages from 18 collections and a collection-disjoint internal
development split of 12 pages from four collections. Exact image-hash overlap
is zero. Early stopping ended the run after epoch 30. The checkpoint metadata
selects the EMA weights whose best validation mAP occurred at epoch 15:

| Metric at best epoch 15 | Value |
| --- | ---: |
| mAP 50:95 | 0.3273 |
| mAP 50 | 0.4055 |
| mAP 75 | 0.3589 |
| mAR | 0.6698 |
| F1 sweep | 0.3883 |
| Precision | 0.4362 |
| Recall | 0.3815 |

Per-class EMA AP at the selected epoch (the checkpoint source is `ema`):

| Class | AP 50:95 | Valid objects |
| --- | ---: | ---: |
| figure | 0.8238 | 7 |
| text_region | 0.3980 | 43 |
| page_number | 0.4040 | 2 |
| heading | 0.2426 | 5 |
| table | 0.0498 | 1 |
| marginalia | 0.0456 | 2 |
| caption | not measurable | 0 |
| header | not measurable | 0 |
| footer | not measurable | 0 |

`final_metrics.json` reports mAP 0.3029 at epoch 30. It is not the score of the
best saved checkpoint. RF-DETR correctly promoted the epoch-15 EMA checkpoint,
whose embedded callback state records 0.3273 and `best_total_source=ema`.
The CSV contains both regular and explicitly prefixed EMA per-class columns;
the table above uses the latter. The next audit re-evaluates the saved checkpoint
directly and is authoritative if those columns disagree.

## Decision

The execution, provenance and checkpoint-integrity gates pass. The result shows
that the student learns the private layout labels, but it does not establish an
improvement over the existing OCR pipeline, final-test quality or SOTA.

The quality gate stays open because the development sample is small, three
classes have no validation support, and `table` and `marginalia` remain weak.
The next experiment is a private visual audit of the best checkpoint on all 12
development pages. It will preserve raw predictions, render GT/prediction
comparisons, and mine fixed-threshold false positives and false negatives.
Retraining or full-page OCR A/B follows only after that review.
