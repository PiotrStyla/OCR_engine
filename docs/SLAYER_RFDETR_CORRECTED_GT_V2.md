# SLAYER-OCR RF-DETR corrected GT v2

Date: 2026-09-30

Status: **pipeline ready; completed human review export still required**

## Goal

Measure the frozen epoch-15 RF-DETR checkpoint against corrected ground truth
without retraining it and without confusing the result with the original
development score.

The original `mAP@50:95 = 0.3272939026` remains a reproduction result for the
original v1 labels. The v2 score will be a separate reanalysis against reviewed
labels.

## Required final review

Open the generated offline review editor, inspect all 12 pages and export the
final JSON. A valid final packet must:

1. use schema `slayer-layout-gt-review-v1` and state `complete`;
2. name the reviewer and include a valid timestamp;
3. cover every frozen validation page exactly once;
4. mark every completed page `verified` or `edited`;
5. preserve source annotation links for retained boxes;
6. use only the nine frozen layout classes and in-bounds boxes.

The importer recomputes the actual annotation difference. The browser's
`changed` flag is retained as an interaction trace, not treated as ground truth,
because it remains set after an edit is reverted. Both `verified` and `edited`
are valid final review statuses.

## Build the corrected dataset

```bash
python -m training.apply_rfdetr_layout_gt_review \
  --review final-review.json \
  --review-source data/slayer-rfdetr-layout-gt-review-v1-20260930 \
  --dataset data/slayer-layout-rfdetr-private-v1-20260929 \
  --output data/slayer-layout-rfdetr-corrected-v2-20260930 \
  --archive data/slayer-layout-rfdetr-corrected-v2-20260930.zip
```

The legacy v1 directory name contains `private`; this is an immutable input
identifier, not the current publication policy.

The output preserves the training split, rewrites only reviewed validation
annotations, copies the exact page images and records:

- original dataset, COCO, audit and checkpoint hashes;
- review packet and review-source hashes;
- reviewer, timestamp and policy version;
- per-page additions, deletions, relabels and box moves;
- a complete file manifest and `checksums.sha256`.

## Reaudit contract

`training.audit_slayer_rfdetr_layout` accepts both dataset schemas:

- v1 becomes `original-gt-reproduction` and must reproduce the logged best mAP;
- v2 becomes `corrected-gt-reanalysis`, must descend from the model's training
  dataset and must point to the same checkpoint used during review.

For v2, the measured mAP is intentionally not compared with `0.3273`; changing
ground truth changes the metric. The checkpoint hash, class order, page split,
collection separation and all archive checksums remain hard requirements.

Run the pinned GPU notebook:

[Open corrected-GT v2 audit in Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_slayer_rfdetr_corrected_gt_v2.ipynb)

It accepts the newly generated dataset ZIP, the existing
`slayer-rfdetr-layout-v1-model.zip` and
`slayer-rfdetr-layout-v1-evidence.zip`. The latter two remain pinned by their
known SHA-256 values; the new dataset hash is computed at upload and recorded in
the resulting audit evidence.

## Publication

After successful reanalysis, publish the corrected dataset and evidence as new
versioned artifacts in the public
[OCR experiment registry](https://huggingface.co/collections/PiotrSty/ocr-experiment-registry-6abca642387af872378a7fee).
Do not overwrite the v1 dataset or its audit evidence.
