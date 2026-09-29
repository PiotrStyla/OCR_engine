# SLAYER-OCR layout consensus: 60-page result

Date: 2026-09-29

Status: **reviewed private training candidate, not published as data**

## Inputs

All three archives passed safe ZIP extraction, embedded checksum validation,
frozen dataset/configuration checks, teacher revision checks and page-identity
checks.

| Teacher | Pages completed | Errors / abstentions | Accepted detections | Unmapped | Archive SHA-256 |
|---|---:|---:|---:|---:|---|
| Qwen3-VL 4B | 56/60 | 4 | 231 | 1 | `7224fb751c95ef335abe0068225b2b00f35e77237fd8d592ebf2c1798252946a` |
| DocLayout-YOLO | 60/60 | 0 | 218 | 34 | `9c330845c950bc1f663266f61b165ef7a9d78240888003b1500abbc1c3608bfc` |
| Surya Layout | 60/60 | 0 | 298 | 76 | `6304bc94d56ed8c7d4f472776c6d875d3ca8dfe402b7ebdfe67eba559d95ec9b` |

The four Qwen failures are explicit abstentions under consensus policy v3.
They do not count as empty or negative votes. Every affected page is marked as
a hard example, and the quorum remains two independent teachers.

## Consensus output

- Pages: **60**
- Teacher proposal rows: **180**
- Canonical teacher detections: **747**
- Accepted weak-label objects: **161**
- Objects requiring review: **371**
- Hard-example pages: **58/60**
- Teacher abstentions: **4 on 4 pages**
- Evidence archive SHA-256:
  `454ac2a80746f0c93e4c368f214338d17819e61d947722d1f24c3bef472e6f84`

The private evidence archive contains 36 members and six verified consensus
checksums. It contains no scan images or reference transcriptions.

## Decision

The execution and provenance gates pass, but the output is **not a clean
training dataset**. With 371 review objects and 58 hard-example pages, teacher
agreement is too low to train the RF-DETR layout model without human
adjudication.

Next gate:

1. Build the private image-backed layout review panel.
2. Review every queued object as `accept`, `reject` or `relabel` against the
   frozen ontology.
3. Export a versioned clean candidate through
   `training.adjudicate_layout_consensus`.
4. Measure class balance and held-out layout quality before RF-DETR training.

This result demonstrates a reproducible weak-supervision pipeline. It is not
human ground truth, OCR accuracy evidence or a SOTA claim.

## Adjudication and RF-DETR package update

The complete 371-item queue was reviewed by one named reviewer. The sealed
adjudication retained 143 review objects, including five relabels, rejected 228,
and combined them with 161 consensus objects. The resulting private candidate
contains 304 objects on all 60 pages and passes the frozen geometry-conflict
gate.

`training.build_layout_rfdetr_dataset` materializes the verified scans into the
RF-DETR COCO directory contract. The deterministic, collection-disjoint split
contains 48 train pages from 18 collections and 12 internal-development pages
from four collections. Exact image SHA-256 overlap and collection overlap are
both zero. The package remains private and the internal `valid` split is not the
final PolOCRBench test.

Rare classes remain a material limitation: `header` has one object and `footer`
has two, all in train; `table` has only three objects in total. Results for these
classes cannot support reliable conclusions. The next gate is a pinned RF-DETR
pilot followed by full-page OCR A/B, not a SOTA claim.
