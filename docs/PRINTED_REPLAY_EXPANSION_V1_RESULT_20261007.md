# Printed replay expansion V1: audited return and geometry bottleneck

## Returned Run

The CPU Colab experiment completed: 34 native pages from eight work families,
1,387 reconstructed lines and 192 image/text pairs. The untouched evidence ZIP
has 94,678,979 bytes and SHA256
`5e32ce9e545f23143ef4f61cedc804e414f5c3e60e6fe23833641702bb453732`.
All 741 payload checksums and complete archive coverage were verified.
Runtime code is pinned to `c8ae68ae741d2c30d95317c2c0bb84d347d79a16`;
the source checksums digest is
`53e3c81ec06e7a1c75f0b3a3553d97252716a5ccfe495e510b61d540f030b5ad`.

[Original ZIP, machine audit, visual sample and geometry diagnostic on HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/610f8a23d931a012088ad20453eaa1a1783fd1e2/experiments/2026-10-07/printed-replay-expansion-v1-result).

All 17 published files were downloaded without authentication at this immutable
HF revision and independently checked for byte size and SHA256.

## Recomputed Results

| Work family | Split | Pairs |
| --- | --- | ---: |
| Schneider: Babiagora | candidate | 23 |
| Witkiewicz: Teatr | candidate | 22 |
| May: Nad Rio de la Plata | candidate | 38 |
| Pamietniki lekarzy | candidate | 12 |
| Prokesch: Nowa Reforma | candidate | 25 |
| Zawadzki: Kurjer | candidate | 7 |
| Andersen: Basnie | probe, excluded from training | 22 |
| Boccaccio: Dekameron | probe, excluded from training | 43 |
| Total | 127 candidate + 65 probe | 192 |

Every accepted crop is pixel-identical to its padded region in the returned
native PNG. The audit recomputed all accepted and rejected records from full
word TSV, including source spans, minimum confidence, text bytes, geometry,
provenance and work-level splits. No local original-DjVu decoding or Tesseract
rerun was performed. Recorded environment: Tesseract 5.3.4, DjVuLibre 3.5.28,
Pillow 11.3.0.

Of 1,195 rejected lines, 745 fail minimum word confidence, 288 lack a unique
exact source anchor, 63 are too short and 99 fail the page geometry/order gate.
Historical spelling is preserved; matching uses NFC and whitespace only.

## Visual Sample

An AI-agent inspected 31 of 192 accepted crops and their native-page contexts:
the lowest-confidence accepted pair on every positive page, plus the longest
pair per family, deduplicated. This is a targeted sample, not a random estimate
of label error and not human annotation. No obvious text/image mismatch or
major neighboring-line intrusion was observed in this sample. Tight margins,
blur and degraded printing remain risks for uninspected pairs.

`andersen-basnie-1929-0265-line-011` is a table-of-contents entry including page
number 103, not a prose sentence. Work-domain labels are bibliographic metadata,
not page-content ground truth. The May volume also includes publisher backmatter
on pages 0557 and 0563. Preserve these findings rather than calling every row
fiction or ordinary prose. Source manifests remain unchanged.

## Geometry Diagnosis

The 99 exact anchors belong to eight pages. There are **27 adjacent bounding-box
vertical-extent conflicts, with overlaps of 1-15 pixels, and zero source-order
conflicts**. The current all-or-nothing page gate discards every otherwise valid
anchor on a page after any such conflict.

Two diagnostic sheets show the first conflicting pair on each affected page.
Several lines are visibly tilted; rectangular extents overlap even though the
line text is distinct. Some rectangles also reach neighboring ink. These
observations motivate geometry work; they do not establish that all 99 crops
would be safe. No rejected pair was rescued or relabeled.

## Decision And Next Experiment

Keep the baseline unchanged. Keep 65 probe pairs and their two entire work
families outside training. The 127 candidate pairs are automatic exact-agreement
proposals, not human-verified gold. Together with the previous pilot there are
152 real-print candidate pairs across seven families, still insufficient for a
SOTA training claim or a representative modern-document corpus.

The next bounded experiment should compare unchanged native segmentation with
deskew/line-geometry handling on the eight affected pages. Freeze input hashes,
record transformations and coordinate mappings, inspect crop boundaries against
all neighboring lines, and preserve exact source matching and confidence gates.
Do not simply disable overlap checks or lower confidence. Distinguish prose,
contents and publisher backmatter before building a replay training dataset.
Measure usable yield and independently validate agreement labels before choosing
a training budget. No further manual review or unchanged Colab rerun is required
from the user at this stage.

The expansion V1 notebook is now archival, not the next notebook to run.
This result is data-engine evidence, not new model weights or a SOTA benchmark.

## Reproduce

```powershell
python -m training.audit_printed_replay_expansion_evidence `
  --archive "C:/Users/Hipek/OneDrive/Pulpit/printed-replay-expansion-v1-evidence.zip" `
  --output data/printed-replay-expansion-v1-audit-reproduced
```

Use a new output directory. The auditor validates a trusted Git-pinned work
selection, extracts verified payloads, recomputes the run, and emits sample and
conflict context sheets. Its visual status deliberately stays pending;
`visual-review.json` is the separate agent inspection receipt. Focused regression
suite: **71 CPU tests passed**, including trusted pins, deterministic sampling,
geometry/order separation and existing baseline protections.
The generalized auditor also re-audited the previous real V2 archive: all
26 pairs and the full visual selection remained correct.

Original scan licenses and pinned transcription attribution are retained inside
the source package. Wikisource pagequality=4 is not project human review;
upstream model pretraining exposure is unknown.
