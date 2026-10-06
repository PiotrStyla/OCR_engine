# Reviewed recognizer: completed training, candidate not promoted

## Decision

Keep `PiotrSty/trocr-pl-mixed-v3` at revision
`85d0c91c26f8e088849096dded7c9ba10b4cd9c9` as the baseline. Preserve the new
weights as an experimental candidate only. Training now works, but this run
does not establish a better general OCR model or a SOTA result.

The user returned `recognizer-reviewed-colab-v1-result.zip` from run
`recognizer-reviewed-colab-v1-20261006T182240529879Z`.

## Recomputed Results

CER is character error rate; lower is better. Raw reference spelling is retained.
Only NFC and whitespace normalization are applied, without `s`/long-s,
diacritic, historical spelling or punctuation substitutions.

| Development domain | Lines | Baseline CER | Candidate CER | Change |
| --- | ---: | ---: | ---: | ---: |
| Historical text | 9 | 33.0317% | 31.9005% | -1.1312 pp |
| Ordinary print | 75 | 5.3307% | 6.1680% | +0.8373 pp |
| Combined, character-weighted | 84 | 8.3727% | 8.9938% | +0.6211 pp |

Historical character errors decrease from 146 to 141: four lines improve, one
regresses, and four retain the same CER (not necessarily identical predictions).
Ordinary-print errors increase from 191 to 221: one line improves, 18 regress,
and 56 retain the same CER. Ordinary predictions contain three replacement
characters U+FFFD versus two in the baseline.

The candidate's combined strict CER reproduces the trainer's selected CER
exactly. The best checkpoint is `checkpoint-98`, after the first epoch. Later
epoch CER values in the training log are worse. Selection only compared trained
checkpoints; it did not include the unchanged baseline as a selectable candidate.

## Integrity And Training

- Outer ZIP SHA-256:
  `7c97b6c99687863368c31bbd40822d9bd3d69493c2fd2c7c1ab02860426efe2c`.
- Model ZIP SHA-256:
  `3a688ce473b3eeebc0f37959e5863fc972390df4bac8857f49b9bad1020adceb`.
- Evidence ZIP SHA-256:
  `ddd1038c71e2a3ad5907f06bff90c8896c56c3cf93e10d762d606bcf66e13850`.
- Complete checksum coverage: 3 outer members, 19 evidence members and 14
  model members; all checksummed payloads verified byte-for-byte.
- Full merged safetensors, not adapter-only weights: 480 tensors and
  333,921,792 float32 parameters. Header offsets/shapes/payload size are valid;
  every weight is finite. No pickle weights need to be loaded for this audit.
- Model/evidence run, selection and best-metrics files match exactly.
- Three epochs and 294 optimizer steps on a Tesla T4. Each epoch contains
  70 reviewed historical lines repeated four times plus 500 synthetic lines.
- Input IDs, text/image hashes, repetitions and development provenance match
  the saved training manifests. Exact train/development image overlap is absent.
- Recipe parameters and corpus configuration match the frozen protocol.
- The disposable adapter preflight passed and recorded 96 updated adapter
  tensors; these preflight updates were not used in the training run.
- Runtime code: `fe074fee1da33a94ea60a0bcd658a8f6afbb7dc8`; Torch 2.11.0+cu130,
  TorchAO 0.17.0, Transformers 4.57.6, PEFT 0.19.1.

## Interpretation

This establishes training execution and artifact integrity, not independent
generalization. There are only nine historical development lines, from one
collection, and the same development data selected the checkpoint. The labels
retain their exploratory status; original frozen review gates are unchanged.
Line crops do not measure page segmentation, reading order, tables or PDF output.
The laptop audit did not reload the full model or repeat GPU inference.

Trainer `eval_loss` is approximately 16.88, while the logged training loss is
approximately 2.29. Do not use this discrepancy as a quality measure or as
proof of a particular failure: aligned evaluation-loss behavior needs separate
verification. Checkpoint selection and the decision above use reproduced CER,
not evaluation loss.

## Next Experiment Requirements

1. Include the unchanged baseline in checkpoint selection. A completed training
   run need not yield a replacement model.
2. Measure both domains at every checkpoint; prevent ordinary-print regression
   instead of accepting a historical improvement alone.
3. Test a bounded adaptation/replay ablation, with frozen text and glyph policy,
   before committing to more training. Lower adaptation strength and a larger
   ordinary-text replay share are hypotheses, not guaranteed fixes.
4. Evaluate any promising candidate on additional untouched work families and
   full pages, without recycling the nine development lines into a final test.
   Keep dataset expansion, work-family independence and SOTA claims separate.

No additional human review is requested by this result audit. No baseline model
was overwritten, and no candidate was automatically promoted.

## Reproduction

Run the lightweight auditor from the repository root with the CPU tooling
dependencies (`jiwer` and `numpy`), using a fresh output directory:

```powershell
python -m training.audit_reviewed_recognizer_result --archive "C:/Users/Hipek/OneDrive/Pulpit/recognizer-reviewed-colab-v1-result.zip" --output data/recognizer-reviewed-colab-result-audit-fresh
```

The auditor streams hashes and weight checks without loading Torch or running
inference. It preserves the source archive and refuses to overwrite an existing
audit. Outputs include the original evidence ZIP, paired raw predictions,
recomputed metrics, trainer state and checksums. Focused validation: 79 passed;
one Torch-dependent test module was skipped in the local CPU environment.
