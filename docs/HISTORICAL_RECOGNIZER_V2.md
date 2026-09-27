# Historical recognizer v2: frozen replay protocol

Status: frozen before implementation and training on 2026-09-27.

Run the pinned [Colab notebook](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_historical_recognizer_v2.ipynb)
with a GPU runtime and **Run all**. Download the evidence ZIP at the end; the
notebook does not publish a model.

V1 improved historical validation CER by 7.66 percentage points but regressed
`real-lines-v1` by 3.99 points and EHRI test by 11.21 points. V2 tests whether
training-domain replay can retain that historical gain without catastrophic
forgetting. It is one fixed experiment, not a hyperparameter search.

## Inputs

All sources are pinned and only training splits enter optimization.

| Domain | Source | Revision | Training exposure |
| --- | --- | --- | ---: |
| Historical print | `PiotrSty/impact-psnc-polish-ocr` | `c7cb156fb95d2880699c33725bbaf1fbc1008fea` | 258 lines repeated twice = 516 |
| Typewritten Polish | `PiotrSty/ehri-pl-lines` | `3003e8614b74a351e7d94aba4f1348368815fb70` | 349 train lines |
| Synthetic print | `PiotrSty/ocr-pl-lines` | `d881debb90045fd71ad8e25faeeafeb6adab6622` | deterministic 500 of 2000 train lines |

The synthetic sample is selected by sorting IDs with SHA-256 of
`historical-replay-v2:<id>` and taking the first 500. The runtime evidence must
record selected IDs, image/text hashes and source archive hashes. No
augmentation is used.

Distinct training lines: 1107. Effective examples per epoch after intentional
historical oversampling: 1365.

## Training

- base: `PiotrSty/trocr-pl-mixed-v3` at
  `85d0c91c26f8e088849096dded7c9ba10b4cd9c9`;
- LoRA decoder attention: rank 16, alpha 32, no MLP adapters;
- full-precision base load with fp16 training, no 4-bit path;
- epochs: 4;
- batch size: 4, gradient accumulation: 2;
- learning rate: `2e-5`;
- target length: 128;
- seed: 42;
- checkpoint selection: lowest historical-validation CER.

V2 uses fewer historical exposures and a lower learning rate than v1. This is
fixed before the run; results may not be used to silently alter the recipe and
rerun against the same guardrails.

## Evaluation and gates

Evaluation uses the same pinned base and the same three sets as v1:

1. historical validation: 139 lines from collection-disjoint sources;
2. `benchmarks/real-lines-v1`: 75 lines;
3. EHRI test: 81 lines.

Before training, exact image-hash overlap between each distinct training line
and every evaluation set must equal zero. The final PolOCRBench collections and
the 12 geometry-holdout collections remain excluded.

Promotion gates are unchanged:

1. historical validation CER improves over mixed-v3;
2. `real-lines-v1` CER regression is at most 1 percentage point;
3. EHRI test CER regression is at most 2 percentage points;
4. tokenizer round-trip mismatches equal zero;
5. no target is truncated;
6. evidence checksums and input provenance are complete.

The model is packaged only when every gate passes. Publication remains a
separate explicit decision.
