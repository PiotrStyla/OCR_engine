# Reviewed recognizer V2: protected ordinary print

## Run One Notebook

[Open Colab V2](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_recognizer_reviewed_training_v2.ipynb).
Select **GPU T4**, then **Run all**. Do not upload files or repeat annotation review.
The setup runs a tiny real TrOCR regression/preflight before downloading and training
the full model. Local CPU checks do not establish that the GPU run succeeds.

The unchanged baseline remains
`PiotrSty/trocr-pl-mixed-v3@85d0c91c26f8e088849096dded7c9ba10b4cd9c9`.
The regressing V1 candidate is **not** the starting point.

## Why V2

[V1's audited result](RECOGNIZER_REVIEWED_COLAB_RESULT_20261006.md) improved the
historical slice but regressed ordinary print and combined CER. Its training loss
and generation-evaluation loss also used different code paths. In the
[pinned Transformers source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/trainer_seq2seq.py),
the generation branch obtains native model loss rather than calling our custom
`compute_loss`. The previous implementation only overrode `compute_loss`.

V2 computes aligned token loss explicitly in that evaluation branch and removes
labels and teacher-forced decoder inputs from generation. Reference text is used
only for loss and scoring. A preflight compares generation-evaluation loss with
the direct aligned loss on the same model and batch. Old V1 loss values are not
comparable with the corrected loss; its audited generated-text CER remains intact.
This is a verified code-path difference, not yet proof that all loss differences
in the returned training run have the same cause.

## Bounded Experiment

Each variant starts from the same frozen baseline and seed 42, trains decoder
attention LoRA (rank 16, alpha 32) for 3 epochs, and repeats the 70 historical
training lines four times. Batch size is 2; accumulation is 4. No 4-bit path,
new labels, new work families, or model-capacity changes are introduced.

| Variant | Learning Rate | Synthetic Replay | Examples / Epoch |
| --- | ---: | ---: | ---: |
| Control | 1e-5 | 500 | 780 |
| Lower LR | 3e-6 | 500 | 780 |
| Lower LR + Replay | 3e-6 | 2000 | 2280 |

The first two differ only in LR; the last two differ only in replay size.
The 500-sample replay is the exact V1 deterministic subset of the 2000-sample
synthetic training pool. Larger replay also increases optimizer steps; this is
not a compute-matched ablation. The control uses the **corrected** evaluation
and checkpoint policy, so it is not an exact replay of V1.

The same frozen reviewed input ZIP is downloaded and verified:
`PiotrSty/slayer-ocr-datasets@3a301fd12a6272b28774d1888213ddf528b47f24`,
SHA256 `3e39b7d02b90a0a741adde506c0467698c46fe86c82846d3cc0e648eebaae32e`.
Its original config checksum, package checksums and synthetic archive checksum
are checked. Exact image duplicates/overlap, counts, target lengths and tokenizer
round-trips fail before training. Raw transcriptions and historical spelling are
retained; scoring applies NFC and whitespace normalization only.

## Selection

The existing 9 historical development lines and 75 ordinary development lines
are never added to training. Every epoch's checkpoint is evaluated against the
same baseline. An eligible checkpoint must satisfy all conditions:

- Strict improvement of historical CER.
- No ordinary CER or WER regression.
- No additional U+FFFD replacement characters in either slice.
- Strict improvement of combined character-weighted CER.

Eligible checkpoints are ranked by combined CER. Ineligible checkpoints receive
a penalty in the trainer's selection score. Each variant's selected checkpoint
is then re-evaluated as a merged full model; paired raw references and identifiers
must match the baseline. The final selector includes **unchanged-baseline**.
If none passes, no new model is selected and no new-weight ZIP is presented as
a successful replacement.

This protects development slices, not unknown ordinary documents. Nine historical
lines from one collection cannot establish generalization. Repeated tuning on
these slices makes a separate untouched work-family/full-page test necessary
before any production or SOTA claim. The notebook does not automatically promote
a model and does not modify the frozen benchmark.

## Download And Retain

The small evidence ZIP downloads after the run, or on failure. It contains pinned
configuration, input manifests, raw predictions, per-checkpoint predictions,
metrics, training logs, trainer state, gate decisions and SHA256 checksums.
Send that ZIP back first.

A complete result ZIP is also created. When a candidate passes, it contains the
merged model ZIP and evidence ZIP. When the baseline wins, it contains evidence
and the exact baseline identity, **not a duplicate download of old weights**.
The other experimental candidates remain in the Colab session; they are not
included in the selected model package.

Large browser downloads failed in V1. To retain the full result, set
`BACKUP_TO_DRIVE = True` in the last cell and run **only that cell**. Authorize
Drive mounting, then obtain the ZIP from `MyDrive/OCR_engine/<run>/`. The copy is
SHA256-checked; a different existing file is not overwritten. Do this before
the session ends. No Drive authorization is requested when the option is false.

## Validation Status

135 focused CPU tests passed (selection, input gates, splits, audit and notebook
structure); one Torch-dependent module was skipped. Torch is not installed
in the local test runtime, so model-dependent tests are deferred to the Colab
setup. GPU training and the new experimental results are not yet verified.
