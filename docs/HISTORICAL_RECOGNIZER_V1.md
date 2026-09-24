# Historical recognizer v1

This experiment specializes `PiotrSty/trocr-pl-mixed-v3` for historical Polish
print while preserving source spelling. It is a bounded development pilot, not
a benchmark run or SOTA claim.

## Corpus construction

Source: `PiotrSty/impact-psnc-polish-ocr` at revision
`c7cb156fb95d2880699c33725bbaf1fbc1008fea`.

`training.prepare_historical_line_corpus` downloads pinned region metadata and
images, verifies their SHA-256 hashes, detects line rectangles without reading
the reference count, and accepts a region only when the detected and reference
line counts agree. Top-to-bottom crops are then paired with reference lines.

The split is collection-disjoint:

- train: 10 collections not used by the geometry holdout;
- validation: the five upstream validation collections;
- geometry holdout: 12 reserved development collections, excluded from train;
- final test: `NA2_FT`, `Nowiny_z_Rakuz_FT` and `Powodzenia_FT`, untouched.

The local deterministic build produced:

| Split | Candidate regions | Accepted regions | Accepted lines | Count mismatch regions |
| --- | ---: | ---: | ---: | ---: |
| Train | 188 | 94 | 258 | 87 |
| Validation | 116 | 69 | 139 | 42 |

Lines containing U+FFFD or a private-use code point are quarantined rather than
silently changed: 212 train lines and 68 validation lines. Historical Unicode
outside the private-use range remains unchanged. Across accepted labels the
inventory includes 478 `á`, 228 long `ſ`, 93 `ɇ`, 115 `ⱥ`, 128 `ß` and several
historical ligatures.

The source references are upstream and not manually adjudicated. Matching line
counts do not prove perfect crop geometry. Visual spot checks show readable
target lines with occasional neighbouring fragments. The rectangle variant is
used because the independent geometry holdout did not support line-band.

## Training pilot

- Base: `PiotrSty/trocr-pl-mixed-v3`, pinned revision
  `85d0c91c26f8e088849096dded7c9ba10b4cd9c9`.
- Train: 258 historical lines only; no holdout or test collection.
- Validation: 139 historical lines from five separate collections.
- LoRA: decoder `q_proj`, `k_proj`, `v_proj` and `out_proj`, rank 16, alpha 32.
- Full precision base load with fp16 training; no 4-bit path in this pilot.
- Epochs: 12; batch 4; gradient accumulation 2; learning rate `5e-5`.
- Best checkpoint: lowest validation CER.
- Target length: 128 tokens, with mandatory preflight and a hard error instead
  of silent truncation.

The Colab notebook must also verify exact tokenizer round-trip for every label
before training. Primary text normalization is NFC plus whitespace only. It may
not modernize `á`, `ſ`, ligatures or other historical forms.

[Open the pinned training notebook in Google Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_historical_recognizer_v1.ipynb).

## Promotion gates

The candidate remains experimental unless all of these hold:

1. Historical validation CER improves over the pinned mixed-v3 baseline.
2. No label is truncated and tokenizer round-trip mismatches equal zero.
3. CER regression on `benchmarks/real-lines-v1` is no more than 1 percentage
   point.
4. CER regression on the frozen EHRI test is no more than 2 percentage points.
5. The evidence ZIP is complete and its metric inputs are reproducible.

Passing this pilot still does not establish SOTA. Full-page evaluation and the
final frozen PolOCRBench test remain separate.

## Privacy and publication

The notebook downloads public CC-BY-3.0 inputs but does not republish scans,
line crops, labels or raw OCR predictions. Its evidence ZIP contains aggregate
metrics, hashes, configuration and provenance only. Model publication is a
separate, explicit decision after reviewing the evidence.
