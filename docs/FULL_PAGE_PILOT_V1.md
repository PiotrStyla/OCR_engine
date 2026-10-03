# Full-page OCR pilot v1

Date: 2026-10-02. This is a development experiment, not a released gold benchmark.

## Current scope

The existing pinned IMPACT HF corpus provides 65 train and 15 validation pages
outside the three historical test collections. All 80 are candidates for
full-page review. The target of 100 reviewed pages requires at least 20 more
pages from additional sources; there are currently zero double-verified,
complete gold pages in this new protocol. Contemporary Polish documents are
a separate next track, not covered by this historical pilot.

The local source audit is complete: [verified counts and limitations](FULL_PAGE_REFERENCE_AUDIT_20261002.md).

Some train pages may have been seen by mixed-v3. Previous validation pages have
also been inspected during development. Neither split is an independent final
test. Collection exclusion is enforced, but near-duplicate and pretrained-model
contamination audits are not complete.

## Run one notebook

[Open the current notebook in Colab](https://colab.research.google.com/github/PiotrStyla/OCR_engine/blob/main/training/colab_full_page_pilot_v1.ipynb), choose GPU and Run all.
There are no data uploads, authentication requirements or model-selector fields.
The default run processes two deterministically selected validation pages.

2026-10-03 bootstrap fix: model venvs are created with `with_pip=False` and
the existing host pip installs into the target interpreter using `--python`.
This avoids Colab's failing `ensurepip` bootstrap. The target interpreter's
prefix is checked before installation. Recreating a partial venv preserves its
files; neither data nor the notebook's Transformers installation is removed.
After an `ensurepip` error, open the current notebook and Run all, or replace
the old environment cell with the current section 4 and rerun from there.
Host pip >= 22.3 is required. See the [official pip guidance](https://pip.pypa.io/en/stable/topics/python-option/).

## Published inputs and audit

- [All 80 candidate pages and source XML on HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/tree/5f4cc0c677b7eb38be55695e67cd13df3329ecb4/data/full-page-pilot-review-v1-20261002).
- [Code-free source-reference audit on HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/8370d9ad540556a072dcfd0bcb0e7d94bc34986b/experiments/2026-10-02/full-page-reference-audit-v1).

The publication is public. All 167 expected dataset paths were checked;
the manifest and one downloaded image/XML pair passed SHA-256 verification,
and the downloaded PNG decoded successfully. This is sample download validation,
not a fresh download of every scan. All local source images/XML passed the staging audit.
The 46 focused tests and notebook schema validation passed, including real local
installation into a pip-less venv and repeat installation after partial creation.
The local tests simulate unavailable ensurepip; they are not a Colab GPU run.
GPU inference for
the new Ovis adapter remains pending; no measured quality gain is claimed.

## Model environments

The first model environment uses mixed-v3 with Transformers 4.57.6. It produces
three separate predictions: automatic row-major, automatic column ordering and
diagnostic source-PAGE region crops with automatic line segmentation inside each
region. No deskew or lexical correction is applied. The real source XML has no
line geometry on these 80 pages. The PAGE region geometry is not independently
reviewed; annotation-assisted input does not mean proven perfect segmentation.

The second environment uses pinned OvisOCR2 with Transformers 5.18.0 and SDPA.
It receives the native full-page image and a fixed transcription prompt.
This portability adapter has not yet been executed on GPU. The official model
card demonstrates vLLM; this notebook intentionally avoids that larger runtime
for the initial Colab pilot. Failure is preserved as evidence, never silently
replaced by a different model or backend.

Model documentation: <https://huggingface.co/ATH-MaaS/OvisOCR2>.

The base OCR code comes from public commit
`ad8c3908a41949719a29ea259c5d286e078f788f`. The new runner is embedded in the
notebook with its own SHA-256, so the notebook does not fetch an unpublished
commit. Data and both model revisions are pinned in
`experiments/2026-10-02/full-page-pilot-v1/config.json`.

## Evidence and failure handling

Model subprocesses receive `inference-inputs.jsonl`, which contains only image
identity, dimensions and optional source region geometry. Reference transcripts and
PAGE XML are not passed to them. Model environments are separate venvs; the
notebook's Transformers installation is never replaced.

Each completed page is flushed to JSONL. Repeating inference resumes missing
records and retains previous errors. A different configuration, page selection
or runner requires a new output directory. Use a new WORK to retry errors.

The final download cell can also package a failed or partial run. The ZIP
contains raw outputs, provisional scores, reference audit, input manifests,
package versions, errors, code provenance and SHA-256. No scans or weights.
Token-limit exhaustion is counted explicitly. Missing/error predictions are
scored as empty text, not excluded from averages.

CER/WER use the existing evaluator: NFC and whitespace only. Ovis Markdown has
a separate, recorded CommonMark-to-visible-text projection; the raw answer is
retained without spelling repair or repeat removal. Oracle scores never appear
as an automatic full-page result. Engine/page latency includes all variants for
that engine and is not a per-variant throughput benchmark.

No model promotion is permitted by this pilot, even if a provisional CER is
lower. Formal reading-order, table structure, hallucination rates and confidence
intervals await a complete reviewed protocol.

## Local reference review

```bash
python -m training.full_page_pilot stage \
  --config experiments/2026-10-02/full-page-pilot-v1/config.json \
  --output data/full-page-pilot-review-v1
python -m training.build_annotation_review \
  --manifest data/full-page-pilot-review-v1/manifest.jsonl \
  --output data/full-page-pilot-review-v1/review
```

The existing editor preserves the original text and exports an append-only
review history. For this task, a verified decision must mean checking the whole
page: every text block, line, marginal note included by policy, historical glyph
and reading order. A clean Unicode audit is not evidence of completeness.
Keep uncertain fragments pending; do not replace private-use characters by a
guess or modernize `á`/`ſ`. Write corrections in the transcription field, not
the notes field.

Two independent reviews can be reconciled with `training.adjudicate_reviews`.
Its names are self-reported, so independence must also be established outside
the tool. Agreement alone does not turn the manifest into an official gold
release: completeness, layout consistency, lineage and split review remain
required. Never overwrite source transcriptions with model outputs.

## Following gates

1. Execute the two-page GPU smoke and audit its outputs and runtime.
2. Repair and independently verify full-page references, beginning with flagged pages.
3. Add 20 or more pages and the contemporary document track; inspect exact and near duplicates.
4. Compare full supported PaddleOCR-VL and TeleOCR pipelines on the same reviewed pages.
5. Select a student and expand real/pseudo/synthetic training data based on measured failure strata.
6. Freeze a new document-disjoint test before tuning or further training.
