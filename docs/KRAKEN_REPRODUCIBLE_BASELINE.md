# Kraken full-page baseline and submission validation

This runner reuses the Kraken 7 task API already used in the project's GPU
notebooks. It is separate from the older library backend and retains every
page prediction instead of only aggregate scores. No model or dependency is
downloaded implicitly by the runner. Reference text is excluded from the
inference callback.

## Local preflight (no ML dependencies)

```bash
python -m training.run_kraken_benchmark --manifest data/polocrbench-history-png-v1/manifest.jsonl --output validation/kraken-preflight
```

This verifies all image hashes and IDs and writes `preflight.json`. It does not
import Kraken, PyTorch or Pillow and is not a model inference test. Use a new
directory for each preflight/run. Generate the portable PNG manifest using
[the reproduction instructions](POLOCRBENCH_REPRODUCTION.md).

## Explicit GPU run

Use an available remote CUDA environment, such as a user-started Kaggle T4
session. The commands below do not provision a machine or purchase compute.
Run the repository code from the same commit for all compared systems.

```bash
python -m pip install kraken==7.1.1 jiwer==4.0.0 huggingface_hub
```

Example candidate: EHRI-finetuned recognizer and EHRI-finetuned segmenter from
the same pinned dataset revision. This is **not** the historical default-BLLA
segmenter configuration. Its suitability for IMPACT print has not been measured
by this runner yet.

```python
from huggingface_hub import hf_hub_download

revision = "1947d4c107936ed59ded16446a77b6587fa3874a"
for filename in ("models/polish_nfd_finetuned.safetensors",
                 "models/polish_seg_best.safetensors"):
    hf_hub_download("PiotrSty/ehri-dataset", filename, repo_type="dataset",
                    revision=revision, local_dir="data/kraken-ehri")
```

Verified HF LFS hashes for that revision:

- Recognizer: `9b731f0694af3c0c4c6f31a689fc4f2788292bab31eedaad85184698ee8f3ff6`.
- Segmenter: `a817f2a1d3a0eee51b456008ba9952afe60b2ade712f2e7d75ec1608d008ea86`.

Linux/Kaggle command:

```bash
python -m training.run_kraken_benchmark \
  --manifest data/polocrbench-history-png-v1/manifest.jsonl \
  --output validation/kraken-ehri-fullpage-v1 --execute \
  --recognizer data/kraken-ehri/models/polish_nfd_finetuned.safetensors \
  --recognizer-sha256 9b731f0694af3c0c4c6f31a689fc4f2788292bab31eedaad85184698ee8f3ff6 \
  --segmenter data/kraken-ehri/models/polish_seg_best.safetensors \
  --segmenter-sha256 a817f2a1d3a0eee51b456008ba9952afe60b2ade712f2e7d75ec1608d008ea86
```

Both local model hashes must match before model loading. CUDA is mandatory:
the runner never silently falls back to laptop CPU inference. It checks a real
CUDA operation, selects frame zero and grayscale L, performs no deskew,
binarization or resize, and retains Kraken's native line order. Deterministic
algorithms are requested; unsupported operations become explicit errors rather
than an unreported switch. Cross-hardware bitwise reproducibility is not promised.

`predictions.jsonl` is flushed after every page. Errors have empty text and only
the exception type; blank successful reads remain valid output for scoring.
`run.json` records progress, model/manifest/runner hashes, code commit, library
versions, the installed distribution inventory, configuration representations,
CUDA device, timing and run completion status. A caught interruption leaves
partial evidence. A killed process may leave state `running`; do not interpret
that as completion. Resume/overwrite is deliberately unsupported.

## Validate and score

```bash
python -m training.validate_submission --manifest data/polocrbench-history-png-v1/manifest.jsonl --predictions validation/kraken-ehri-fullpage-v1/predictions.jsonl --output validation/kraken-ehri-fullpage-v1/submission-validation.json
python -m training.transcription_eval --manifest data/polocrbench-history-png-v1/manifest.jsonl --predictions validation/kraken-ehri-fullpage-v1/predictions.jsonl --output validation/kraken-ehri-fullpage-v1/metrics.json
```

Validation rejects duplicate JSON keys/IDs, unknown pages, invalid status/text,
non-finite JSON constants, negative/non-numeric times, nonempty error outputs and
missing pages. For diagnostic partial runs use `--allow-missing`; the report
still says `complete: false`. Every missing page must remain in the scorer's
denominator. Optional extra fields are allowed for existing provenance records.
Provided `source_sha256` or `sha256` fields must match the manifest image hash.
The validator neither verifies track eligibility nor implements AmuEval packaging.

The current 36-page references retain their known Unicode annotation issues.
Do not tune against the held-out test or claim SOTA from this small subset.

## Verification scope

Local tests exercise injected recognizers, failure and interruption persistence,
image mutation detection, reference exclusion, strict prediction validation and
preflight under Python `-S`. Real Kraken/CUDA execution is still required before
claiming a working GPU baseline or reporting a Kraken score. In particular, the
weights, environment and task configuration must be smoke-tested remotely first.

API reference: https://kraken.re/7.1/api/reference.html
Pinned weights: https://huggingface.co/datasets/PiotrSty/ehri-dataset/tree/1947d4c107936ed59ded16446a77b6587fa3874a/models
