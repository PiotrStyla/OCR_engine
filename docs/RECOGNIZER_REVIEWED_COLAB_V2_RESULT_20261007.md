# Reviewed recognizer V2: audited result

## Decision

The three variants completed on Tesla T4. **Retain the unchanged baseline**
`PiotrSty/trocr-pl-mixed-v3@85d0c91c26f8e088849096dded7c9ba10b4cd9c9`.
None of the nine epoch checkpoints or three selected merged candidates passed
the protected-domain gate. No production promotion or SOTA claim.

## Integrity

Returned ZIP: `recognizer-reviewed-colab-v2-evidence.zip`, 853,540 bytes.
SHA256: `a63e0829d97576edb978dc21e47b9724f5f71ab4abf93b88f6a2d11d0b753746`.
All 50 ZIP members have complete checksum coverage: 49 payloads and the manifest.
Runtime: `ee90eb4426232f0a03937787c98291b59ce8398a`.
Config SHA256: `165cbff13fa6f8b12190eca7f60ef27e5e2b0121a241f2bf76d249177a2b88ad`.

Recorded stack: Python 3.13.15, Torch 2.11.0+cu130, TorchAO 0.17.0,
Transformers 4.57.6, PEFT 0.19.1, Accelerate 1.13.0, jiwer 4.0.0.
The audit verifies configuration, raw reference hashes, pair IDs, input lineage,
historical repeats, replay subsets, target-length audit, completed steps/epochs,
all checkpoint scores and final selection. It also verifies the original reviewed
input package and local ordinary development image/text files against the manifests.

This ZIP contains **no model weights**. Weight finiteness, synthetic image payloads
and actual GPU inference were not independently inspected/repeated. Metrics are
recomputed from returned raw predictions, preserving historical spelling with
NFC and whitespace normalization only.

## Paired CER

There are 9 historical and 75 ordinary development lines. Lower CER is better.
These are the final merged-model predictions, not just adapter evaluation.

| Model | Historical | Ordinary | Combined | Decision |
| --- | ---: | ---: | ---: | --- |
| Baseline | 33.0317% | 5.3307% | 8.3727% | Retained |
| LR 1e-5, replay 500 | 31.9005% | 6.1680% | 8.9938% | Rejected |
| LR 3e-6, replay 500 | 31.9005% | 5.5261% | 8.4224% | Rejected |
| LR 3e-6, replay 2000 | 32.1267% | 5.6936% | 8.5963% | Rejected |

Character errors (historical + ordinary): baseline 146 + 191 = 337;
candidates 141 + 221 = 362, 141 + 198 = 339, and 142 + 204 = 346.
The closest candidate saves five historical errors but adds seven ordinary errors.
Its ordinary WER also rises from 22.88% to 23.68%.
Ordinary improved/regressed/equal-CER line counts: 1/18/56, 0/7/68, 0/12/63.
U+FFFD increases from 2 to 3 for the control; the lower-LR variants retain 2.

## Checkpoints And Loss

Each variant completed three epochs: 294, 294 and 855 optimizer steps.
The selected checkpoints are first-epoch `checkpoint-98`, `checkpoint-98`, and
`checkpoint-285`. **0/9 epoch checkpoints passed**, reproducing the baseline fallback.

Full-model preflight reports aligned generation-evaluation loss equality, finite
gradients, 96 updated adapter tensors and successful merge/generation. Its disposable
updates were not used in training. Selected evaluation losses are approximately
1.0066, 1.0289 and 1.0229. These are reported, not independently recomputed here,
and not comparable with V1's old native-loss path. The control reproduces V1's
generated-text scores; correcting evaluation loss did not change its training objective.

## Merge Difference

The first two candidates have identical predictions before/after merge.
Replay 2000 changes one ordinary line, `odezwa-12-17`, adding one character error.
Ordinary CER changes from 5.6656% to 5.6936%; combined CER from 8.5714% to 8.5963%.
The cause is not established by this ZIP. Both versions fail, so the selection
decision is unchanged. Keep checking merged models separately in future runs.

## Next Step

Lower LR reduced regression in this fixed experiment; larger synthetic replay
did not resolve it. This does not prove the general ineffectiveness of synthetic
data or identify the cause. Do not repeat V2 unchanged or replace the baseline.
A next data-focused test should examine separate real ordinary-print training/replay
inputs, never these 75 development lines or protected historical work families.
Repeatedly tuned, nine-line historical development is not independent gold;
untouched work-family and full-page evaluation remain necessary for SOTA claims.

## Reproduce

```bash
python -m training.audit_reviewed_recognizer_v2_evidence \
  --archive recognizer-reviewed-colab-v2-evidence.zip \
  --input-root data/recognizer-reviewed-colab-input-v1-20261006 \
  --output data/recognizer-reviewed-colab-v2-audit-v1-20261007
```

Use a new output directory. The CPU auditor preserves the ZIP and writes
`audit.json`, paired `line-comparison.jsonl`, implementation hashes and checksums.
It needs jiwer, not Torch; it does not load a model.
