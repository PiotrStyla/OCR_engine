# Recognizer expansion v1: completed teacher run, review required

## Observed Result

The returned `recognizer-data-v3-expansion-v1-evidence (1).zip` passed the audit
against the exact frozen config and published inference code
`0057272a158dfa9941be49678007c0632e02bb40`. It contains the intended NEW batch,
not the previously reviewed 64 roots or their replacement crops.
Both workers report Tesla T4 execution:

| Worker | Returned | Healthy EOS | Missing/error/capped |
| --- | ---: | ---: | ---: |
| Qwen3-VL-4B | 64/64 | 64 | 0 |
| TrOCR mixed-v3 | 64/64 | 64 | 0 |

All **64 cases are teacher disagreements** under diplomatic NFC/whitespace-only
comparison. There are zero agreements and zero abstentions. This is successful
inference execution, **not transcription accuracy or clean labels**.
No model was trained. No automatic label is accepted and no SOTA claim is made.
The previously accepted pool remains 64 pairs until new human decisions are imported.

The output glyph counts are Qwen: `ſ` 1, `á` 23, `ɇ` 0; TrOCR: `ſ` 0,
`á` 3, `ɇ` 0. These are counts of emitted characters, **not recall or accuracy**:
the input references are unverified, and no independent gold exists for this run.
They reinforce why neither output should silently replace diplomatic source text.

## Audit Evidence

- All 22 evidence members and exact checksum coverage verified.
- Frozen config, Git-blob runner hashes, dependency preflight, worker identities,
  GPU/environment reports and reference-free input hashes matched.
- All generation traces checked: token counts, EOS termination, caps and
  Qwen/TrOCR input geometry matched the frozen settings.
- Raw predictions independently recombined; returned proposals/report reproduced.
- A second fresh audit reproduced `audit.json`, glyph diagnostics and both
  recomputed files byte-for-byte.
- 231 focused CPU regression tests passed. No model inference ran locally.

Original returned ZIP SHA-256:
`4d17eac7f389763f9d66b5ed4e52e120657f134b7e5fd9098343becfc3d70dca`.
Recomputed proposals SHA-256:
`6744f037000b0d4d77052d5578b7d9793b56b3d82381715a7a834f2cdbe79d30`.
Model/version claims refer to the verified recorded run, not arbitrary current
package releases. Remote execution evidence is checked; it is not independent
hardware attestation or independent verification of the pretrained weight bytes.

## New Human Review

The offline editor contains **64 new line crops and 34 unchanged source-region
JPEGs**, source labels and both raw model outputs. Every crop and region digest
was verified. Source spelling and pixels are unchanged; all decisions start
unreviewed. No prior accepted decisions are copied into this batch.

Review manifest SHA-256:
`31debfc142b523bf7786b42ee9d53c8fcd3218cf09fe99fa97967e448fb4627c`.
Expected export: **`slayer-recognizer-line-review-31debfc142b5.json`**.

1. Inspect the SMALL line crop first; use the full region to resolve glyphs.
   Check that all glyphs of the intended physical line are present and no
   neighboring line is included. A printed hyphenated word ending at the real
   line boundary is normal; a crop cutting off a glyph is not.
2. Enter only the text visible in that crop in **Transkrypcja**, not **Uwagi**.
   Preserve `ſ`, `á`, `ɇ`, capitalization, punctuation and historical spelling;
   do not modernize or invent missing fragments. The editable source label is
   also unverified, not automatically correct because it looks plausible.
3. If both text and geometry are confirmed, choose **Cała jedna linia** and
   **Zweryfikowano wzrokowo**, then **Zapisz decyzję**. If geometry is wrong,
   choose **Odrzuć wycinek** and explain the clipping/extra line in **Uwagi**.
   Leave genuinely ambiguous glyphs unresolved rather than guessing.
4. Export history and return the named JSON. No further GPU run is needed now.

These rules apply to all 64 cases: model disagreement is not a reason to exclude
an inconvenient example from the dataset or mark either output correct by default.
Accepted new pairs can later extend the pool; rejected geometry requires a new
crop identity and explicit review before admission.

## UI And Publication Checks

Isolated Playwright/Edge validation used 1366x900 and 390x844 viewports.
All 64 crops and their contexts rendered; no console errors or horizontal
mobile overflow occurred. The geometry gate, export manifest/context hashes and
reload persistence worked. QA decisions were simulated in a separate temporary
browser context and are excluded from datasets and evidence publications.
The user's browser storage was not accessed.

Code/UI remain on GitHub. Public HF packages contain source scans, manifests,
diagnostics and evidence, not HTML/JS/Python or weights. IMPACT/PSNC scans retain
CC-BY-3.0 attribution via `PiotrSty/impact-psnc-polish-ocr` at
`c7cb156fb95d2880699c33725bbaf1fbc1008fea`.

After extracting the code-free review ZIP, reconstruct its editor with GitHub
code using a NEW review output directory:

```powershell
python -m training.build_annotation_review `
  --manifest input/manifest.jsonl --output review `
  --diagnostics diagnostics.json --contexts contexts.json --geometry-review
```

This remains training-source data preparation. Bibliographic work/edition grouping,
larger clean training data and independent frozen development labels are still
required before selecting a recognizer and evaluating untouched final test.

## Public Artifacts

- [Complete code-free review ZIP with all 64 crops, 34 source regions and both teacher outputs](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/b0a44817cf7da627ac27bb82b873415067330b0c/data/recognizer-data-v3-expansion-review-v1-20261005/recognizer-data-v3-expansion-v1-review-data.zip).
- [Pinned review data provenance](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/tree/b0a44817cf7da627ac27bb82b873415067330b0c/data/recognizer-data-v3-expansion-review-v1-20261005).
- [Pinned original run, reproduced audit, glyph diagnostics and scalar UI QA](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/848e4ddcfa8f6c13219687ae6bbd972ae94cc354/experiments/2026-10-05/recognizer-data-v3-expansion-result-v1).

Every published file was downloaded at its immutable revision and checked by
SHA-256. Every member of the downloaded review ZIP was checked as well.
The original returned evidence ZIP is published byte-for-byte under a canonical
filename, without the local download suffix `(1)`. The review ZIP includes the
unchanged frozen pilot config and an empty agent-observation record; no simulated
review events or automatic human labels were created or published.
