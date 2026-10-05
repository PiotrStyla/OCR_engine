# Recognizer DATA ENGINE teacher pilot: result and crop review

## Audited Remote Run

The returned archive SHA256 is
`e7f54254c33eeadaf6c873539d2657ca751909cb2f0cb66c4b815bf25ea27811`.
All 22 members and checksum coverage passed verification against the frozen
input ZIP, code commit `e24545b22b65d42937fd88714ff99c7b81e50f90`, model
revisions, reference-free inputs and pinned import versions. Token termination
and processor geometry were checked and proposals/report reproduced from raw
predictions. The evidence reports Tesla T4, Python 3.13.15, Torch 2.11.0+cu130
and Transformers 4.57.6. No inference was rerun locally.

- Qwen: 64 returned, 63 EOS, one 512-token cap/repetition, no missing inputs.
- TrOCR: 64 returned, 64 EOS, no caps or missing inputs.
- Proposals: **62 disagreements, one abstention, one agreement**.
- The agreement differs from the unreviewed source label: both models output
  `Ołob` where the label contains `Oſob`. Agreement is not a ground-truth rule.
- Raw emitted glyph counts: Qwen `ſ` 4, `á` 31, `ɇ` 0; TrOCR `ſ` 0, `á` 3,
  `ɇ` 0. These are output counts, **not recall or accuracy**.
- Zero clean training examples, zero gold labels, no SOTA claim. No CER/WER
  quality ranking is computed against these unverified crop/label pairs.

## Confirmed Crop/Label Problem

Visual inspection of `NA1_FT__433925__r011__line000` shows only a word
fragment in a 204x50 crop. Its unreviewed label is the entire line
`Na nieſkończoną BOGU Chwałɇ.`. The frozen original 903x109 region shows
that entire line. This is not solely a recognizer failure: the existing
component grouping retained a fragment while line-count equality accepted
the label. Count equality does not establish crop/reference alignment.

This is an agent inspection, not a user review event or gold transcription.
The original crop and label remain unchanged. At least this pair must be
rejected or re-cropped and independently reviewed before training. The
remaining 63 crops are **not** assumed geometrically correct.

## Review Now, Do Not Rerun Colab

Prepared 64 line crops with 35 checksum-verified original source-region JPEGs.
The offline editor shows the line, full region, original weak label and both
raw OCR outputs. It exports a separate `slayer-recognizer-line-review-v1`
history with crop hash, region hash, text changes, reviewer and geometry choice.
Existing page-review exports keep their previous schema and behavior.

1. Inspect the line crop first. If it contains exactly one complete line,
   choose **Cała jedna linia**. If it is a fragment, clips letters, combines
   multiple lines or belongs to a different source line, choose **Odrzuć
   wycinek** and record the reason. Do not complete cut text from context.
2. For complete crops, edit the transcription to match the visible image,
   retaining old spelling, punctuation, abbreviations and historical glyphs.
   Model/source labels are suggestions, not authority. Choose **Zweryfikowano
   wzrokowo** only after checking the text; ambiguous readings remain pending.
3. Save each decision, export history and return the JSON. The first entry is
   the confirmed fragment, followed by the agreement and capped output.
   An image-only geometry choice is distinct from a text-verification choice.

The editor blocks verified text with unreviewed geometry, and requires a reason
for a rejected crop. Rejected geometry stays explicit in queue and history.
Even verified text does not turn a rejected crop into a training example.
This stage creates no training labels automatically. Later import must require
both complete geometry and verified diplomatic text, preserve all originals,
and maintain document/split and duplicate controls before student training.

## Reproduce The Offline Review

Use the frozen public review data bundle and GitHub code; the bundle contains
only images and JSON/JSONL metadata, not HTML/JS/Python or model weights.
After unpacking it, run the current repository command against its directory:

```powershell
python -m training.build_annotation_review --manifest input/manifest.jsonl --diagnostics diagnostics.json --contexts contexts.json --geometry-review --output review
```

142 focused CPU tests passed after the auditor/editor changes. Isolated
Playwright/Edge checks at 1366x900 and 390x844 verified crop/context rendering,
mandatory geometry and rejection reason, export schema, reload persistence,
legacy mode and no console errors or mobile horizontal overflow. QA-generated
events are test-only and are excluded from research evidence and training data.
