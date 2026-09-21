# Printed development control: 2026-09-21

Source: user-provided `printed-dev-20260921T103012950790Z.zip`.
All 23 checksums verified; no unlisted payload files. Selection reproduced from
archived validation/test metadata with the original selection and overlap guards.
Metrics independently recomputed from aligned prediction IDs and references.

## Results

15 regions from three development pages/collections; 474 normalized reference
characters. Both models returned 15 nonempty outputs, zero recorded errors and
zero flagged truncations. Tesla T4, Torch 2.10.0+cu128, Transformers 4.57.6.

| Model | CER | WER | Lowercase CER (diagnostic only) |
|---|---:|---:|---:|
| microsoft/trocr-base-printed | 69.4093% | 89.5522% | 25.9494% |
| PiotrSty/trocr-pl-mixed-v3 | 23.4177% | 70.1493% | 20.2532% |

Primary metrics use NFC + whitespace only. The additional diagnostic applies
Python `lower()` after normalization, not casefold or historical spelling repair.
Microsoft frequently emits uppercase text. Thus much of the 45.99 percentage-point
primary CER gap is case error; the lowercase gap is only 5.70 points. Retain both
raw predictions and primary metrics; do not describe this as a general threefold
recognition improvement.

| Collection | Regions | Base CER | Mixed-v3 CER |
|---|---:|---:|---:|
| Choragiew_FT | 5 | 72.11% | 32.65% |
| Relacja_koronacji_FT | 7 | 74.09% | 17.00% |
| Wiesc_FT | 3 | 50.00% | 26.25% |

## Interpretation and limits

Mixed-v3 is a promising candidate for further development evaluation, not a
production-ready historical recognizer. Both models still misread historical
glyphs and decorative type; no model selection for the frozen benchmark follows
from this small test. Region scores cannot be compared directly with full-page
Kraken/Tesseract scores.

The references include three private-use characters and one replacement character
across three regions. They remain in scoring. References were not manually audited.
Only 15 heading/publication-detail regions, from three pages, were evaluated;
upstream model-training overlap and near-duplicates remain unaudited.

Visual spot-check: Choragiew r003 is a single historical blackletter line with
small adjacent-line fragments at the lower edge; Relacja r016 is one legible line.
This is not a visual audit of all 15 crops. In r016 mixed-v3 reads almost all the
text but predicts a dotted z in place of the accented a in the opening word.

Next: prepare a larger, manually checked development set of genuine body-text
lines, stratified by typeface/document, with audited references and document-level
separation. Freeze selection before evaluating further preprocessing or training.
Keep the 36-page frozen test untouched until development choices are finalized.

Raw files remain local in `data/printed-dev-20260921/`; `review.html` shows crops,
references and both outputs. No images or predictions were published by this audit.
