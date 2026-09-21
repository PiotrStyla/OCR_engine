# Body-text draft diagnostic: 2026-09-21

User archive: `body-dev-diagnostic-20260921T153113386749Z.zip`.
All five payload checksums verified. The input manifest exactly matches the
published input ZIP (SHA-256
`913ded2bd5d742099567a2652460baaf3c4a8bb16625492d0931607396dbdd55`).
All reported metrics were recomputed locally from aligned predictions/references.

Both models returned 63 nonempty predictions, zero recorded errors, zero flagged
truncations. Environment: Tesla T4, Torch 2.10.0+cu128, Transformers 4.57.6,
Pillow 11.3.0, jiwer 4.0.0, huggingface_hub 0.36.0. Model revisions and FP32 greedy
generation match the diagnostic protocol.

## Results

| Scope | Model | CER | WER | Lowercase CER, diagnostic |
|---|---|---:|---:|---:|
| All 63 draft lines | Microsoft base-printed | 85.30% | 100.00% | 39.61% |
| All 63 draft lines | PiotrSty mixed-v3 | 31.64% | 79.47% | 31.10% |
| 56 without needs-review | Microsoft base-printed | 85.31% | 99.73% | 38.82% |
| 56 without needs-review | PiotrSty mixed-v3 | 31.28% | 80.27% | 30.72% |

Primary normalization is NFC and whitespace only. Lowercase CER is an additional
diagnostic, not a replacement metric. Microsoft's uppercase outputs explain much
of the primary gap; after lowercasing mixed-v3 leads by 8.51 percentage points
on all lines. Removing seven uncertain lines barely changes mixed-v3 CER, so
those seven labels alone do not explain the poor overall quality.

## Collection Breakdown

| Collection | Lines | Base CER | Mixed-v3 CER |
|---|---:|---:|---:|
| Choragiew_FT | 16 | 89.69% | 47.94% |
| Relacja_koronacji_FT | 27 | 80.99% | 19.92% |
| Slawna_wiktoria_FT | 4 | 100.00% | 32.77% |
| Wiesc_FT | 16 | 87.70% | 41.83% |

The evaluated crops represent six pages, four collection IDs and 2,585 normalized
reference characters. This is narrower than the initially selected ten pages:
eight of nineteen source regions failed line-count matching and did not yield
aligned line candidates. These results do not measure that segmentation loss.

## Interpretation

Mixed-v3 is the stronger of these two candidates on this particular draft sample,
but remains unsuitable for unattended high-quality historical transcription.
The difference across collections warrants inspecting crop geometry, typeface
coverage and historical glyph errors before choosing a training recipe. It is
not causal proof of any one failure mechanism.

One short mixed-v3 output exceeds 100% line CER due to extra text. For
`Slawna_wiktoria_FT__437103__r003__line003`, inspect the source region and crop
before attributing this solely to recognizer hallucination: neighbouring-line
fragments were a known risk of the proposed geometry.

These references are user-corrected drafts; repeat review was explicitly skipped.
Three private-use/replacement flags remain across three lines. The 56-line subset
is not independently verified gold. No near-duplicate or upstream-training audit
establishes independence. Do not compare these scores directly with the earlier
15 heading regions or the frozen 36-page full-page test as a performance trend.

Next: visually inspect a bounded set of high-error crop/reference/output triples,
then test any geometry change on the same development IDs. Keep raw predictions
and this reference version unchanged. Do not train on the frozen historical test.
