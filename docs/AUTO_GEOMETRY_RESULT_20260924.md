# Automatic geometry: verified paired development result

Source: `body-dev-diagnostic-20260924T102032899332Z.zip`.
SHA256: `ba094189d571715c9fa6086ac557c8881a0c6a68f6f73a62f4efc6234821f9c3`.

Both models completed 126 predictions, with no errors or empty outputs.
Environment: Transformers 4.57.6, huggingface_hub 0.36.0, jiwer 4.0.0,
Torch 2.11.0+cu128, Pillow 11.3.0. Model/data revisions are unchanged.
All five evidence payload checksums and both models' metrics were verified.
The exact input manifest was reconstructed from the two pinned source ZIPs.
The locally rebuilt combined ZIP has a different byte hash; compression/runtime
differences are a possible cause. We do not claim byte-identical reconstruction
of the combined archive or infer its original bytes from the manifest alone.

## Results

| Model / subset | Original CER | Automatic + fallback CER |
| --- | ---: | ---: |
| mixed-v3, all 63 | 31.6441% | 29.4391% |
| mixed-v3, changed 31 | 45.8986% | 40.6452% |
| Microsoft, all 63 | 85.2998% | 84.4874% |
| Microsoft, lowercase diagnostic only | 39.6132% | 37.2147% |

Mixed-v3: 20 changed lines improve, nine regress, two tie in edit count.
All 32 fallback predictions remain identical between arms. Net improvement:
57 character edits. WER falls from 79.4749% to 77.5656% across all 63 lines.
Historical spelling is preserved; primary normalization is NFC + whitespace.
Microsoft's case-sensitive score is strongly affected by casing: do not claim
a proportional recognition advantage from the primary CER comparison alone.

## Visual inspection of all nine regressions

Pairs are ordered by descending additional character edits. Observations are
visual hypotheses, not causal ablations. Neither references nor boxes were edited.

| ID | Extra edits | Observation |
| --- | ---: | --- |
| Choragiew_FT__436799__r002__line003 | 3 | Tighter horizontal framing; fragments at upper/lower boundaries. Reference needs review. |
| Choragiew_FT__436804__r002__line000 | 3 | New lower boundary admits visible fragments of the next line. |
| Choragiew_FT__436804__r002__line002 | 3 | New crop contains substantial next-line fragments along its lower edge. |
| Wiesc_FT__436884__r000__line000 | 3 | New crop admits next-line fragments; already degraded print. Reference contains a private-use glyph. |
| Wiesc_FT__436884__r001__line002 | 3 | Original terminal colon is absent from the automatic crop. This is concrete content loss. |
| Slawna_wiktoria_FT__437103__r003__line002 | 2 | Both crops contain neighbors; automatic crop has upper and lower fragments. Reference needs review and contains a private-use glyph. |
| Choragiew_FT__436804__r001__line001 | 1 | Similar target text, tighter horizontal framing and edge fragments; cause not isolated. |
| Wiesc_FT__436884__r001__line003 | 1 | Tighter crop and boundary fragments; no clear missing word identified. |
| Wiesc_FT__436899__r003__line000 | 1 | Both crops have adjacent-line contamination; automatic crop also has lower fragments. |

## Decision and next experiment

Do not promote this detector to the default yet. The net development gain is
real for these draft references, but nine regressions include lost punctuation.
Do not introduce a per-ID fallback selected using these OCR scores.

Next engineering target: image-only ownership of punctuation/small components,
followed by line-following boundaries instead of one axis-aligned rectangle.
Add synthetic geometry tests for a terminal colon, detached accents and slanted
neighbors before creating a new GPU experiment. Keep all 63 records and the same
references in comparisons. Any tuning on this sample is development work;
evaluate a fixed candidate on separate pages before broader quality claims.

## Reproduction

`training/audit_geometry_run.py` verifies artifacts and produces `audit.json`,
an HTML report and nine paired crop images without running OCR or modifying data.
The current local output is `data/body-geometry-audit-20260924/`.
The input ZIPs are the published original body-dev and body-auto-geometry ZIPs.
The result ZIP remains local. No new data or predictions were uploaded.

```powershell
python -m training.audit_geometry_run --run <result.zip> --original experiments/2026-09-21/body-dev-diagnostic/body-dev-input.zip --automatic experiments/2026-09-24/body-auto-geometry/body-auto-geometry-input.zip --output <new-output-directory>
```
