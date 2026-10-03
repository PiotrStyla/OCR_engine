# Full-page validation v4: EOS fix does not generalize reliably

Date: 2026-10-03. Expanded development diagnostics, not an independent benchmark.
Follow-up: [all 15 review decisions imported as a single-review draft, preserving the original score](FULL_PAGE_REVIEW_IMPORT_V1_20261003.md).
Evidence ZIP SHA-256: `475510b3436d93d3dd65401a1630254a319d436b871d060b07faf131529f258f`.

All 17 archive members passed checksum verification. The frozen config,
deterministic selection of all 15 validation pages, source metadata hashes,
unchanged reference fields, model/input identities and generated token traces
passed the audit. No training pages or final-test pages entered inference.
Raw micro CER/WER were independently recomputed and match the supplied report.

[Public HF evidence and audit](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/308da61cbc389c23f9cc9876d461dfc2d5f15f75/experiments/2026-10-03/full-page-validation-v4-result).
[Complete code-free review ZIP, including all 15 scans](https://huggingface.co/datasets/PiotrSty/slayer-ocr-datasets/resolve/e1bbed0e83db24842a0481bc3b6b4c27c87e3243/data/full-page-validation-v4-review-20261003/full-page-validation-v4-review-inputs.zip?download=true).
The publication contains 25 evidence files and 10 dataset paths. Both ZIPs were
downloaded from their published revisions and SHA-256 verified. Images and XML
are included inside the complete review ZIP; its size is 59,736,080 bytes.
Code remains on GitHub, with publication provenance pinned to
`1e6382d47cc74c98c6af545e6db406e01d3969d4`.

## Observed Result

- Worker execution: 15/15 pages, zero inference exceptions or OOM.
- EOS termination: 11/15; four responses reached the 4,096-token limit.
- Primary all-page CER: **243.7391%**; WER: **515.8644%**.
- Macro CER: 433.7164%; insertions can legitimately produce scores above 100%.
- Runtime: Tesla T4, Python 3.13.15, Torch 2.11.0+cu130, Transformers 5.18.0.
- Sum of per-page generation times: 1,344.45 seconds (22.41 minutes), excluding
  model loading (26.74 seconds), downloads and environment preparation.
- Peak allocated VRAM: 3,676,280,320 bytes. No claim about total reserved VRAM.

The two pages previously inspected in v3 reproduce the exact raw texts and token
IDs. This verifies that the expanded run retains the v3 behavior for those pages,
not that the two-page sample represented all pages.

| Collection | Pages | Raw CER | Raw WER |
| --- | --- | --- | --- |
| Choragiew_FT | 3 | 220.0402% | 698.3923% |
| Relacja_koronacji_FT | 3 | 387.7561% | 473.9264% |
| SLAWNA_VICTORIA_FT | 3 | 30.6878% | 73.7003% |
| Slawna_wiktoria_FT | 3 | 348.8462% | 748.5000% |
| Wiesc_FT | 3 | 245.8771% | 692.6045% |

For triage only, the selected 11 EOS-terminated responses have micro CER 26.9350%
and WER 71.1667%. This survivor subset is explicitly selection-biased and does not
replace the 15-page primary score. No post-hoc deduplication, spelling correction,
failure exclusion or repetition penalty was applied to primary predictions.

## Four Generation Failures

| Page | Finish | Tokens | Raw CER | Observed Repetition |
| --- | --- | --- | --- | --- |
| Slawna_wiktoria_FT__437089 | length | 4096 | 3004.44% | repeated `S.` lines |
| Relacja_koronacji_FT__436547 | length | 4096 | 1661.07% | repeated imprint/footer block |
| Choragiew_FT__436794 | length | 4096 | 898.64% | repeated `1` lines |
| Wiesc_FT__436899 | length | 4096 | 643.77% | repeated `1` lines |

The explicit EOS override is active and verified in all token traces. These four
responses do not emit a configured EOS before the cap. This is not evidence of an
ignored emitted EOS and cannot be repaired merely by adding the same stop IDs again.
Correct termination on two pages was a real adapter improvement, not a general
guarantee of reliable full-page generation. Do not use this profile as a production
OCR backend or accept its output automatically as teacher labels.

## Reference Defects and Visual Checks

The reference audit flags private-use Unicode on 13/15 pages (103 characters),
replacement characters on 5/15 pages (14 characters), and incomplete source
reading order on 4/15 pages. All 15 lack source line geometry. None is reviewed gold.

Visual inspection of `Slawna_wiktoria_FT__437116` confirmed a reference reading-order
defect: the scan starts with the stanza beginning `Mieymy`, then the `A wy` stanza,
then the `Ale` stanza. The source reference starts with the last stanza. The model
follows the visible stanza order, although it still makes transcription errors.
Its raw CER 77.9221% must not be interpreted as character-recognition error alone.
The source text is preserved unchanged; this finding is not a complete adjudication.

On `Relacja_koronacji_FT__436547`, inspection also shows a substantial central block
between the papal heading and the imprint. The raw model output skips that block
and repeats the imprint. Removing the repeated tail would not restore missing text.

Four outputs contain generated image placeholders and one contains Cyrillic text.
These are triage flags, not calibrated hallucination-rate measurements.
Do not guess replacements for private-use characters. Preserve historical
`á`, `ſ`, spelling, punctuation and line/stanza order supported by the scan.

## Next Gate: Repair References Before Another Larger Run

A complete code-free review-input ZIP contains 15 native-size PNG scans, 15 source
PAGE XML files, unchanged references, raw predictions, triage, provenance and
checksums. No model weights or program files. All review PNGs were verified against
cached source JPEG decoded pixels and source hashes. Encoded PNG hashes may differ
from Colab; both review and inference hashes are retained separately.

Review archive SHA-256: `5b90ca396592c151cbfabd38a02f91f9171649605223d6ce066b1f8552ce2938`.
An offline editor was generated locally with the existing annotation-review tool.
All 15 pages start pending. Model text does not overwrite or prefill corrected labels.
The queue places capped outputs first, followed by reading-order flags and CER.

1. Compare the whole scan against the transcription, not only highlighted Unicode.
2. Put corrections in the transcription field. Keep unclear fragments unresolved.
3. Use `Propozycja korekty` for a draft; use `Zweryfikowano wzrokowo` only after checking
   every included text block and its order. Never modernize historical spelling.
4. Save decisions, then use `Eksportuj historię` and return that JSON.

The next quality comparison should use adjudicated references, retain all 15 pages,
and compare a second supported full-page backend before selecting a student or
teacher. Freeze a new independent reviewed test before fine-tuning or SOTA claims.
Do not spend another larger GPU run on unchanged Ovis decoding merely to collect
more capped output. This result is a rejected deployment/automatic-teacher gate,
not a rejected data-engine architecture or proof that every Ovis backend fails.

The triage/review preparation and existing evaluator safeguards passed 84 focused
CPU tests. No new GPU inference, training or modern-document benchmark was run.
