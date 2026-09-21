# EHRI in-domain control result, 2026-09-21

Source: user-supplied `ehri-control-20260921T101507457069Z.zip`.
All 62 listed checksums match. Page: EHRI-ET-ZIH3010106_01.
Both tracks completed with 29 output lines and no empty lines.

| Track | Page CER | Page WER |
| --- | ---: | ---: |
| ALTO line geometry + EHRI recognizer | 2.81% | 15.95% |
| EHRI segmenter + EHRI recognizer | 7.60% | 30.17% |

Oracle line-micro CER is 2.86% (line boundaries are scored separately).
Page metrics were recomputed locally using archived predictions and reference
XML, NFC and whitespace normalization. Oracle line IDs match the ALTO order.
The model checksums match the earlier IMPACT experiment.

## Interpretation

The recognizer can produce mostly correct text on this in-domain page when
given ALTO geometry. This weakens the hypothesis of a global broken checkpoint
or grossly incorrect decoding path. It does not prove all inference is correct.

Replacing ALTO geometry/order with predicted segmentation increases page CER
by 4.79 percentage points. This difference includes crop geometry, detection
and order effects; it must not be attributed solely to missed lines. The
end-to-end output near the header changes the order of the name and headings.

Combined with the historical-title-page inspection, domain mismatch remains
a strong explanation for much of the historical-print failure, alongside
confirmed segmentation and alphabet limitations. This is not causal proof.

## Limits and Next Step

One page, potentially used during training or validation; NOT an independent
benchmark and NOT evidence of generalization or SOTA. Do not compare its CER
directly with the 36-page historical test as if datasets were interchangeable.

Keep this checkpoint as an EHRI control. Next compare a printed-text recognizer
on fixed, validated line crops from a separate historical development set.
Evaluate recognition independently first, then compare full-page segmentation.
Do not train on or tune to the frozen historical test.
