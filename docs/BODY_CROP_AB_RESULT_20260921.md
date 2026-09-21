# One-line geometry A/B result

Source: `body-dev-diagnostic-20260921T154245122854Z.zip` supplied by the user.
Five payload checksums verified. Input manifest matches the published notebook's
embedded ZIP. Both A predictions exactly reproduce the earlier 63-line run.
Per-variant CER/WER recomputed locally. All four outputs are nonempty, successful,
with no flagged truncations. Same pinned models, T4, FP32, greedy generation.

| Model | A: original CER | B: manual geometry CER | B: lowercase CER |
|---|---:|---:|---:|
| Microsoft base-printed | 172.22% | 77.78% | 22.22% |
| PiotrSty mixed-v3 | 122.22% | 5.56% | 5.56% |

Draft reference: `W świetne błáwaty.`

- Mixed-v3 B: `W świetne bławaty.`
- Microsoft B: `W SWICTNE BLAWATY.`

Mixed-v3 falls from 22 character edits to one (plain a versus accented a) against
the unchanged 18-character normalized reference. B WER remains 33.33% because
one of three words differs. No reference correction or modernization was applied.

### Historical spelling check

A subsequent visual inspection of the B crop shows an acute mark above the first
a in `błáwaty`. The pre-review source already contained `W świetne błáwaty.`.
The user's exported event has identical before/after text, an empty note and a
verified decision. This spelling was not introduced by the note migration.
Keep the original accented character; do not rewrite the reference to agree with
the recognizer. This is a visual spot check, not independent double adjudication.
NFC may reconcile composed/decomposed encodings, but accent stripping, mapping
long s to modern s, and historical spelling modernization are not permitted in
the primary transcription metric. Lowercase CER remains a separately named
diagnostic and also retains accents.

## Conclusion

For this selected line, changing geometry substantially improves both recognizers.
The original crop clipped the target and included a neighbouring line. The effect
also includes altered image framing/aspect ratio under the model's preprocessing;
this experiment does not isolate which geometric factor contributes how much.
It does not validate the remaining manual crop edges or every reference glyph.

This is one short, post-hoc example, not a representative benchmark or a tested
automatic segmentation fix. Substituting only this B prediction into the old
mixed-v3 results would change aggregate CER from 31.64% to approximately 30.83%
(797/2585), a counterfactual calculation, not a new full-sample run.

Next: develop image-only line boundaries on the separate development regions,
preserving ascenders/descenders and excluding adjacent-line fragments. Evaluate
the same IDs without silently dropping failed regions, and compare geometry
before spending resources on recognizer training. Keep the frozen test untouched.
