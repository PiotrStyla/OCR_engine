# Kraken recognizer input check, 2026-09-21

Source: `kraken-input-check-20260921T073230245786Z.zip` supplied by the user.
All 533 listed checksums match. The three selected pages contain 175 line
records (42, 81, 52), all with returned images.

## Confirmed

- All 175 returned recognizer crops match the previous diagnostic crops in
  mode, dimensions and decoded pixels. All corresponding predictions match.
- The completed diagnostic validates segmentation/record IDs and order.
- Model input specification is `[1, 1, 48, 0]`: one grayscale channel,
  height 48, variable width. All recorded reconstructed tensors have height
  48, nonzero standard deviation and values within [0, 1].
- Actual configuration uses FP32 (`32-true`), batch size 1, greedy CTC decoding,
  padding 16, `one_channel_mode=L`, and `use_legacy_polygons=False`.
- Visually inspected normalized previews retain readable text. Polarity is
  reversed relative to source images; this alone is not evidence of a bug.
- The codec has 85 single-character entries with distinct labels. Missing
  output characters include uppercase Y, Q, V, X, uppercase Polish L with
  stroke (U+0141), lowercase v and long s (U+017F). These remain unavailable
  under NFD normalization. Polish nasal vowels, for example, may instead be
  represented by base letters plus combining marks.

## Interpretation

The prior separate-crop export did not cause the observed discrepancy for
these 175 lines. The inspected normalization does not visibly erase the text.
The alphabet cannot exactly reproduce all historical typography; this is a
confirmed model limitation, but does not explain all the severe substitutions
on readable words made entirely of supported characters.

These tensors were reconstructed from returned crops and the active transform
configuration, not captured directly at the neural network input. No forward
hook, weight-to-codec semantic alignment test or in-domain EHRI control has
yet been run. This evidence therefore does not prove the entire inference
path is correct or that domain mismatch is the sole cause.

## Next Decision

Run an in-domain EHRI control with the same checkpoint and independently
checked transcriptions. If this is also poor, inspect checkpoint conversion,
codec alignment and training preprocessing before any new training. If it is
good, compare an appropriate historical-print recognizer on a development set.
Keep line crops fixed when comparing recognizers, and keep recognizer fixed
when comparing segmenters. Adding codec characters alone does not teach the
network their appearance and may invalidate the output-layer mapping.

Artifacts are under `data/kraken-input-check-20260921/`. The contact sheet
`input-comparison.png` pairs actual crops with normalized previews. Frozen
test references, models and benchmark scores were not changed.
