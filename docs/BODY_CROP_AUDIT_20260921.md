# Body crop visual audit

The six highest mixed-v3 line-CER examples were inspected against the images.
Selection is explicitly post-hoc and cannot establish the prevalence of defects.

## Confirmed geometry defect

`Slawna_wiktoria_FT__437103__r003__line003` has draft reference
`W świetne błáwaty.` and CER 122.22% (22 edits / 18 normalized characters).
The original box `[0, 435, 2209, 550]` cuts the top of the desired line and includes
the next line beginning `Tam za zwycięstwo...`. Predicted extra text corresponds
partly to visible neighbouring text; this is not evidence of pure hallucination.
Even eliminating all 22 edits would lower aggregate CER by at most 0.85 percentage
points (22 / 2585). This one defect does not explain the overall 31.64% CER.

The A/B control retains A exactly and crops B from the verified source region
at `[965, 375, 1950, 520]`. The reference, model revisions and generation settings
are unchanged. This is a manually selected geometry intervention, not a new
automatic detector. Small neighbouring fragments remain around the edges; the
crop is a proposal, not a certified clean line. No inference improvement is yet
claimed. Pooled metrics across A and B must not be treated as a benchmark.

## Other inspected lines

- `Choragiew_FT__436799__r002__line000`: mostly one historical blackletter line,
  with fragments at the top edge. Severe recognition errors persist.
- `Choragiew_FT__436804__r002__line001`: one dominant blackletter line; predictions
  contain many substitutions and spacing errors. Upper-edge clipping remains a risk.
- `Choragiew_FT__436804__r001__line002`: one dominant blackletter line, minor edge
  fragments; many glyph substitutions.
- `Choragiew_FT__436799__r003__line003`: visible next-line fragments at the bottom;
  both geometry and recognition remain plausible error sources.
- `Wiesc_FT__436899__r002__line000`: one dominant degraded blackletter line with
  speckling and severe substitutions. Reference itself was marked needs-review.

These observations support auditing geometry alongside historical typeface
coverage. They do not isolate the effect of pretraining, alphabet, reference
conventions or resizing. Do not start broad training on the strength of this audit.

Run `training/kaggle_body_crop_ab.ipynb` in a fresh Kaggle GPU + Internet session.
The notebook is under 1 MB and embeds the two inputs; no dataset upload is needed.
It produces the usual evidence ZIP and prints separate A/B CER for both models.
