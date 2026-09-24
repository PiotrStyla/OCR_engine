# Experimental line bands, version 1

Status: local development candidate, no OCR measurement yet, not the default.
Based on visual inspection of the prior development regressions, not held-out
test data. References and historical spelling are unchanged.

## Changes

`detect_lines(image, follow_lines=True)` expands small-component horizontal
attachment from 0.25 to 0.8 estimated character heights. Vertical attachment
uses the fitted local line position instead of the whole rectangle's center.
This allows a separated terminal colon to remain with its line.

Each pair of neighboring fitted lines defines a per-column midpoint boundary.
Assigned component bounds, including detached accents, are preserved with
padding. `crop_line_band` converts the crop to RGB and whitens pixels outside
these boundaries. Inside-band pixels are unchanged; no resize, deskew or
language correction is performed. This is masking, not lossless cropping.
Ownership can still be wrong; preserving assigned components is not proof that
all genuine target characters have been assigned correctly.

The old rectangular path remains the default. The foreign-ink gate remains
the pre-mask rectangular fraction, so the candidate is conservatively rejected
when that fraction exceeds the existing 0.10 threshold.

## Local validation

- 13 focused tests passed, including old behavior, a detached accent, terminal
  colon, slanted neighboring lines, invalid band shape, paired audit and hashes.
- Old default outputs match the saved v3 outputs across all 19 source regions.
- All 63 IDs and reference strings retained; 33 new proposals, 30 fallbacks.
- All 30 fallback image byte strings match their original inputs.
- Visual spot checks: the previously missing terminal colon in
  `Wiesc_FT__436884__r001__line002` is now visible; next-line fragments in
  `Choragiew_FT__436804__r002__line000` are removed by the mask.
- `Slawna_wiktoria_FT__437103__r003__line003` retains the acute in the scan,
  but still contains some next-line fragments. The geometry problem is not solved.

Input ZIP: `data/body-auto-geometry-bands-v1/body-auto-geometry-input.zip`.
SHA256: `730a7cfe4a0c2d2a6575f53fdc7d65a36fb2bf31fa8e9f7d9d734cc514e34431`.
Band boundaries and detector checksum are included in ZIP provenance.

## Reproduce

```powershell
python -m training.prepare_auto_geometry --input-zip experiments/2026-09-21/body-dev-diagnostic/body-dev-input.zip --region-manifest data/body-dev-review-v2/region-manifest.jsonl --output data/body-auto-geometry-bands-v1 --follow-lines
```

The output directory must not already exist. No files are overwritten.

## Next measurement

Compare all 63 original crops against these 63 candidate/fallback inputs with
the pinned models and Transformers 4.57.6 environment. Report the new 33-line
changed subset separately; do not compare its aggregate directly to the old
31-line changed subset. Keep the all-63 paired score as the main development
comparison. No CER improvement is claimed before that GPU run.
