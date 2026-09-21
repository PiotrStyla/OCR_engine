# Kraken visual diagnosis, 2026-09-21

Source: user-supplied `kraken-diagnostics-20260921T071439598318Z.zip`.
All 183 listed file hashes verified. Three fixed first-ID test pages were
inspected; these are title pages, not a representative sample of body text.
No model parameters were changed and no training was performed.

## Observations

- NA2_FT__433927: 42 detections. Overlay omits most of the large NOWE ATENY
  and AKADEMIA titles, fragments some large letters and includes handwritten
  marginalia. Several body-text crops nevertheless contain readable lines.
- Nowiny_z_Rakuz_FT__436609: 81 detections. Overlay includes bleed-through,
  ornament fragments, marginal marks and handwriting. Some title crops cut
  through the letters rather than enclosing their full height.
- Powodzenia_FT__436636: 52 detections. Selected exported crops show both
  readable printed lines and severely clipped large title letters.

## Recognition Evidence

In NA2 line 38, the crop visibly contains the uppercase surname
`BENEDYKTA CHMIELOWSKIEGO`, but the corresponding prediction does not preserve
those words. Line 18 visibly contains `IDIOTOM dla Nauki`; its prediction is
also severely corrupted. This is evidence of a recognition-path problem in
addition to segmentation, not merely a page reading-order error.

The exported crops precede recognizer resizing/normalization. Visual readability
does not establish that the tensors delivered to the recognizer are correct.
Possible causes still include domain mismatch, checkpoint/codec problems,
input transformations and pairing/order issues. This inspection does not
distinguish those causes and does not justify retraining immediately.

## Next Controlled Check

1. Check checkpoint input specifications, codec and actual normalized line
   tensors. Verify record-to-line identity, not only matching list lengths.
2. Establish an in-domain EHRI control using known line transcriptions.
3. On separate development data, compare recognizers on identical validated
   line regions, then compare segmenters with the recognizer held fixed.
4. Define treatment of marginalia, stamps and bleed-through in references;
   full-page output versus print-only references can inflate insertion errors.

The frozen test remains unchanged. Do not select parameters or discard hard
pages based on these diagnostics. No new CER/WER or accuracy claim is made.

Local inspection artifacts: `data/kraken-diagnostics-20260921/index.html`,
`crop-contact-sheet.png`, and `page-*/overlay.jpg`.
