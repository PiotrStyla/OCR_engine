# Experimental automatic line geometry

This experiment changes image geometry only. Historical draft references, line
IDs, model checkpoints and decoding remain unchanged. It does not train a model
or replace the production detector.

## Method

`training/body_line_geometry.py` uses OpenCV Otsu thresholding and connected
components on single-column region images. It estimates character height from
components, groups character centres with local linear fits for slant, attaches
small nearby components (including accents), and bounds the complete components.
Small height-dependent padding is added; original RGB pixels are cropped without
resizing, deskew, binarization or text correction in the saved inputs.

The detector receives an image only, not reference text or expected line counts.
Regions with a majority of edge-truncated anchor components in a group reject
that group. Proposed crops track the fraction of ink from components not assigned
to the line. Count agreement gates tentative ordering against existing IDs; crops
with more than 10% foreign ink retain their original image. Count agreement alone
does not prove alignment. No thresholds were selected using new OCR predictions.

## Local result

The v3 preparation examines all 19 development regions. In the existing 63-line
comparison, 31 receive automatic proposed geometry and 32 retain original crops
because of count mismatch or foreign-ink checks. None of the 63 IDs is dropped.
The eight regions outside that original comparison remain reported but unscored.

The previously problematic `W świetne błáwaty.` line is found automatically at
`[993, 385, 1900, 538]`, without its manually specified A/B box. Visual inspection
shows the intended line and retained accent; small neighbouring fragments remain
at the edges. Two earlier component-only prototypes were retained locally; the
final one includes slope-aware grouping and per-crop foreign-ink gating.

The method still oversegments the long Relacja paragraphs, so those inputs fall
back to their originals. It is experimental, not a solved general segmenter.
Parameters were developed on these regions after prior error inspection. This
is development evidence and must not be promoted to an independent benchmark.

## Colab comparison

`training/colab_auto_geometry.ipynb` downloads both hash-pinned input archives
automatically. It runs the original and automatic-with-fallback versions in the
same session on both pinned TrOCR models. It prints separate 63-line metrics,
56-line metrics excluding prior needs-review IDs, and matched metrics for only
the 31 altered inputs. There is no pooled score across both arms.

Enable GPU, Run all, download the final evidence ZIP. No manual dataset upload.
No inference improvement is claimed until that GPU run is returned.
