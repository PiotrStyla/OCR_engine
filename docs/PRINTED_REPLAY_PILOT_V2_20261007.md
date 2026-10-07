# Printed replay pilot V2

## Run

Open `training/colab_printed_replay_pilot_v2.ipynb` in Colab. Select **CPU**,
then **Run all**. Upload nothing. The notebook downloads the same pinned source
ZIP from Hugging Face and verifies its SHA256 before extraction.

No GPU, paid job, model training or model publication is started.
The result is `printed-replay-pilot-v2-evidence.zip`; download it before closing
the session and return it for audit. If automatic download is blocked, use the
Colab Files panel: `printed-replay-pilot-v2-.../mining/`, right-click the ZIP,
Download. Do not use a manually assembled `/content` browser URL.

## Changes From V1

V1 completed 12 pages and 396 lines but accepted zero pairs. Twenty-six exact
line anchors were discarded solely because of the whole-page 40% coverage gate.
See [the audited V1 result](PRINTED_REPLAY_PILOT_V1_RESULT_20261007.md).

V2 changes **candidate mining**, not training admission:

- Page coverage is recorded as a diagnostic, not a minimum admission threshold.
- Exact NFC/whitespace-only source matching, unique word-boundary anchors, minimum
  line length, word confidence >=90, line-like geometry, source order and
  nonoverlap requirements remain unchanged.
- No case folding, historical spelling modernization, fuzzy replacement,
  long-s replacement or hyphen repair.
- Both training and evaluation eligibility remain false for every candidate.
- The separate Lalka probe work is not replay training data.

Using the old returned line records reproduces **25 replay candidate proposals
and one probe proposal** under this rule. This is not a prediction or guarantee
of the next extraction count: installed OCR versions/model packages may differ.
Actual versions, runner hash and code revision are recorded in the new evidence.

## Complete Evidence

The ZIP contains candidate PNG/TXT pairs and their hashes, original references,
source revisions/license snapshots, complete exclusions and page coverage,
runtime versions and file checksums. V2 additionally includes:

- `native-pages/<page-id>/page.png`: decoded original DjVu page pixels, not preview JPEGs;
- `native-pages/<page-id>/tesseract.tsv`: full returned word geometry and confidence;
- `native-pages/<page-id>/tesseract.stderr.txt`: OCR diagnostics;
- explicit line indices and Tesseract grouping keys in candidate records.

Original DjVu books remain in the Colab working directory and are not duplicated
inside the result ZIP. They can be recovered from the pinned original URLs/SHA1.
Generated native page PNGs and crops have SHA256 checksums. Source images are
public-domain; Wikisource transcription attribution and CC BY-SA 4.0 terms remain
in `source-package/NOTICE.md` and source records.

This evidence permits later pixel-level crop inspection without rerunning OCR.
It does not itself establish that all crop boundaries are correct or that source
transcriptions are error-free. The small pool and source/model exposure limits
remain; no benchmark gold or SOTA claim is made.

## Validation

CPU tests check the V2 selection, unchanged glyph/confidence protection, notebook
syntax and pins, and evidence packaging with mocked external decoder/OCR calls.
The real returned V1 ZIP was independently audited. V2 has not yet been executed
end-to-end in a Linux Colab runtime; the notebook is the next validation run.
