# Printed replay expansion V1

## Scope

A frozen input contains **34 source pages from eight new work families**:
24 candidate pages from six families and 10 probe pages from two other families.
Scans include historical essays, memoir, regional geography, clipped newspaper
articles and fiction. This improves source diversity, not representativeness
for modern administrative documents. Newspaper articles are clippings, not
complete newspaper-layout benchmark pages.

All retained source revisions have Wikisource quality 4. Scans were checked
against pinned Commons SHA1 and public-domain license metadata; source
transcriptions retain CC BY-SA 4.0 attribution and revision links.
No project human review, training eligibility or SOTA claim follows from this.
Publication dates unavailable in source metadata remain `null`.

| Family | Domain | Split | Retained pages |
| --- | --- | --- | ---: |
| Schneider: Babiagora | Regional geography | Candidate | 2 |
| Witkiewicz: Teatr | Cultural essay | Candidate | 5 |
| May: Nad Rio de la Plata | Adventure fiction | Candidate | 8 |
| Pamietniki lekarzy (1939) | Memoir | Candidate | 2 |
| Prokesch: Nowa Reforma (1923) | Newspaper article | Candidate | 4 |
| Zawadzki: Kurjer Warszawski (1904) | Newspaper article | Candidate | 3 |
| Andersen: Basnie (1929) | Literary fiction | Probe only | 5 |
| Boccaccio: Dekameron | Literary fiction | Probe only | 5 |

No prior Zeromski/Prus pilot pages or protected work families are included.
The entire Andersen and Boccaccio families remain probe-only; other editions
must not be added to training as aliases. Model pretraining exposure is unknown,
so this is a diagnostic work-held-out probe, not a certified untouched test.

## Discovery Evidence

The first bounded selection requested 29 quality-4 pages, but 27 were short title
pages and only two candidate pages survived. It had no probe and was rejected.
The selection and raw results are preserved, not silently replaced.

The final selection used four bounded category metadata batches of 500 pages,
requiring quality-4 membership and revision size at least 1800 bytes before
selecting works. Size is only a discovery signal, not proof of useful text.
Of 44 requested pages, full rendered-body extraction retained 34 and rejected
10 with footnotes/complex markup. Rejected snapshots and the discovery receipts
are included in the frozen package; they are not input rows for OCR.

Archive SHA256:
`95661ff30f5b904f67896e8079d8332d4665cdf9b9b3d579bc4a2e3f6d548769`.
The source package is independently checksum-verified and the selected image
previews were inspected for all eight families. Preview JPEGs are not native OCR
inputs: the notebook retrieves original DjVu files and verifies their SHA1.

## Run One Notebook

Use `training/colab_printed_replay_expansion_v1.ipynb` on **CPU** and run all cells.
No files, tokens or GPU are needed. The notebook downloads the complete frozen
source package from a pinned public HF revision and verifies its SHA256.
Code and work selection are pinned separately to complete revision/checksum
receipts. Original DjVu downloads total about 152 MB, based on the verified
Commons sizes; each download is bounded at 100 MB.

The miner uses V2 exact-line rules unchanged: Tesseract Polish, native page
geometry, minimum word confidence 90, unique exact source anchors, line order
and geometry checks. No fuzzy matching, spelling modernization or confidence
relaxation is introduced to increase the yield. The only change is the source
pool; this isolates whether data diversity and scan quality improve acquisition.

Download **`printed-replay-expansion-v1-evidence.zip`** and return it for audit.
It includes native PNGs, complete word TSV, crop/text pairs, quarantined lines,
source receipts, environment versions and checksums. If automatic download is
blocked, use the Colab Files panel and download the ZIP inside the timestamped
`printed-replay-expansion-v1-...` directory.

The notebook does **not train** or change the baseline. No expected candidate
count is promised before OCR runs. If acquisition remains sparse, the next change
must target segmentation/teacher agreement rather than training on a tiny pool.

## Validation Boundary

The source acquisition and package verification completed locally. Regression
tests cover work/scan alias separation, frozen input checks, CPU-only notebook
generation, original V1/V2 behavior and crop audits. Notebook schema and Python
cell syntax are validated. Native DjVu decoding/Tesseract and Colab end-to-end
execution are **not verified locally**, because those Linux executables and the
Colab runtime are unavailable on this laptop. A returned evidence ZIP is the
required execution check, not notebook generation alone.
