# Recognizer v3: work-family risk before training

## Result

The original 89-page image/text screening found no proposals at its frozen
thresholds. That result was explicitly not proof of work-disjointness. A separate
source inspection now identifies **two protective work families spanning splits**:

| Protective family | Existing source splits | Future training quarantine |
| --- | --- | --- |
| Nowe Ateny: NA1, NA2, NA3, NA4 | train, test, geometry-holdout | NA1_FT, NA3_FT |
| Wyprawa: versions 1 and 2 | train, geometry-holdout | Wyprawa_1_FT |

This is conservative grouping for leakage prevention, not certification of exact
bibliographic editions, copies, or identical JPEGs. The titles, historical spelling,
source manifests, human decisions, existing splits and previous measurements
remain unchanged. No model is trained and no new accuracy/SOTA claim is made.

The impact on current inputs is:

- **9 of 29 permitted train-source pages** quarantined for future training.
- **19 of 64 single-human-reviewed pairs** affected; all 64 decisions retained.
- **20 of 64 new review proposals** affected; no automatic annotation decisions.
- 20 train-source pages in seven other collections remain unresolved, **not cleared**.
- `training_freeze_ready=false`; zero bibliographic edition identities certified.

## Source Evidence

The first, second and third NA title pages identify Nowe Ateny and their volume;
the NA4 opening page refers to the first part and the fourth volume. The corpus
author describes Nowe Ateny as a four-volume component of the IMPACT corpus.
This corroborates protective family grouping, not a page-to-copy mapping.
[Bień, 2014, section 3.2](https://journals.ispan.edu.pl/index.php/cs-ec/en/article/download/cs.2014.008/174/0).

Agent inspection of `Wyprawa_1_FT__436980` and `Wyprawa_2_FT__436948` shows
corresponding numbered paragraphs 48-55. Both title pages concern Sultan Amurat
and 1634, with differently worded titles. Bibliographic corroboration is available
from [ESJP source StarWyp](https://sxvii.pl/index.php?id_zrodla=1399&strona=kw_cyt_zr)
and the [Kórnik catalog record](https://platforma.bk.pan.pl/pl/search_results/1276476?q%5Bsort_attributes%5D%5Bfield_name%5D=title.sort&q%5Bsort_attributes%5D%5Border%5D=desc).
Neither record proves the exact origin of either IMPACT scan. Agent inspection
does not count as the user's transcription review or independent human gold.

Access receipts record three successful primary-source downloads with byte counts
and SHA-256. The original ribes-80 page fails expired-certificate verification;
the old corpus-description endpoint times out. TLS checks were not disabled.
No full third-party article/catalog content is republished. Receipts establish
access and content digests, not verified bibliographic identities.

Eight inspected original scans are bound by page ID and image SHA-256 and included
unchanged in the code-free visual-evidence bundle. Attribution: IMPACT/PSNC via
`PiotrSty/impact-psnc-polish-ocr`, revision
`c7cb156fb95d2880699c33725bbaf1fbc1008fea`, CC-BY-3.0.

## Reproducible Overlay

`training.audit_recognizer_work_groups` verifies complete checksums of the original
source audit, reviewed pool and new review package, then checks four frozen input
hashes. Protective groups require explicit evidence page IDs for every collection.
It rejects unknown/overlapping groups, false identity certification, cross-batch
root duplicates, held-out input lines and dataset/page/collection mismatches.

Outputs include `work-family-groups.jsonl`, `line-dispositions.jsonl`, source-access
receipts, frozen policy, copied input manifests and a future-selection exclusion
list. Line dispositions are a **separate overlay**, never rewritten review events.
Every disposition remains ineligible for training until the remaining gates pass;
being outside the two protective families does not certify independence.

The original immutable run configurations are not retroactively edited. Future
training must use the new exclusion list and resolved bibliographic groups rather
than directly loading the older pool's `eligible_for_training` flags.

Two fresh CPU audits reproduce every output byte. **256 focused tests passed**,
including 25 tests for the work-family protocol. No heavy model ran locally.

```powershell
python -m training.audit_recognizer_work_groups --source-audit SOURCE_AUDIT --policy experiments/2026-10-05/recognizer-work-family-audit-v1/policy.json --sources SOURCE_RECEIPTS.json --reviewed-pool REVIEWED_POOL --expansion-review EXPANSION_REVIEW --output FRESH_OUTPUT
```

This audit does not cover the separate 36-page IMPACT-v2 evaluation archive,
external corpora or upstream model pretraining. Existing reported scores remain
diagnostic, not evidence of work-disjoint SOTA generalization.

## Next Action

No Colab rerun is needed. Continue the existing 64-line review and return
`slayer-recognizer-line-review-31debfc142b5.json`. Reviews from quarantined families
still have archival value; do not delete or modernize their transcriptions.
The editor and its manifest have not been changed by this audit.

Next project gate: resolve the seven remaining train-source collections and frozen
development works, expand work-disjoint training data, then train and compare the
recognizer on independently adjudicated development pages. Final test stays frozen.
