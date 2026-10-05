# Historical glyph review: status import, 5 October 2026

The user's export matches mining manifest
`25c3ba0766e1b34798b1fd5397cc05f3c37dd260e3c3b02234e7e89394ba4961`.
All 14 events pass the existing review validator: page/image/text identities,
history, unique event IDs, reviewer and chronology. There are 12 reviewed pages,
one reviewer and zero text changes. Initially, nine latest decisions were
`verified`, three `needs-review`; three pages had no events.

The user explicitly confirmed that the three `needs-review` pages should also
be visually verified. A separate source-bound confirmation records this message
verbatim; the original JSON is not overwritten and no review events/timestamps
are synthesized. Effective status: **12 verified, 3 unreviewed**.

Confirmed pages:
- `SLAWNA_VICTORIA_FT__437145`
- `Relacja_koronacji_FT__436578`
- `Wiesc_FT__436868`

Still without a decision:
- `Slawna_wiktoria_FT__437103`
- `Choragiew_FT__436804`
- `Slawna_wiktoria_FT__437089`

The original export SHA-256 is
`1340e526aebb42c5d9edf3f4ada08d3f7add14cddfb40c081449565448a70732`.
The status-confirmation SHA-256 is
`0df3ff059651670e4cbc8679d41651e70ab7362e66f2b2089c95214bb02c6e40`.
This is one reviewer's visual verification, not independent adjudication/gold.
All pages remain validation-only, ineligible for training. Historical spelling,
all draft text, raw v5/v7 predictions and metrics remain unchanged. No GPU run
or model improvement is claimed.

`training.import_review_status_confirmation` validates the original export and
requires an explicit confirmation bound to its hash and the source manifest.
Only latest `needs-review` pages can be confirmed; absent pages cannot be
silently verified. The output keeps the original JSON, confirmation, immutable
manifest snapshot, status report and SHA-256 manifest. The snapshot's image
paths refer to the original mining bundle, not images duplicated in this receipt.
The focused importer/review suite passes 32 CPU tests.

## Public Receipt

[Original export, scoped confirmation, status report and receipt ZIP on HF](https://huggingface.co/datasets/PiotrSty/slayer-ocr-experiment-evidence/tree/6d97f1694553071f7139dbb220a4211b1558a463/experiments/2026-10-05/historical-glyph-review-v1).
All nine published files were downloaded at this immutable revision and verified
by SHA-256. This receipt contains no program files, scans or weights. Its source
manifest snapshot resolves images in the earlier mining bundle. The original
export still records its historical statuses; the separate confirmation/report
provides the effective 12 verified / 3 unreviewed status without rewriting history.
