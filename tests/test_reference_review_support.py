import json

from training import reference_review_support as support


PUA = chr(0xEADA)        # U+EADA: dominant PUA codepoint in test A
TIRONIAN = chr(0x204A)   # U+204A: tironian et, what systems print
GT = "Ala ma kota" + PUA + " oraz psa w ogrodzie"
PRED = "Ala ma kota" + TIRONIAN + " oraz psa w ogrodzie"


def rows():
    return [{"id": "page-1", "text": GT, "image": "x", "sha256": "0"}]


def test_defect_indices_finds_pua_and_replacement_chars():
    text = "a" + chr(0xFFFD) + "b" + PUA + "c"
    pua, uffd = support.defect_indices(text)
    assert uffd == [1] and pua == [3], "PUA by codepoint, U+FFFD separately"


def test_replacements_map_defects_to_model_output():
    gt = "kot" + chr(0xFFFD) + " oraz"
    pred = "kotX oraz"
    out = support.replacements_at(gt, pred, [3])
    assert out == {3: "X"}


def test_collect_evidence_counts_spots_not_models():
    predictions = {"m1": {"page-1": PRED}, "m2": {"page-1": PRED}}
    evidence, uffd = support.collect_evidence(rows(), predictions)
    assert evidence["U+EADA"]["occurrences"] == 1, "one spot counted once despite two systems"
    assert evidence["U+EADA"]["replacements"]["m1"] == {TIRONIAN: 1}
    assert evidence["U+EADA"]["replacements"]["m2"] == {TIRONIAN: 1}
    assert uffd == []


def test_errata_draft_lists_every_system():
    rows_ = [{"id": "NA2_FT__434735", "text": "Tam‑", "image": "x", "sha256": "0"}]
    predictions = {"best": {"NA2_FT__434735": "TU ERRATA\nTak czytaiac"},
                   "other": {"NA2_FT__434735": "ERRATA"}}
    draft = support.errata_draft(rows_, predictions, page_id="NA2_FT__434735")
    assert "TU ERRATA" in draft and "## other" in draft
    assert "Tam‑" in draft
    assert "Proposed consensus draft" in draft and "best" in draft


def test_run_writes_evidence_package(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows()) + "\n" +
                        json.dumps({"id": "NA2_FT__434735", "text": "Tam‑", "image": "x", "sha256": "0"}) + "\n",
                        encoding="utf-8")
    pred_dir = tmp_path / "preds"
    (pred_dir / "a").mkdir(parents=True)
    (pred_dir / "b").mkdir(parents=True)
    (pred_dir / "a" / "predictions.jsonl").write_text(
        json.dumps({"id": "page-1", "text": PRED}) + "\n" +
        json.dumps({"id": "NA2_FT__434735", "text": "TU ERRATA"}) + "\n", encoding="utf-8")
    (pred_dir / "b" / "predictions.jsonl").write_text(
        json.dumps({"id": "page-1", "text": GT}) + "\n" +
        json.dumps({"id": "NA2_FT__434735", "text": "ERRATA"}) + "\n", encoding="utf-8")
    out = tmp_path / "review"
    support.run(manifest, pred_dir, out, {"m1": "a/predictions.jsonl", "m2": "b/predictions.jsonl"})
    pua = json.loads((out / "pua-evidence.json").read_text(encoding="utf-8"))
    assert pua["codepoints"] == 1 and pua["occurrences"] == 1
    assert set(pua["systems"]) == {"m1", "m2"}
    draft = (out / "errata-draft.md").read_text(encoding="utf-8")
    assert "TU ERRATA" in draft and "ERRATA" in draft
