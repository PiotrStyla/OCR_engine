import json

from training import build_reference_candidate as candidate


PUA = chr(0xEADA)
S_LONG = chr(0x017F)


def test_apply_mapping_only_touches_mapped_codepoints():
    text = "kot" + PUA + " oraz " + chr(0xFFFD) + " koniec"
    out = candidate.apply_mapping(text, candidate.MAPPING)
    assert out == "kot" + S_LONG + "t oraz " + chr(0xFFFD) + " koniec"
    assert candidate.apply_mapping("czysty tekst", candidate.MAPPING) == "czysty tekst"


def test_verify_candidate_rejects_text_changes_beyond_mapping():
    frozen = [{"id": "a", "sha256": "0", "text": "x" + PUA}]
    ok = [{"id": "a", "sha256": "0", "text": "x" + S_LONG + "t"}]
    assert candidate.verify_candidate(frozen, ok, candidate.MAPPING)["mapped_spots"] == 1
    bad = [{"id": "a", "sha256": "0", "text": "y" + S_LONG + "t"}]
    try:
        candidate.verify_candidate(frozen, bad, candidate.MAPPING)
        raise SystemExit("should have rejected")
    except ValueError as error:
        assert "mapping" in str(error)
    try:
        candidate.verify_candidate(frozen, [{"id": "b", "sha256": "0", "text": "x" + S_LONG + "t"}],
                                   candidate.MAPPING)
        raise SystemExit("should have rejected")
    except ValueError as error:
        assert "identity" in str(error)


def test_score_impact_reports_deltas():
    frozen = [{"id": "p1", "text": "kot" + PUA}]
    cand = [{"id": "p1", "text": "kot" + S_LONG + "t"}]
    predictions = {"m": {"p1": "kot" + S_LONG + "t"}, "n": {"p1": "kotft"}}
    impact = candidate.score_impact(frozen, cand, predictions)
    assert impact["m"]["cer_candidate"] == 0.0 and impact["m"]["cer_frozen"] > 0
    assert impact["m"]["delta"] < 0, "mapping helps the system that reads the glyph"
    assert set(impact["n"]) == {"cer_frozen", "cer_candidate", "delta", "pages"}


def test_run_writes_candidate_package(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    row = {"id": "p1", "sha256": "0", "text": "kot" + PUA + " i " + chr(0xEBA6), "image": "x"}
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    preds = tmp_path / "preds"
    preds.mkdir()
    (preds / "m.jsonl").write_text(json.dumps({"id": "p1", "text": "kot" + S_LONG + "t i ff"}) + "\n",
                                   encoding="utf-8")
    out = tmp_path / "cand"
    candidate.run(manifest, preds, out, {"m": "m.jsonl"})
    rows = (out / "history_testA_manifest.candidate-v2.jsonl").read_text(encoding="utf-8")
    assert S_LONG + "t" in rows and chr(0xEBA6) in rows, "mapped glyph in, unmapped codepoint kept"
    impact = json.loads((out / "impact.json").read_text(encoding="utf-8"))
    assert impact["verification"]["mapped_spots"] == 1
    assert impact["systems"]["m"]["pages"] == 1
    mapping = json.loads((out / "mapping.json").read_text(encoding="utf-8"))
    assert mapping["decision"].startswith("map PUA")
