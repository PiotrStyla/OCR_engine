import json

from training import build_recognizer_v3_package as pkg


ACCEPTED = {
    "id": "w-0001-line-001", "page_id": "w-0001", "work_family": "w", "split": "replay-candidate",
    "text": "ala ma kota oraz", "min_word_confidence": 95.0, "source_span_normalized": [0, 16],
    "image": "pairs/replay-candidate/x.png", "image_sha256": "a", "text_file": "pairs/x.txt",
    "text_sha256": "b", "crop_review_status": "not-human-reviewed"}
PROBE = {**ACCEPTED, "id": "w-0002-line-001", "page_id": "w-0002", "split": "replay-probe"}
POOL = {"id": "w-0003-line-001", "page_id": "w-0003", "work_family": "w", "split": "replay-candidate",
        "cut_rule": "ink-gap-approved", "content_type": "unclassified-pending-review",
        "text": "zosia czyta", "min_word_confidence": 91.0, "source_span_normalized": [20, 30],
        "review_status": "candidate-pending-human-sign-off",
        "ink_verified_clean": True, "eligible_for_training": False}


def roots(tmp_path, accepted, pool_rows):
    audit = tmp_path / "audit"
    (audit / "evidence").mkdir(parents=True)
    (audit / "evidence" / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in accepted), encoding="utf-8")
    pool = tmp_path / "pool"
    (pool / "crops").mkdir(parents=True)
    (pool / "texts").mkdir(parents=True)
    (pool / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in pool_rows), encoding="utf-8")
    for row in pool_rows:
        (pool / "crops" / f"{row['id']}.png").write_bytes(b"png")
        (pool / "texts" / f"{row['id']}.txt").write_text(row["text"], encoding="utf-8")
    return audit, pool


def test_package_separates_probe_and_counts_sources(tmp_path):
    audit, pool = roots(tmp_path, [ACCEPTED, PROBE], [POOL])
    out = tmp_path / "pkg"
    pkg.run(audit, pool, out)
    candidates = [json.loads(x) for x in (out / "train-candidates.jsonl").read_text().splitlines()]
    probe = [json.loads(x) for x in (out / "probe-heldout.jsonl").read_text().splitlines()]
    assert len(candidates) == 2 and len(probe) == 1
    assert {r["source"] for r in candidates} == {"expansion-mining-accepted", "replay-pool-recovered"}
    assert probe[0]["split"] == "replay-probe"
    assert all(r["eligible_for_training"] is False for r in candidates + probe)
    report = json.loads((out / "report.json").read_text())
    assert report["totals"]["from_expansion_mining"] == 1
    assert report["totals"]["from_replay_pool"] == 1
    assert report["totals"]["probe_held_out"] == 1


def test_package_rejects_overlapping_ids(tmp_path):
    clash = {**POOL, "id": ACCEPTED["id"]}
    audit, pool = roots(tmp_path, [ACCEPTED], [clash])
    try:
        pkg.run(audit, pool, tmp_path / "pkg2")
        raise SystemExit("should have rejected")
    except ValueError as error:
        assert "Overlapping" in str(error)


def test_unify_preserves_provenance_and_flags():
    row = pkg.unify_accepted(ACCEPTED)
    assert row["image_sha256"] == "a" and row["eligible_for_training"] is False
    assert row["source"] == "expansion-mining-accepted"
