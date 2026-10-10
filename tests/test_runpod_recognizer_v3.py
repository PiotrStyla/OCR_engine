import json
from pathlib import Path

from training import runpod_recognizer_v3 as v3


CFG = json.loads((Path(__file__).resolve().parents[1] / "experiments/2026-10-10/recognizer-v3/config.json")
                 .read_text(encoding="utf-8"))


def fixtures(tmp_path, pool_ids=("p1", "p2"), candidates=("c1", "c2", "c3"), probe=("h1",)):
    repo = tmp_path / "repo"
    pool = repo / "experiments/2026-10-10/replay-pool"
    (pool / "crops").mkdir(parents=True)
    (pool / "texts").mkdir(parents=True)
    rows = [{"id": i, "text": "tekst " + i} for i in pool_ids]
    (pool / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for row in rows:
        (pool / "crops" / f"{row['id']}.png").write_bytes(b"png")
        (pool / "texts" / f"{row['id']}.txt").write_text(row["text"], encoding="utf-8")
    expansion = tmp_path / "expansion"
    (expansion / "pairs").mkdir(parents=True)
    manifest = []
    for ident, split in [(i, "replay-candidate") for i in candidates] + [(i, "replay-probe") for i in probe]:
        (expansion / "pairs" / f"{ident}.png").write_bytes(b"png")
        (expansion / "pairs" / f"{ident}.txt").write_text("tekst " + ident, encoding="utf-8")
        manifest.append({"id": ident, "split": split, "image": f"pairs/{ident}.png",
                         "text_file": f"pairs/{ident}.txt"})
    (expansion / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest),
                                             encoding="utf-8")
    return repo, expansion


def test_config_declares_the_v3_contract():
    assert CFG["schema"] == "slayer-recognizer-guarded-training-v3"
    assert CFG["reviewed_v3"]["expected_pairs"] == 226
    assert [v["id"] for v in CFG["variants"]], "at least one training variant"
    assert CFG["selection"]["fallback"] == "unchanged-baseline"
    assert CFG["automatic_production_promotion"] is False and CFG["sota_claim"] is False


def test_build_reviewed_v3_merges_and_holds_out_probe(tmp_path):
    repo, expansion = fixtures(tmp_path)
    cfg = {**CFG, "reviewed_v3": {"expected_pairs": 5, "from_pool": 2, "from_expansion": 3,
                                  "probe_held_out": 1, "split": "replay-candidate"}}
    out, rows, probe = v3.build_reviewed_v3(repo, expansion, cfg, tmp_path / "work")
    assert len(rows) == 5 and len(probe) == 1 and probe[0]["id"] == "h1"
    assert (out / "p1.png").is_file() and (out / "c3.txt").is_file()
    assert not (out / "h1.png").is_file(), "probe never enters training"


def test_build_reviewed_v3_rejects_drift_and_overlap(tmp_path):
    repo, expansion = fixtures(tmp_path)
    bad = {**CFG, "reviewed_v3": {"expected_pairs": 4, "from_pool": 2, "from_expansion": 3,
                                  "probe_held_out": 1, "split": "replay-candidate"}}
    try:
        v3.build_reviewed_v3(repo, expansion, bad, tmp_path / "w2")
        raise SystemExit("should have rejected count drift")
    except ValueError as error:
        assert "count drift" in str(error)
    overlap = {**CFG, "reviewed_v3": {"expected_pairs": 5, "from_pool": 2, "from_expansion": 3,
                                      "probe_held_out": 1, "split": "replay-candidate"}}
    pool_manifest = repo / "experiments/2026-10-10/replay-pool/manifest.jsonl"
    rows = [json.loads(x) for x in pool_manifest.read_text().splitlines()]
    rows[0]["id"] = "c1"
    pool_manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    try:
        v3.build_reviewed_v3(repo, expansion, overlap, tmp_path / "w3")
        raise SystemExit("should have rejected overlap")
    except ValueError as error:
        assert "Overlapping" in str(error)
