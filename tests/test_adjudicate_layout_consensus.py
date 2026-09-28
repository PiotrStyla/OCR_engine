import hashlib
import json

import pytest

from tests.test_build_layout_consensus_review import _evidence
from training.adjudicate_layout_consensus import adjudicate, file_digest


def _ontology(tmp_path):
    path = tmp_path / "ontology.json"
    path.write_text(json.dumps({
        "schema": "slayer-layout-ontology-v2",
        "classes": {
            "text_region": {}, "heading": {}, "figure": {},
        },
    }), encoding="utf-8")
    return path


def _packet(tmp_path, archive, ontology, *, action="accept", source_hash=None,
            label=None):
    decision = {
        "review_id": "p1:cluster-2",
        "action": action,
        "note": "checked",
    }
    if label is not None:
        decision["label"] = label
    path = tmp_path / f"{action}.json"
    path.write_text(json.dumps({
        "schema": "slayer-layout-review-patch-v1",
        "source_evidence_sha256": source_hash or file_digest(archive),
        "ontology_sha256": file_digest(ontology),
        "reviewer": "Reviewer A",
        "timestamp": "2026-09-28T18:00:00Z",
        "decisions": [decision],
    }), encoding="utf-8")
    return path


def test_complete_review_builds_traceable_clean_candidate(tmp_path):
    archive, _ = _evidence(tmp_path / "source")
    ontology = _ontology(tmp_path)
    packet = _packet(tmp_path, archive, ontology)
    output = tmp_path / "candidate"
    report = adjudicate(archive, packet, ontology, output)
    assert report["consensus_objects"] == 1
    assert report["accepted_review_objects"] == 1
    assert report["final_objects"] == 2
    rows = [json.loads(line) for line in
            (output / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {item["origin"] for item in rows[0]["objects"]} == {
        "teacher-consensus", "human-review"}
    assert not any(path.suffix.lower() in {".jpg", ".png"} for path in output.iterdir())
    for line in (output / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert file_digest(output / name) == expected
    with pytest.raises(FileExistsError):
        adjudicate(archive, packet, ontology, output)


@pytest.mark.parametrize("action,label,accepted,relabeled,rejected", [
    ("reject", None, 0, 0, 1),
    ("relabel", "figure", 1, 1, 0),
])
def test_reject_and_relabel_are_explicit(tmp_path, action, label, accepted,
                                         relabeled, rejected):
    archive, _ = _evidence(tmp_path / "source")
    ontology = _ontology(tmp_path)
    packet = _packet(tmp_path, archive, ontology, action=action, label=label)
    report = adjudicate(archive, packet, ontology, tmp_path / "candidate")
    assert report["accepted_review_objects"] == accepted
    assert report["relabeled_review_objects"] == relabeled
    assert report["rejected_review_objects"] == rejected


def test_incomplete_or_foreign_review_is_rejected_before_writes(tmp_path):
    archive, _ = _evidence(tmp_path / "source")
    ontology = _ontology(tmp_path)
    packet = _packet(tmp_path, archive, ontology, source_hash="0" * 64)
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="identity mismatch"):
        adjudicate(archive, packet, ontology, output)
    assert not output.exists()

    content = json.loads(packet.read_text())
    content["source_evidence_sha256"] = file_digest(archive)
    content["decisions"] = []
    packet.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="Incomplete layout review"):
        adjudicate(archive, packet, ontology, output)
    assert not output.exists()


def test_tampered_ontology_is_rejected(tmp_path):
    archive, _ = _evidence(tmp_path / "source")
    ontology = _ontology(tmp_path)
    packet = _packet(tmp_path, archive, ontology)
    ontology.write_text(ontology.read_text() + " ")
    with pytest.raises(ValueError, match="identity mismatch"):
        adjudicate(archive, packet, ontology, tmp_path / "candidate")
