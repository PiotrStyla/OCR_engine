import hashlib
import json
import zipfile

import pytest

from training.build_layout_consensus_review import build_review


TEACHERS = ("qwen3-vl-4b", "doclayout-yolo", "surya-layout2")


def _jsonl(rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


def _evidence(tmp_path, *, tamper=False, tamper_policy=False):
    image = b"private-image-fixture"
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "p1.jpg").write_bytes(image)
    image_meta = {
        "file_name": "images/p1.jpg",
        "sha256": hashlib.sha256(image).hexdigest(),
        "width": 100,
        "height": 200,
    }
    accepted = [{
        "schema": "slayer-layout-consensus-v1",
        "page_id": "p1",
        "image": image_meta,
        "objects": [{
            "id": "p1:cluster-1",
            "label": "text_region",
            "bbox_xyxy": [10, 20, 80, 150],
            "teachers": ["doclayout-yolo", "surya-layout2"],
            "mean_score": 0.9,
        }],
    }]
    review = [{
        "id": "p1:cluster-2",
        "page_id": "p1",
        "label": "heading",
        "bbox_xyxy": [20, 5, 70, 18],
        "teachers": ["qwen3-vl-4b"],
        "mean_score": 0.5,
        "score_kinds": ["neutral-unavailable"],
        "reasons": ["below-quorum"],
    }]
    consensus_files = {
        "consensus.jsonl": _jsonl(accepted),
        "review-queue.jsonl": _jsonl(review),
    }
    consensus_checksums = "".join(
        f"{hashlib.sha256(content.encode()).hexdigest()}  {name}\n"
        for name, content in consensus_files.items()
    )
    policy = json.dumps({
        "categories": ["text_region", "heading", "figure"],
        "quorum": 2,
        "iou_threshold": 0.5,
        "conflict_iou": 0.5,
        "min_score": 0.0,
        "containment_threshold": 0.9,
        "granularity_ratio": 2.0,
    }).encode()
    files = {
        "run.json": json.dumps({
            "pages": 1,
            "consensus": {
                "accepted_objects": 1,
                "review_objects": 1,
                "categories": ["text_region", "heading", "figure"],
            },
            "consensus_policy_sha256": hashlib.sha256(policy).hexdigest(),
        }),
        "consensus-policy.json": policy,
        **{f"consensus/{name}": content for name, content in consensus_files.items()},
        "consensus/checksums.sha256": consensus_checksums,
    }
    for teacher in TEACHERS:
        proposal = _jsonl([{
            "page_id": "p1",
            "detections": [{
                "id": f"{teacher}-1",
                "label": "text_region",
                "bbox_xyxy": [10, 20, 80, 150],
                "score": 0.8,
            }],
        }])
        files[f"teachers/{teacher}/teacher-proposals.jsonl"] = proposal
        files[f"teachers/{teacher}/checksums.json"] = json.dumps({
            "teacher-proposals.jsonl": hashlib.sha256(proposal.encode()).hexdigest(),
        })
    if tamper:
        files["consensus/consensus.jsonl"] += "{}\n"
    if tamper_policy:
        files["consensus-policy.json"] += b" "
    archive = tmp_path / "evidence.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name, content in files.items():
            output.writestr(name, content)
    return archive, image_dir


def test_builds_private_static_review_without_embedding_image(tmp_path):
    archive, image_dir = _evidence(tmp_path)
    output = build_review(archive, image_dir, tmp_path / "review" / "index.html")
    html = output.read_text(encoding="utf-8")
    assert "SLAYER layout review" in html
    assert "../images/p1.jpg" in html
    assert "private-image-fixture" not in html
    assert all(teacher in html for teacher in TEACHERS)
    assert "accepted" in html and "review" in html

    with pytest.raises(FileExistsError):
        build_review(archive, image_dir, output)


def test_rejects_tampered_evidence_and_wrong_image(tmp_path):
    archive, image_dir = _evidence(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="Consensus checksum mismatch"):
        build_review(archive, image_dir, tmp_path / "tampered.html")

    archive, image_dir = _evidence(tmp_path / "second")
    (image_dir / "p1.jpg").write_bytes(b"different")
    with pytest.raises(ValueError, match="Missing or mismatched image"):
        build_review(archive, image_dir, tmp_path / "wrong-image.html")


def test_rejects_tampered_consensus_policy(tmp_path):
    archive, image_dir = _evidence(tmp_path, tamper_policy=True)
    with pytest.raises(ValueError, match="policy checksum mismatch"):
        build_review(archive, image_dir, tmp_path / "tampered-policy.html")


def test_interactive_review_is_bound_to_archive_and_ontology(tmp_path):
    archive, image_dir = _evidence(tmp_path)
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps({
        "schema": "slayer-layout-ontology-v2",
        "classes": {"text_region": {}, "heading": {}, "figure": {}},
    }), encoding="utf-8")
    output = build_review(
        archive, image_dir, tmp_path / "review" / "index.html", ontology)
    html = output.read_text(encoding="utf-8")
    assert hashlib.sha256(ontology.read_bytes()).hexdigest() in html
    assert "slayer-layout-review-patch-v1" in html
    assert "slayer-layout-review-decisions.json" in html
    assert "JSON.stringify(packet,null,2)+'\\n'" in html
    assert "localStorage" in html
    assert "Export JSON" in html


def test_rejects_ontology_with_different_categories(tmp_path):
    archive, image_dir = _evidence(tmp_path)
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps({
        "schema": "slayer-layout-ontology-v2",
        "classes": {"text_region": {}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Ontology does not match"):
        build_review(archive, image_dir, tmp_path / "review.html", ontology)
