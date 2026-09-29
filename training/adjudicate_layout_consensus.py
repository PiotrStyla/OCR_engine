"""Apply complete, traceable human decisions to private layout consensus evidence."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from training.build_layout_consensus import containment, iou
from training.build_layout_consensus_review import load_evidence


ACTIONS = {"accept", "reject", "relabel"}
SCHEMA = "slayer-layout-review-patch-v1"


def file_digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _jsonl(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _timestamp(value) -> str:
    if not isinstance(value, str):
        raise ValueError("Invalid review timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Invalid review timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("Review timestamp must include a timezone")
    return value


def _load_ontology(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    ontology = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(ontology, dict) or
            ontology.get("schema") != "slayer-layout-ontology-v2" or
            not isinstance(ontology.get("classes"), dict) or not ontology["classes"]):
        raise ValueError("Invalid layout ontology")
    return ontology, file_digest(path)


def _validate_packet(packet, archive_hash, ontology_hash, review_rows, categories):
    if (not isinstance(packet, dict) or packet.get("schema") != SCHEMA or
            packet.get("source_evidence_sha256") != archive_hash or
            packet.get("ontology_sha256") != ontology_hash):
        raise ValueError("Review schema or source identity mismatch")
    reviewer = packet.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 100:
        raise ValueError("Invalid reviewer")
    _timestamp(packet.get("timestamp"))
    decisions = packet.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Invalid decisions")
    expected = {row["id"] for row in review_rows}
    by_id = {}
    for item in decisions:
        if not isinstance(item, dict):
            raise ValueError("Invalid decision")
        review_id = item.get("review_id")
        action = item.get("action")
        note = item.get("note", "")
        if (review_id not in expected or review_id in by_id or action not in ACTIONS or
                not isinstance(note, str) or len(note) > 2000):
            raise ValueError("Unknown, duplicate or invalid decision")
        source = next(row for row in review_rows if row["id"] == review_id)
        label = item.get("label")
        if action == "relabel":
            if label not in categories or label == source["label"]:
                raise ValueError("Invalid relabel decision")
        elif label is not None:
            raise ValueError("Only relabel decisions may provide label")
        by_id[review_id] = item
    if set(by_id) != expected:
        missing = sorted(expected - set(by_id))
        raise ValueError(f"Incomplete layout review: {missing}")
    return reviewer.strip(), by_id


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def geometry_conflicts(pages, policy):
    conflicts = []
    for page in pages:
        objects = page["objects"]
        for left_index, left in enumerate(objects):
            for right in objects[left_index + 1:]:
                if (left["label"] != right["label"] and
                        iou(left["bbox_xyxy"], right["bbox_xyxy"]) >= policy["conflict_iou"]):
                    conflicts.append({
                        "page_id": page["page_id"], "reason": "label-conflict",
                        "left_id": left["id"], "left_label": left["label"],
                        "right_id": right["id"], "right_label": right["label"],
                    })
                if left["label"] != right["label"]:
                    continue
                left_area, right_area = _area(left["bbox_xyxy"]), _area(right["bbox_xyxy"])
                ratio = max(left_area, right_area) / min(left_area, right_area)
                if (ratio >= policy.get("granularity_ratio", math.inf) and
                        containment(left["bbox_xyxy"], right["bbox_xyxy"]) >=
                        policy.get("containment_threshold", 1.0)):
                    conflicts.append({
                        "page_id": page["page_id"], "reason": "granularity-conflict",
                        "left_id": left["id"], "left_label": left["label"],
                        "right_id": right["id"], "right_label": right["label"],
                    })
    return conflicts


def _validate_clean_geometry(pages, policy):
    conflicts = geometry_conflicts(pages, policy)
    if conflicts:
        raise ValueError(
            f"Unresolved geometry conflicts ({len(conflicts)}): "
            + json.dumps(conflicts, ensure_ascii=False))


def adjudicate(evidence_path: str | Path, decisions_path: str | Path,
               ontology_path: str | Path, output_path: str | Path) -> dict:
    evidence_path, decisions_path, output_path = map(
        Path, (evidence_path, decisions_path, output_path))
    if output_path.exists():
        raise FileExistsError(output_path)
    files, archive_hash = load_evidence(evidence_path)
    ontology, ontology_hash = _load_ontology(ontology_path)
    run = json.loads(files["run.json"])
    policy = json.loads(files["consensus-policy.json"])
    categories = list(run["consensus"]["categories"])
    if set(categories) != set(ontology["classes"]):
        raise ValueError("Ontology categories do not match consensus evidence")
    consensus_rows = _jsonl(files["consensus/consensus.jsonl"])
    review_rows = _jsonl(files["consensus/review-queue.jsonl"])
    packet = json.loads(decisions_path.read_text(encoding="utf-8"))
    reviewer, decisions = _validate_packet(
        packet, archive_hash, ontology_hash, review_rows, set(categories))

    reviews_by_page = {}
    for row in review_rows:
        reviews_by_page.setdefault(row["page_id"], []).append(row)
    clean_pages = []
    reviewed_count = relabeled_count = rejected_count = 0
    for page in consensus_rows:
        objects = [{
            **item,
            "origin": "teacher-consensus",
        } for item in page["objects"]]
        for row in reviews_by_page.get(page["page_id"], []):
            decision = decisions[row["id"]]
            if decision["action"] == "reject":
                rejected_count += 1
                continue
            label = decision.get("label", row["label"])
            relabeled_count += decision["action"] == "relabel"
            reviewed_count += 1
            objects.append({
                "id": row["id"],
                "label": label,
                "bbox_xyxy": row["bbox_xyxy"],
                "teachers": row["teachers"],
                "mean_score": row["mean_score"],
                "score_kinds": row["score_kinds"],
                "origin": "human-review",
                "source_label": row["label"],
            })
        clean_pages.append({
            "schema": "slayer-layout-clean-candidate-v1",
            "page_id": page["page_id"],
            "image": page["image"],
            "objects": sorted(objects, key=lambda item: item["id"]),
        })
    _validate_clean_geometry(clean_pages, policy)

    category_ids = {label: index + 1 for index, label in enumerate(categories)}
    images, annotations = [], []
    annotation_id = 1
    for image_id, page in enumerate(clean_pages, 1):
        image = page["image"]
        images.append({
            "id": image_id,
            "file_name": image["file_name"],
            "width": image["width"],
            "height": image["height"],
            "sha256": image["sha256"],
            "page_id": page["page_id"],
        })
        for item in page["objects"]:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_ids[item["label"]],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "object_id": item["id"],
                "origin": item["origin"],
            })
            annotation_id += 1

    output_path.mkdir(parents=True)
    manifest_path = output_path / "annotations.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clean_pages),
        encoding="utf-8", newline="\n")
    coco_path = output_path / "annotations.coco.json"
    coco_path.write_text(json.dumps({
        "info": {
            "description": "SLAYER-OCR private reviewed layout candidate",
            "schema": "slayer-layout-clean-candidate-v1",
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_ids[label], "name": label} for label in categories
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    evidence_copy = output_path / "review-evidence.json"
    evidence_copy.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    report = {
        "schema": "slayer-layout-adjudication-report-v1",
        "release_status": "private-candidate-not-published",
        "source_evidence_sha256": archive_hash,
        "source_code_revision": run.get("code_revision"),
        "consensus_policy_sha256": run.get("consensus_policy_sha256"),
        "ontology_sha256": ontology_hash,
        "review_input_sha256": file_digest(decisions_path),
        "reviewer": reviewer,
        "pages": len(clean_pages),
        "consensus_objects": sum(len(page["objects"]) for page in consensus_rows),
        "accepted_review_objects": reviewed_count,
        "relabeled_review_objects": relabeled_count,
        "rejected_review_objects": rejected_count,
        "final_objects": len(annotations),
        "images_or_references_included": False,
        "limitations": [
            "Reviewer identity is self-reported and not authenticated.",
            "This candidate is not ground truth, a public release, or evidence of OCR accuracy.",
        ],
    }
    report_path = output_path / "adjudication.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    artifacts = [manifest_path, coco_path, evidence_copy, report_path]
    (output_path / "checksums.sha256").write_text(
        "".join(f"{file_digest(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(adjudicate(
        args.evidence, args.decisions, args.ontology, args.output), indent=2))


if __name__ == "__main__":
    main()
