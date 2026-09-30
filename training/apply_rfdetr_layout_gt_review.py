"""Apply a completed RF-DETR layout GT review to the frozen development split."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import zipfile


SCHEMA = "slayer-layout-rfdetr-dataset-v2"
REVIEW_SCHEMA = "slayer-layout-gt-review-v1"
SOURCE_SCHEMA = "slayer-layout-gt-review-source-v1"


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_checksums(root: Path) -> dict[str, str]:
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        raise ValueError(f"Missing checksum manifest: {checksum_path}")
    expected = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if (relative in expected or len(digest) != 64 or
                not path.is_relative_to(root.resolve()) or not path.is_file()):
            raise ValueError("Invalid checksum manifest")
        if file_digest(path) != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        expected[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if actual != set(expected):
        raise ValueError("Checksum manifest does not cover the complete directory")
    return expected


def _xyxy(bbox: list[float]) -> list[float]:
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def _same_box(left: list[float], right: list[float], tolerance: float = 0.002) -> bool:
    return all(math.isclose(a, b, abs_tol=tolerance) for a, b in zip(left, right))


def _validate_box(value: object, width: int, height: int) -> list[float]:
    if (not isinstance(value, list) or len(value) != 4 or
            any(isinstance(item, bool) or not isinstance(item, (int, float)) or
                not math.isfinite(item) for item in value)):
        raise ValueError("Annotation bbox_xyxy must contain four finite numbers")
    box = [float(item) for item in value]
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError(f"Annotation box outside page bounds: {box}")
    return box


def _validate_source(source: dict, source_dir: Path, dataset_dir: Path,
                     valid_coco: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    if source.get("schema") != SOURCE_SCHEMA:
        raise ValueError("Unexpected GT review source schema")
    category_by_id = {
        int(item["id"]): item["name"] for item in valid_coco["categories"]
    }
    categories = [category_by_id[key] for key in sorted(category_by_id)]
    if source.get("categories") != categories:
        raise ValueError("Review source category order differs from COCO")
    valid_coco_path = dataset_dir / "valid" / "_annotations.coco.json"
    if source.get("source_coco_sha256") != file_digest(valid_coco_path):
        raise ValueError("Review source points to a different validation COCO")

    images = {item["page_id"]: item for item in valid_coco["images"]}
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in valid_coco["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
    pages = source.get("pages")
    if not isinstance(pages, list) or len(pages) != len(images):
        raise ValueError("Review source page count differs from validation COCO")
    source_pages = {}
    for page in pages:
        page_id = page.get("page_id")
        if not isinstance(page_id, str) or page_id in source_pages or page_id not in images:
            raise ValueError("Review source contains an unknown or duplicate page")
        image = images[page_id]
        image_path = dataset_dir / "valid" / image["file_name"]
        if (page.get("width") != image["width"] or page.get("height") != image["height"] or
                page.get("image_sha256") != image["sha256"] or
                file_digest(image_path) != image["sha256"]):
            raise ValueError(f"Image contract mismatch for {page_id}")
        original = {item["id"]: item for item in page.get("original_annotations", [])}
        coco_original = {
            f"gt:{item['id']}": item
            for item in annotations_by_image.get(image["id"], [])
        }
        if set(original) != set(coco_original):
            raise ValueError(f"Original annotation IDs differ for {page_id}")
        for key, item in original.items():
            coco_item = coco_original[key]
            try:
                label = category_by_id[int(coco_item["category_id"])]
            except KeyError as error:
                raise ValueError(f"Unknown COCO category ID for {page_id}") from error
            if (item.get("source_annotation_id") != coco_item["id"] or
                    item.get("label") != label or
                    not _same_box(item.get("bbox_xyxy", []), _xyxy(coco_item["bbox"]))):
                raise ValueError(f"Original annotation contract mismatch for {page_id}")
        source_pages[page_id] = page
    if set(source_pages) != set(images):
        raise ValueError("Review source does not cover every validation page")
    return source_pages, images


def _validate_review(review: dict, source: dict, source_pages: dict[str, dict]) -> dict:
    if (review.get("schema") != REVIEW_SCHEMA or review.get("state") != "complete" or
            review.get("release_status") not in {
                "private-review-not-published", "public-review-candidate"
            }):
        raise ValueError("Review is not a completed GT review packet")
    for key in (
        "source_audit_sha256", "source_audit_report_sha256",
        "source_dataset_archive_sha256", "source_coco_sha256",
        "checkpoint_sha256", "policy_version",
    ):
        if review.get(key) != source.get(key):
            raise ValueError(f"Review provenance mismatch: {key}")
    reviewer = review.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Review requires a named reviewer")
    try:
        datetime.fromisoformat(str(review["timestamp"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as error:
        raise ValueError("Review timestamp is invalid") from error

    pages = review.get("pages")
    if not isinstance(pages, list) or len(pages) != len(source_pages):
        raise ValueError("Review must cover every source page exactly once")
    reviewed = {}
    categories = set(source["categories"])
    for page in pages:
        page_id = page.get("page_id")
        if page_id in reviewed or page_id not in source_pages:
            raise ValueError("Review contains an unknown or duplicate page")
        if page.get("status") not in {"verified", "edited"}:
            raise ValueError(f"Incomplete page status: {page_id}")
        if not isinstance(page.get("note", ""), str):
            raise ValueError(f"Page note must be text: {page_id}")
        source_page = source_pages[page_id]
        source_ids = {
            item["source_annotation_id"] for item in source_page["original_annotations"]
        }
        annotations = page.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"Annotations must be a list: {page_id}")
        annotation_ids = set()
        linked_source_ids = set()
        normalized = []
        for item in annotations:
            annotation_id = item.get("id")
            source_id = item.get("source_annotation_id")
            if not isinstance(annotation_id, str) or not annotation_id or annotation_id in annotation_ids:
                raise ValueError(f"Invalid or duplicate annotation ID: {page_id}")
            if source_id is not None:
                if (not isinstance(source_id, int) or source_id not in source_ids or
                        source_id in linked_source_ids):
                    raise ValueError(f"Invalid source annotation link: {page_id}")
                linked_source_ids.add(source_id)
            label = item.get("label")
            if label not in categories:
                raise ValueError(f"Unknown layout label: {label}")
            annotation_ids.add(annotation_id)
            normalized.append({
                "id": annotation_id,
                "source_annotation_id": source_id,
                "label": label,
                "bbox_xyxy": _validate_box(
                    item.get("bbox_xyxy"), source_page["width"], source_page["height"]),
                "origin": str(item.get("origin") or "human-review"),
            })
        actual_changed = _page_changed(source_page["original_annotations"], normalized)
        if bool(page.get("changed")) != actual_changed:
            raise ValueError(f"Review changed flag is inconsistent: {page_id}")
        expected_status = "edited" if actual_changed else "verified"
        if page["status"] != expected_status:
            raise ValueError(f"Review status does not match annotation changes: {page_id}")
        reviewed[page_id] = {**page, "annotations": normalized}
    if set(reviewed) != set(source_pages):
        raise ValueError("Review does not cover the frozen page set")
    return reviewed


def _page_changed(original: list[dict], reviewed: list[dict]) -> bool:
    source = {item["source_annotation_id"]: item for item in original}
    linked = {item["source_annotation_id"]: item for item in reviewed
              if item["source_annotation_id"] is not None}
    if set(source) != set(linked) or any(
        item["source_annotation_id"] is None for item in reviewed
    ):
        return True
    return any(
        source[key]["label"] != linked[key]["label"] or
        not _same_box(source[key]["bbox_xyxy"], linked[key]["bbox_xyxy"])
        for key in source
    )


def _corrected_coco(valid_coco: dict, reviewed: dict[str, dict]) -> tuple[dict, dict]:
    categories = {item["name"]: item["id"] for item in valid_coco["categories"]}
    original_by_id = {item["id"]: item for item in valid_coco["annotations"]}
    output_annotations = []
    report_pages = []
    annotation_id = 1
    for image in valid_coco["images"]:
        page = reviewed[image["page_id"]]
        before = [item for item in valid_coco["annotations"] if item["image_id"] == image["id"]]
        original_ids = {item["id"] for item in before}
        kept_ids = {item["source_annotation_id"] for item in page["annotations"]
                    if item["source_annotation_id"] is not None}
        added = sum(item["source_annotation_id"] is None for item in page["annotations"])
        relabeled = 0
        moved = 0
        for item in page["annotations"]:
            box = item["bbox_xyxy"]
            width, height = box[2] - box[0], box[3] - box[1]
            source_id = item["source_annotation_id"]
            base = dict(original_by_id[source_id]) if source_id is not None else {}
            if source_id is not None:
                original = original_by_id[source_id]
                relabeled += original["category_id"] != categories[item["label"]]
                moved += not _same_box(_xyxy(original["bbox"]), box)
            output_annotations.append({
                **base,
                "id": annotation_id,
                "image_id": image["id"],
                "category_id": categories[item["label"]],
                "bbox": [box[0], box[1], width, height],
                "area": width * height,
                "iscrowd": 0,
                "object_id": base.get("object_id") or
                    f"{image['page_id']}:gt-review-{annotation_id:04d}",
                "origin": "human-gt-review",
                "review_annotation_id": item["id"],
                "source_annotation_id": source_id,
            })
            annotation_id += 1
        report_pages.append({
            "page_id": image["page_id"],
            "status": page["status"],
            "before": len(before),
            "after": len(page["annotations"]),
            "added": added,
            "deleted": len(original_ids - kept_ids),
            "relabeled": relabeled,
            "moved": moved,
            "note": page.get("note", ""),
        })
    result = {
        **valid_coco,
        "info": {
            **valid_coco.get("info", {}),
            "description": "SLAYER-OCR RF-DETR validation split after GT review",
            "version": "2",
            "schema": SCHEMA,
        },
        "annotations": output_annotations,
    }
    totals = Counter()
    for page in report_pages:
        totals.update({key: page[key] for key in ("added", "deleted", "relabeled", "moved")})
    return result, {"pages": report_pages, "totals": dict(totals)}


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def _write_checksums(output: Path) -> None:
    files = sorted(path for path in output.rglob("*")
                   if path.is_file() and path.name != "checksums.sha256")
    (output / "checksums.sha256").write_text(
        "".join(f"{file_digest(path)}  {path.relative_to(output).as_posix()}\n"
                for path in files), encoding="utf-8", newline="\n")


def _archive(source: Path, archive_path: Path) -> None:
    if archive_path.exists():
        raise FileExistsError(archive_path)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            info = zipfile.ZipInfo(path.relative_to(source.parent).as_posix(),
                                   date_time=(2026, 9, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def apply_review(review_path: str | Path, review_source_dir: str | Path,
                 dataset_dir: str | Path, output_dir: str | Path,
                 archive_path: str | Path | None = None) -> dict:
    review_path = Path(review_path)
    source_dir = Path(review_source_dir)
    dataset_dir = Path(dataset_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    _verify_checksums(source_dir)
    _verify_checksums(dataset_dir)
    source = _json(source_dir / "source.json")
    review = _json(review_path)
    valid_coco = _json(dataset_dir / "valid" / "_annotations.coco.json")
    train_coco = _json(dataset_dir / "train" / "_annotations.coco.json")
    source_pages, _ = _validate_source(source, source_dir, dataset_dir, valid_coco)
    reviewed = _validate_review(review, source, source_pages)
    corrected_coco, change_report = _corrected_coco(valid_coco, reviewed)

    output.mkdir(parents=True)
    for split in ("train", "valid"):
        (output / split).mkdir()
        for image in (train_coco if split == "train" else valid_coco)["images"]:
            _link_or_copy(dataset_dir / split / image["file_name"],
                          output / split / image["file_name"])
    _write_json(output / "train" / "_annotations.coco.json", train_coco)
    _write_json(output / "valid" / "_annotations.coco.json", corrected_coco)
    shutil.copy2(review_path, output / "gt-review.json")
    shutil.copy2(source_dir / "source.json", output / "gt-review-source.json")
    _write_json(output / "change-report.json", change_report)

    manifest_rows = []
    for split, coco in (("train", train_coco), ("valid", corrected_coco)):
        counts = Counter(item["image_id"] for item in coco["annotations"])
        for image in coco["images"]:
            manifest_rows.append({
                "page_id": image["page_id"],
                "split": split,
                "image": f"{split}/{image['file_name']}",
                "sha256": image["sha256"],
                "width": image["width"],
                "height": image["height"],
                "objects": counts[image["id"]],
            })
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in manifest_rows), encoding="utf-8", newline="\n")
    provenance = {
        "schema": SCHEMA,
        "release_status": "public-development-dataset",
        "source": {
            "original_dataset_provenance_sha256": file_digest(dataset_dir / "provenance.json"),
            "original_valid_coco_sha256": source["source_coco_sha256"],
            "original_dataset_archive_sha256": source["source_dataset_archive_sha256"],
            "gt_review_sha256": file_digest(review_path),
            "gt_review_source_sha256": file_digest(source_dir / "source.json"),
            "source_audit_sha256": source["source_audit_sha256"],
            "source_audit_report_sha256": source["source_audit_report_sha256"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "policy_version": source["policy_version"],
            "reviewer": review["reviewer"],
            "review_timestamp": review["timestamp"],
        },
        "train": {"pages": len(train_coco["images"]),
                  "objects": len(train_coco["annotations"])},
        "valid": {"pages": len(corrected_coco["images"]),
                  "objects": len(corrected_coco["annotations"])},
        "changes": change_report["totals"],
        "collection_overlap": _json(dataset_dir / "provenance.json").get(
            "collection_overlap", []),
        "image_hash_overlap": _json(dataset_dir / "provenance.json").get(
            "image_hash_overlap", []),
        "intended_use": "Re-evaluate the frozen RF-DETR checkpoint against corrected development GT.",
        "limitations": [
            "The 12-page validation split remains development data, not a final test set.",
            "Ground truth reflects one completed human review under annotation policy v1.",
            "Model predictions were visible as an optional review layer and may influence adjudication.",
        ],
    }
    _write_json(output / "provenance.json", provenance)
    _write_checksums(output)
    if archive_path is not None:
        _archive(output, Path(archive_path))
        provenance["archive_sha256"] = file_digest(archive_path)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--review-source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--archive")
    args = parser.parse_args()
    result = apply_review(args.review, args.review_source, args.dataset,
                          args.output, args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
