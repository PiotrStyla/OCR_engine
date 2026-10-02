import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from training.apply_rfdetr_layout_gt_review import apply_review


CLASSES = [
    "text_region", "heading", "table", "figure", "caption",
    "marginalia", "header", "footer", "page_number",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_checksums(root: Path) -> None:
    files = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.name != "checksums.sha256")
    (root / "checksums.sha256").write_text(
        "".join(f"{digest(path)}  {path.relative_to(root).as_posix()}\n"
                for path in files), encoding="utf-8")


def coco(images, annotations):
    return {
        "info": {"schema": "slayer-layout-rfdetr-dataset-v1"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index, "name": name, "supercategory": "layout"}
                       for index, name in enumerate(CLASSES, 1)],
    }


def fixture(tmp_path: Path):
    dataset = tmp_path / "dataset"
    source_dir = tmp_path / "source"
    for split in ("train", "valid"):
        (dataset / split).mkdir(parents=True)
    train_image = dataset / "train" / "train.jpg"
    valid_a = dataset / "valid" / "page-a.jpg"
    valid_b = dataset / "valid" / "page-b.jpg"
    train_image.write_bytes(b"train-image")
    valid_a.write_bytes(b"valid-a")
    valid_b.write_bytes(b"valid-b")
    train = coco(
        [{"id": 1, "page_id": "Train__1", "file_name": "train.jpg",
          "width": 100, "height": 100, "sha256": digest(train_image)}],
        [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 20, 20],
          "area": 400, "iscrowd": 0, "object_id": "train:1", "origin": "review"}],
    )
    valid = coco(
        [
            {"id": 1, "page_id": "Page__A", "file_name": "page-a.jpg",
             "width": 100, "height": 100, "sha256": digest(valid_a)},
            {"id": 2, "page_id": "Page__B", "file_name": "page-b.jpg",
             "width": 120, "height": 80, "sha256": digest(valid_b)},
        ],
        [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 20],
             "area": 600, "iscrowd": 0, "object_id": "a:1", "origin": "review"},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [5, 5, 40, 10],
             "area": 400, "iscrowd": 0, "object_id": "b:1", "origin": "review"},
        ],
    )
    write_json(dataset / "train" / "_annotations.coco.json", train)
    write_json(dataset / "valid" / "_annotations.coco.json", valid)
    write_json(dataset / "provenance.json", {
        "schema": "slayer-layout-rfdetr-dataset-v1",
        "collection_overlap": [], "image_hash_overlap": [],
    })
    (dataset / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    write_checksums(dataset)

    source_dir.mkdir()
    source = {
        "schema": "slayer-layout-gt-review-source-v1",
        "policy_version": "slayer-layout-annotation-policy-v1",
        "source_audit_sha256": "a" * 64,
        "source_audit_report_sha256": "b" * 64,
        "source_dataset_archive_sha256": "c" * 64,
        "source_coco_sha256": digest(dataset / "valid" / "_annotations.coco.json"),
        "checkpoint_sha256": "d" * 64,
        "categories": CLASSES,
        "pages": [
            {"page_id": "Page__A", "image_sha256": digest(valid_a),
             "width": 100, "height": 100,
             "original_annotations": [{"id": "gt:1", "source_annotation_id": 1,
                 "label": "text_region", "bbox_xyxy": [10, 10, 40, 30]}]},
            {"page_id": "Page__B", "image_sha256": digest(valid_b),
             "width": 120, "height": 80,
             "original_annotations": [{"id": "gt:2", "source_annotation_id": 2,
                 "label": "heading", "bbox_xyxy": [5, 5, 45, 15]}]},
        ],
    }
    write_json(source_dir / "source.json", source)
    (source_dir / "README.txt").write_text("review source", encoding="utf-8")
    write_checksums(source_dir)
    review = {
        "schema": "slayer-layout-gt-review-v1",
        "state": "complete",
        "release_status": "public-review-candidate",
        **{key: source[key] for key in (
            "policy_version", "source_audit_sha256", "source_audit_report_sha256",
            "source_dataset_archive_sha256", "source_coco_sha256", "checkpoint_sha256")},
        "reviewer": "tester",
        "timestamp": "2026-09-30T12:00:00Z",
        "pages": [
            {"page_id": "Page__A", "status": "verified", "note": "", "changed": False,
             "annotations": [{"id": "gt:1", "source_annotation_id": 1,
                 "label": "text_region", "bbox_xyxy": [10, 10, 40, 30],
                 "origin": "frozen-development-gt"}]},
            {"page_id": "Page__B", "status": "edited", "note": "fixed", "changed": True,
             "annotations": [
                 {"id": "gt:2", "source_annotation_id": 2, "label": "text_region",
                  "bbox_xyxy": [6, 6, 46, 18], "origin": "human-review"},
                 {"id": "manual:1", "source_annotation_id": None, "label": "page_number",
                  "bbox_xyxy": [100, 60, 115, 75], "origin": "human-review"},
             ]},
        ],
    }
    review_path = tmp_path / "review.json"
    write_json(review_path, review)
    return dataset, source_dir, review_path


def test_apply_review_builds_corrected_dataset_and_archive(tmp_path):
    dataset, source_dir, review_path = fixture(tmp_path)
    output = tmp_path / "corrected"
    archive = tmp_path / "corrected.zip"
    report = apply_review(review_path, source_dir, dataset, output, archive)

    corrected = json.loads(
        (output / "valid" / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert report["schema"] == "slayer-layout-rfdetr-dataset-v2"
    assert report["valid"] == {"pages": 2, "objects": 3}
    assert report["changes"] == {"added": 1, "deleted": 0, "relabeled": 1, "moved": 1}
    assert corrected["info"]["schema"] == "slayer-layout-rfdetr-dataset-v2"
    assert [item["category_id"] for item in corrected["annotations"]] == [1, 1, 9]
    assert all(item["origin"] == "human-gt-review" for item in corrected["annotations"])
    assert archive.is_file()
    with zipfile.ZipFile(archive) as bundle:
        assert "corrected/provenance.json" in bundle.namelist()
    expected = {
        line.split("  ", 1)[1]
        for line in (output / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    }
    actual = {
        path.relative_to(output).as_posix() for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    assert expected == actual


def test_apply_review_accepts_verified_pages_with_edits(tmp_path):
    dataset, source_dir, review_path = fixture(tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["pages"][1]["status"] = "verified"
    write_json(review_path, review)
    report = apply_review(
        review_path, source_dir, dataset, tmp_path / "output")
    assert report["changes"] == {
        "added": 1, "deleted": 0, "relabeled": 1, "moved": 1}


def test_apply_review_rejects_incomplete_page_status(tmp_path):
    dataset, source_dir, review_path = fixture(tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["pages"][1]["status"] = "pending"
    write_json(review_path, review)
    with pytest.raises(ValueError, match="Incomplete page status"):
        apply_review(review_path, source_dir, dataset, tmp_path / "output")
