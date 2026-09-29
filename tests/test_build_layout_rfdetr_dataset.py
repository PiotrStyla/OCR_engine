import hashlib
import json

from PIL import Image
import pytest

from training.build_layout_rfdetr_dataset import build_dataset, file_digest


def _candidate(tmp_path):
    candidate = tmp_path / "candidate"
    images = tmp_path / "images"
    candidate.mkdir(parents=True)
    images.mkdir(parents=True)
    coco_images = []
    annotations = []
    annotation_id = 1
    for image_id, page_id in enumerate([
            "alpha__1", "alpha__2", "beta__1", "beta__2", "gamma__1", "delta__1"], 1):
        path = images / f"{page_id}.jpg"
        Image.new("RGB", (32, 24), (image_id * 20, 10, 10)).save(path)
        coco_images.append({
            "id": image_id, "page_id": page_id, "file_name": f"images/{path.name}",
            "width": 32, "height": 24, "sha256": file_digest(path),
        })
        annotations.append({
            "id": annotation_id, "image_id": image_id, "category_id": 1,
            "bbox": [1, 2, 10, 8], "area": 80, "iscrowd": 0,
        })
        annotation_id += 1
        if image_id % 2 == 0:
            annotations.append({
                "id": annotation_id, "image_id": image_id, "category_id": 2,
                "bbox": [2, 3, 8, 5], "area": 40, "iscrowd": 0,
            })
            annotation_id += 1
    coco = {
        "info": {"schema": "slayer-layout-clean-candidate-v1"},
        "images": coco_images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "text_region"}, {"id": 2, "name": "heading"}],
    }
    files = {
        "annotations.coco.json": json.dumps(coco),
        "annotations.jsonl": "{}\n",
        "review-evidence.json": "{}\n",
        "adjudication.json": json.dumps({
            "schema": "slayer-layout-adjudication-report-v1",
            "release_status": "private-candidate-not-published",
            "images_or_references_included": False,
            "review_input_sha256": "1" * 64,
            "source_evidence_sha256": "2" * 64,
            "source_code_revision": "3" * 40,
        }),
    }
    for name, content in files.items():
        (candidate / name).write_text(content, encoding="utf-8")
    (candidate / "checksums.sha256").write_text("".join(
        f"{file_digest(candidate / name)}  {name}\n" for name in files
    ), encoding="utf-8")
    return candidate, images


def _verify_checksums(root):
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        assert file_digest(root / relative) == expected


def test_builds_deterministic_collection_disjoint_rfdetr_dataset(tmp_path):
    candidate, images = _candidate(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = build_dataset(candidate, images, first, valid_pages=2, salt="fixed")
    build_dataset(candidate, images, second, valid_pages=2, salt="fixed")

    assert report["collection_overlap"] == []
    assert report["image_hash_overlap"] == []
    assert report["train"]["pages"] == 4
    assert report["valid"]["pages"] == 2
    assert report["train"]["collections"] == json.loads(
        (second / "provenance.json").read_text())["train"]["collections"]
    for split in ("train", "valid"):
        coco = json.loads((first / split / "_annotations.coco.json").read_text())
        assert coco["info"]["schema"] == "slayer-layout-rfdetr-dataset-v1"
        assert all("/" not in image["file_name"] for image in coco["images"])
        assert {path.name for path in (first / split).glob("*.jpg")} == {
            image["file_name"] for image in coco["images"]}
    _verify_checksums(first)
    assert not any(row["split"] == "valid" and row["collection"] in report["train"]["collections"]
                   for row in map(json.loads, (first / "manifest.jsonl").read_text().splitlines()))
    with pytest.raises(FileExistsError):
        build_dataset(candidate, images, first, valid_pages=2)


def test_rejects_tampered_candidate_or_image(tmp_path):
    candidate, images = _candidate(tmp_path)
    (candidate / "annotations.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_dataset(candidate, images, tmp_path / "out-a", valid_pages=2)

    candidate, images = _candidate(tmp_path / "fresh")
    image = next(images.glob("*.jpg"))
    image.write_bytes(image.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Image checksum mismatch"):
        build_dataset(candidate, images, tmp_path / "out-b", valid_pages=2)
