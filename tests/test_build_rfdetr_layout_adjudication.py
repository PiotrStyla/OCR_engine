import hashlib
import json
from pathlib import Path

from PIL import Image

from training.build_rfdetr_layout_adjudication import build


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_builds_private_editable_review_workspace(tmp_path):
    dataset = tmp_path / "dataset"
    valid = dataset / "valid"
    valid.mkdir(parents=True)
    categories = [
        "text_region", "heading", "table", "figure", "caption",
        "marginalia", "header", "footer", "page_number",
    ]
    images, annotations, predictions = [], [], []
    for index in range(1, 13):
        name = f"page-{index}.jpg"
        image_path = valid / name
        Image.new("RGB", (120, 160), "white").save(image_path)
        image_hash = digest(image_path)
        images.append({
            "id": index, "page_id": f"p{index}", "file_name": name,
            "width": 120, "height": 160, "sha256": image_hash,
        })
        annotations.append({
            "id": index, "image_id": index, "category_id": 1,
            "bbox": [10, 20, 60, 80],
        })
        predictions.append({
            "schema": "slayer-layout-rfdetr-prediction-v1",
            "page_id": f"p{index}", "image_sha256": image_hash,
            "predictions": [{
                "label": "text_region", "score": 0.8,
                "bbox_xyxy": [10, 20, 70, 100],
            }],
            "audit": {"totals": {"tp": 1, "fp": index % 2, "fn": 0}},
        })
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name}
            for index, name in enumerate(categories, 1)],
    }
    (valid / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")

    audit = tmp_path / "audit"
    audit.mkdir()
    report = {
        "schema": "slayer-layout-rfdetr-audit-v1", "state": "completed",
        "pages": 12, "release_status": "private-audit-not-published",
        "input_archives": {"dataset": "d" * 64},
        "checkpoint_sha256": "c" * 64, "audit_threshold": 0.25,
    }
    files = {
        "audit.json": json.dumps(report).encode(),
        "dataset-provenance.json": json.dumps({"pages": 12}).encode(),
        "predictions.jsonl": "".join(
            json.dumps(row) + "\n" for row in predictions).encode(),
    }
    for name, content in files.items():
        (audit / name).write_bytes(content)
    (audit / "checksums.sha256").write_text("".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n"
        for name, content in files.items()), encoding="utf-8")

    output = tmp_path / "review"
    result = build(dataset, audit, output, "a" * 64)
    assert result["pages"] == 12
    assert result["annotations"] == 12
    assert result["release_status"] == "private-review-not-published"
    assert len(list((output / "images").glob("*.jpg"))) == 12
    html = (output / "index.html").read_text(encoding="utf-8")
    script = (output / "app.js").read_text(encoding="utf-8")
    assert "slayer-layout-gt-review-source-v1" in html
    assert "slayer-layout-annotation-policy-v1" in html
    assert "Export final" in script
    assert "pointerdown" in script
    assert "localStorage" in script
    assert "historical" not in script.lower()
