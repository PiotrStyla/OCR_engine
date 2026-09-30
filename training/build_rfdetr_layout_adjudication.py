"""Build an offline editor for RF-DETR development-page ground truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from training.audit_slayer_rfdetr_layout import COLORS, file_digest, verify_checksums


SCHEMA = "slayer-layout-gt-review-source-v1"
POLICY_VERSION = "slayer-layout-annotation-policy-v1"
POLICY = [
    "text_region: jeden spójny blok lub akapit; nie dziel wiersz po wierszu, "
    "jeżeli należą do tego samego przepływu.",
    "table: obejmuje całą strukturę tabelaryczną i nie jest dublowana przez "
    "nakładający się text_region.",
    "figure: samodzielna ilustracja lub ornament; inicjał osadzony w tekście "
    "pozostaje częścią text_region.",
    "Stemple biblioteczne, palce, ramy skanu i zabrudzenia są artefaktami i nie "
    "otrzymują klasy figure.",
    "marginalia: semantyczny tekst poza głównym przepływem, nie przypadkowy znak "
    "ani artefakt skanu.",
    "heading i page_number: ramka ciasna, bez sąsiedniego tekstu; header/footer "
    "tylko dla powtarzalnych elementów strony.",
]


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _copy_assets(output: Path) -> None:
    source = Path(__file__).with_name("rfdetr_layout_review")
    for name in ("app.js", "styles.css"):
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copyfile(path, output / name)


def build(dataset_dir: str | Path, audit_dir: str | Path,
          output_dir: str | Path, audit_archive_sha256: str) -> dict:
    dataset_dir, audit_dir, output_dir = map(
        Path, (dataset_dir, audit_dir, output_dir))
    dataset_dir, audit_dir = dataset_dir.resolve(), audit_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if len(audit_archive_sha256) != 64:
        raise ValueError("Invalid audit archive SHA-256")
    verify_checksums(audit_dir)

    report = _json(audit_dir / "audit.json")
    if (report.get("schema") != "slayer-layout-rfdetr-audit-v1" or
            report.get("state") != "completed" or report.get("pages") != 12 or
            report.get("release_status") != "private-audit-not-published"):
        raise ValueError("Unexpected or incomplete RF-DETR audit")
    provenance = _json(audit_dir / "dataset-provenance.json")
    valid_dir = dataset_dir / "valid"
    coco_path = valid_dir / "_annotations.coco.json"
    coco = _json(coco_path)
    predictions = _jsonl(audit_dir / "predictions.jsonl")
    if len(coco.get("images", [])) != 12 or len(predictions) != 12:
        raise ValueError("Expected exactly 12 development pages")

    categories = {int(item["id"]): item["name"] for item in coco["categories"]}
    category_names = [categories[key] for key in sorted(categories)]
    if category_names != list(COLORS):
        raise ValueError("Layout ontology does not match the frozen class order")
    prediction_by_page = {row["page_id"]: row for row in predictions}
    if len(prediction_by_page) != len(predictions):
        raise ValueError("Duplicate prediction page IDs")
    annotations = {}
    for item in coco["annotations"]:
        x, y, width, height = map(float, item["bbox"])
        annotations.setdefault(int(item["image_id"]), []).append({
            "id": f"gt:{item['id']}",
            "source_annotation_id": item["id"],
            "label": categories[int(item["category_id"])],
            "bbox_xyxy": [x, y, x + width, y + height],
            "origin": "frozen-development-gt",
        })

    output_dir.mkdir(parents=True)
    image_output = output_dir / "images"
    image_output.mkdir()
    pages = []
    try:
        for image in coco["images"]:
            page_id = image["page_id"]
            prediction = prediction_by_page.get(page_id)
            if (prediction is None or prediction.get("image_sha256") != image["sha256"]):
                raise ValueError(f"Prediction/image mismatch: {page_id}")
            source = valid_dir / image["file_name"]
            if not source.is_file() or file_digest(source) != image["sha256"]:
                raise ValueError(f"Missing or mismatched development image: {page_id}")
            target = image_output / Path(image["file_name"]).name
            if target.exists():
                raise ValueError(f"Duplicate image basename: {target.name}")
            shutil.copyfile(source, target)
            audit = prediction["audit"]
            pages.append({
                "page_id": page_id,
                "image_src": f"images/{target.name}",
                "image_sha256": image["sha256"],
                "width": int(image["width"]),
                "height": int(image["height"]),
                "original_annotations": annotations.get(int(image["id"]), []),
                "predictions": prediction["predictions"],
                "audit": audit,
                "error_count": int(audit["totals"].get("fp", 0))
                + int(audit["totals"].get("fn", 0)),
            })
        pages.sort(key=lambda page: (-page["error_count"], page["page_id"]))
        payload = {
            "schema": SCHEMA,
            "policy_version": POLICY_VERSION,
            "policy": POLICY,
            "source_audit_sha256": audit_archive_sha256,
            "source_audit_report_sha256": file_digest(audit_dir / "audit.json"),
            "source_dataset_archive_sha256": report["input_archives"]["dataset"],
            "source_coco_sha256": file_digest(coco_path),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "categories": category_names,
            "colors": COLORS,
            "default_prediction_threshold": report["audit_threshold"],
            "pages": pages,
            "provenance": provenance,
            "release_status": "public-review-candidate",
        }
        embedded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
        html = f'''<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SLAYER layout GT review</title><link rel="stylesheet" href="styles.css"></head>
<body><div id="app"></div><script>window.REVIEW_DATA={embedded};</script><script src="app.js"></script></body></html>'''
        (output_dir / "index.html").write_text(html, encoding="utf-8", newline="\n")
        _copy_assets(output_dir)
        (output_dir / "source.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "README.txt").write_text(
            "Offline layout GT review. Open index.html. Export a draft at any "
            "time; final export requires every page to be verified or edited.\n",
            encoding="utf-8", newline="\n")
        artifacts = sorted(path for path in output_dir.rglob("*") if path.is_file())
        checksum_path = output_dir / "checksums.sha256"
        checksum_path.write_text("".join(
            f"{file_digest(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in artifacts), encoding="utf-8", newline="\n")
    except Exception:
        shutil.rmtree(output_dir)
        raise
    return {
        "schema": SCHEMA,
        "pages": len(pages),
        "annotations": sum(len(page["original_annotations"]) for page in pages),
        "output": str(output_dir / "index.html"),
        "release_status": "public-review-candidate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-archive-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        args.dataset, args.audit, args.output, args.audit_archive_sha256),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
