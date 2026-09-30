"""Audit a frozen RF-DETR checkpoint against original or corrected layout GT."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from PIL import Image, ImageDraw


SCHEMA = "slayer-layout-rfdetr-audit-v1"
COLORS = {
    "text_region": "#1f77b4",
    "heading": "#d62728",
    "table": "#9467bd",
    "figure": "#2ca02c",
    "caption": "#8c564b",
    "marginalia": "#e377c2",
    "header": "#17becf",
    "footer": "#bcbd22",
    "page_number": "#ff7f0e",
}


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_zip(archive_path: str | Path, output_dir: str | Path,
                     max_uncompressed: int = 1_000_000_000) -> Path:
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        if sum(member.file_size for member in members) > max_uncompressed:
            raise ValueError("ZIP exceeds the uncompressed size limit")
        for member in members:
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (path.is_absolute() or ".." in path.parts or "\\" in member.filename or
                    ":" in member.filename or (mode and stat.S_ISLNK(mode))):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        output_dir.mkdir(parents=True)
        archive.extractall(output_dir)
    return output_dir


def _one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name}, found {len(matches)}")
    return matches[0]


def verify_checksums(root: str | Path) -> dict[str, str]:
    root = Path(root).resolve()
    checksum_path = _one(root, "checksums.sha256")
    base = checksum_path.parent.resolve()
    expected = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = (base / relative).resolve()
        if (relative in expected or len(digest) != 64 or
                not path.is_relative_to(base) or not path.is_file()):
            raise ValueError("Invalid checksum manifest")
        if file_digest(path) != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        expected[relative] = digest
    actual = {
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if actual != set(expected):
        raise ValueError("Checksum manifest does not cover the complete archive")
    return expected


def _best_metric_row(metrics_path: Path) -> dict:
    rows = list(csv.DictReader(io.StringIO(
        metrics_path.read_text(encoding="utf-8-sig"))))
    scored = [row for row in rows if row.get("val/mAP_50_95")]
    if not scored:
        raise ValueError("Evidence contains no validation mAP")
    best = max(scored, key=lambda row: float(row["val/mAP_50_95"]))
    return {
        key: (int(value) if key == "epoch" else float(value))
        for key, value in best.items()
        if value and (key in {"epoch", "step"} or key.startswith("val/"))
    }


def validate_dataset_lineage(provenance: dict, model_run: dict,
                             dataset_archive_sha256: str,
                             checkpoint_sha256: str) -> str:
    """Return the audit mode after verifying dataset-to-checkpoint lineage."""
    schema = provenance.get("schema")
    if schema == "slayer-layout-rfdetr-dataset-v1":
        if model_run.get("dataset_zip_sha256") != dataset_archive_sha256:
            raise ValueError("Run manifest points to a different dataset")
        return "original-gt-reproduction"
    if schema == "slayer-layout-rfdetr-dataset-v2":
        source = provenance.get("source", {})
        if provenance.get("release_status") != "public-development-dataset":
            raise ValueError("Corrected dataset is not a released development artifact")
        if source.get("original_dataset_archive_sha256") != model_run.get(
                "dataset_zip_sha256"):
            raise ValueError("Corrected GT does not descend from the training dataset")
        if source.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("Corrected GT was reviewed against a different checkpoint")
        if not all(source.get(key) for key in (
                "gt_review_sha256", "gt_review_source_sha256",
                "source_audit_sha256", "source_audit_report_sha256",
                "policy_version", "reviewer", "review_timestamp")):
            raise ValueError("Corrected GT lineage is incomplete")
        return "corrected-gt-reanalysis"
    raise ValueError("Unexpected RF-DETR dataset schema")


def prepare_inputs(dataset_archive: str | Path, model_archive: str | Path,
                   evidence_archive: str | Path, output_dir: str | Path,
                   expected_hashes: dict[str, str]) -> dict:
    archives = {
        "dataset": Path(dataset_archive),
        "model": Path(model_archive),
        "evidence": Path(evidence_archive),
    }
    for name, path in archives.items():
        if file_digest(path) != expected_hashes[name]:
            raise ValueError(f"{name} archive SHA-256 mismatch")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    roots = {
        name: safe_extract_zip(path, output / name)
        for name, path in archives.items()
    }
    checksum_manifests = {
        name: verify_checksums(root) for name, root in roots.items()
    }
    dataset_root = _one(roots["dataset"], "provenance.json").parent
    model_root = _one(roots["model"], "checkpoint_best_total.pth").parent
    evidence_root = _one(roots["evidence"], "metrics.csv").parent
    provenance = json.loads((dataset_root / "provenance.json").read_text(encoding="utf-8"))
    model_run = json.loads((model_root / "run_manifest.json").read_text(encoding="utf-8"))
    evidence_run = json.loads(
        (evidence_root / "run_manifest.json").read_text(encoding="utf-8"))
    training_config = json.loads(
        (model_root / "training_config.json").read_text(encoding="utf-8"))
    if model_run != evidence_run:
        raise ValueError("Model and evidence run manifests differ")
    if (model_run.get("schema") != "slayer-layout-rfdetr-run-v1" or
            model_run.get("status") != "completed"):
        raise ValueError("Unexpected RF-DETR artifact schema or state")
    checkpoint_sha256 = checksum_manifests["model"]["checkpoint_best_total.pth"]
    audit_mode = validate_dataset_lineage(
        provenance, model_run, expected_hashes["dataset"], checkpoint_sha256)
    if (provenance.get("collection_overlap") or provenance.get("image_hash_overlap") or
            provenance.get("train", {}).get("pages") != 48 or
            provenance.get("valid", {}).get("pages") != 12):
        raise ValueError("Frozen dataset split contract mismatch")
    model_config = training_config.get("model_config", {})
    class_names = training_config.get("class_names")
    if (training_config.get("model_config_type") != "RFDETRSmallConfig" or
            model_config.get("resolution") != 512 or
            model_config.get("num_classes") != 9 or
            not isinstance(class_names, list) or len(class_names) != 9):
        raise ValueError("Unexpected trained model configuration")
    best = _best_metric_row(evidence_root / "metrics.csv")
    if best["epoch"] != 15 or not math.isclose(
            best["val/mAP_50_95"], 0.32729390263557434, abs_tol=1e-12):
        raise ValueError("Unexpected best checkpoint metric in evidence")
    return {
        "archives": archives,
        "archive_hashes": expected_hashes,
        "roots": roots,
        "dataset_root": dataset_root,
        "model_root": model_root,
        "evidence_root": evidence_root,
        "checkpoint": model_root / "checkpoint_best_total.pth",
        "checkpoint_sha256": checkpoint_sha256,
        "audit_mode": audit_mode,
        "provenance": provenance,
        "run_manifest": model_run,
        "training_config": training_config,
        "class_names": class_names,
        "best_logged_metrics": best,
    }


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def match_page(ground_truth: list[dict], predictions: list[dict],
               threshold: float = 0.25, iou_threshold: float = 0.5) -> dict:
    labels = sorted({item["label"] for item in ground_truth + predictions})
    per_class = {}
    for label in labels:
        truth = [item for item in ground_truth if item["label"] == label]
        predicted = sorted(
            (item for item in predictions
             if item["label"] == label and item["score"] >= threshold),
            key=lambda item: item["score"], reverse=True)
        unmatched = set(range(len(truth)))
        true_positives = 0
        for prediction in predicted:
            candidates = [(iou(prediction["bbox_xyxy"], truth[index]["bbox_xyxy"]), index)
                          for index in unmatched]
            overlap, index = max(candidates, default=(0.0, -1))
            if overlap >= iou_threshold:
                unmatched.remove(index)
                true_positives += 1
        per_class[label] = {
            "tp": true_positives,
            "fp": len(predicted) - true_positives,
            "fn": len(unmatched),
        }
    totals = Counter()
    for counts in per_class.values():
        totals.update(counts)
    return {"threshold": threshold, "iou_threshold": iou_threshold,
            "totals": dict(totals), "per_class": per_class}


def _draw_boxes(image: Image.Image, objects: list[dict], scale: float,
                show_score: bool) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for item in objects:
        color = COLORS.get(item["label"], "#ffffff")
        box = [round(value * scale) for value in item["bbox_xyxy"]]
        draw.rectangle(box, outline=color, width=3)
        label = item["label"]
        if show_score:
            label += f" {item['score']:.2f}"
        text_box = draw.textbbox((box[0], box[1]), label)
        draw.rectangle(text_box, fill=color)
        draw.text((box[0], box[1]), label, fill="black")
    return result


def render_comparison(image_path: str | Path, ground_truth: list[dict],
                      predictions: list[dict], output_path: str | Path,
                      threshold: float = 0.25, max_width: int = 900) -> None:
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    scale = min(1.0, max_width / source.width)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.LANCZOS)
    predicted = [item for item in predictions if item["score"] >= threshold]
    left = _draw_boxes(resized, ground_truth, scale, False)
    right = _draw_boxes(resized, predicted, scale, True)
    title_height, gap = 28, 12
    canvas = Image.new(
        "RGB", (left.width * 2 + gap, left.height + title_height), "white")
    canvas.paste(left, (0, title_height))
    canvas.paste(right, (left.width + gap, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "Ground truth", fill="black")
    draw.text((left.width + gap + 8, 7), f"RF-DETR predictions >= {threshold:.2f}",
              fill="black")
    canvas.save(output_path, format="JPEG", quality=88, optimize=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def resolve_prediction_labels(
        class_ids, provided_names, class_names: list[str]) -> list[str | None]:
    ids = [int(class_id) for class_id in class_ids]
    if provided_names is not None:
        names = [str(name) for name in provided_names]
        if len(names) != len(ids):
            raise ValueError("Prediction class IDs and names have different lengths")
        if all(name in class_names or name == "__background__" for name in names):
            for class_id, name in zip(ids, names):
                if name == "__background__" and class_id != len(class_names):
                    raise ValueError(
                        f"Unexpected background class ID: {class_id}; "
                        f"expected {len(class_names)}")
            return [None if name == "__background__" else name for name in names]

    if not ids:
        return []
    if 0 in ids:
        if not all(0 <= class_id <= len(class_names) for class_id in ids):
            raise ValueError(f"Invalid zero-based predicted class IDs: {ids}")
        return [
            None if class_id == len(class_names) else class_names[class_id]
            for class_id in ids
        ]
    if len(class_names) in ids:
        if not all(1 <= class_id <= len(class_names) for class_id in ids):
            raise ValueError(f"Invalid one-based predicted class IDs: {ids}")
        return [class_names[class_id - 1] for class_id in ids]
    raise ValueError(
        "Ambiguous predicted class ID scheme; model output omitted usable class names"
    )


def run(dataset_archive: str | Path, model_archive: str | Path,
        evidence_archive: str | Path, expected_hashes: dict[str, str],
        output_root: str | Path = "/content", code_revision: str | None = None,
        prediction_threshold: float = 0.05, audit_threshold: float = 0.25) -> tuple[Path, dict]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    work = Path(output_root) / f"slayer-rfdetr-layout-audit-{timestamp}"
    inputs = prepare_inputs(
        dataset_archive, model_archive, evidence_archive, work / "inputs",
        expected_hashes)
    import torch
    from rfdetr import RFDETRSmall

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime and run the notebook from the start")
    model_config = inputs["training_config"]["model_config"]
    model = RFDETRSmall(
        pretrain_weights=str(inputs["checkpoint"]),
        resolution=model_config["resolution"],
        num_classes=model_config["num_classes"],
    )
    best_metrics = _jsonable(model.evaluate(
        dataset_dir=str(inputs["dataset_root"]), split="val", eval_batch_size=4,
        device="cuda", log_per_class_metrics=True, tensorboard=False, wandb=False))
    measured_map = best_metrics.get("val/mAP_50_95")
    expected_map = inputs["best_logged_metrics"]["val/mAP_50_95"]
    if measured_map is None:
        raise ValueError("RF-DETR evaluation returned no validation mAP")
    if (inputs["audit_mode"] == "original-gt-reproduction" and not math.isclose(
            measured_map, expected_map, rel_tol=0.0, abs_tol=5e-4)):
        raise ValueError(
            f"Best checkpoint mAP mismatch: measured={measured_map}, expected={expected_map}")

    valid_coco = json.loads(
        (inputs["dataset_root"] / "valid" / "_annotations.coco.json").read_text(
            encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in valid_coco["categories"]}
    annotations_by_image = {}
    for annotation in valid_coco["annotations"]:
        x, y, width, height = annotation["bbox"]
        annotations_by_image.setdefault(annotation["image_id"], []).append({
            "id": annotation["id"],
            "label": categories[annotation["category_id"]],
            "bbox_xyxy": [x, y, x + width, y + height],
        })

    output = work / "evidence"
    overlays = output / "overlays"
    overlays.mkdir(parents=True)
    prediction_rows, hard_examples = [], []
    aggregate = Counter()
    background_detections_discarded = 0
    per_class = {}
    class_names = inputs["class_names"]
    for image_info in valid_coco["images"]:
        image_path = inputs["dataset_root"] / "valid" / image_info["file_name"]
        with Image.open(image_path) as opened:
            prediction = model.predict(
                opened.convert("RGB"), threshold=prediction_threshold,
                include_source_image=False)
        names = prediction.data.get("class_name")
        labels = resolve_prediction_labels(prediction.class_id, names, class_names)
        page_background_detections = sum(label is None for label in labels)
        background_detections_discarded += page_background_detections
        predictions = []
        for box, score, class_id, label in zip(
                prediction.xyxy, prediction.confidence, prediction.class_id, labels):
            if label is None:
                continue
            predictions.append({
                "label": label,
                "class_id": int(class_id),
                "score": float(score),
                "bbox_xyxy": [float(value) for value in box],
            })
        truth = annotations_by_image.get(image_info["id"], [])
        matching = match_page(truth, predictions, audit_threshold, 0.5)
        aggregate.update(matching["totals"])
        for label, counts in matching["per_class"].items():
            per_class.setdefault(label, Counter()).update(counts)
        page_row = {
            "schema": "slayer-layout-rfdetr-prediction-v1",
            "page_id": image_info["page_id"],
            "image_sha256": image_info["sha256"],
            "raw_threshold": prediction_threshold,
            "background_detections_discarded": page_background_detections,
            "predictions": predictions,
            "audit": matching,
        }
        prediction_rows.append(page_row)
        error_count = matching["totals"].get("fp", 0) + matching["totals"].get("fn", 0)
        if error_count:
            hard_examples.append({
                "page_id": image_info["page_id"],
                "errors": error_count,
                "audit": matching,
                "reasons": ["rfdetr-false-positive"] * bool(matching["totals"].get("fp"))
                + ["rfdetr-false-negative"] * bool(matching["totals"].get("fn")),
            })
        render_comparison(
            image_path, truth, predictions,
            overlays / f"{image_info['page_id']}.jpg", audit_threshold)

    hard_examples.sort(key=lambda item: (-item["errors"], item["page_id"]))
    report = {
        "schema": SCHEMA,
        "state": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision,
        "input_archives": inputs["archive_hashes"],
        "checkpoint_sha256": inputs["checkpoint_sha256"],
        "audit_mode": inputs["audit_mode"],
        "best_logged_metrics": inputs["best_logged_metrics"],
        "best_checkpoint_metrics": best_metrics,
        "pages": len(valid_coco["images"]),
        "prediction_threshold": prediction_threshold,
        "audit_threshold": audit_threshold,
        "iou_threshold": 0.5,
        "fixed_threshold_counts": dict(aggregate),
        "fixed_threshold_per_class": {
            label: dict(counts) for label, counts in sorted(per_class.items())},
        "background_detections_discarded": background_detections_discarded,
        "hard_example_pages": len(hard_examples),
        "images_or_references_included": True,
        "release_status": (
            "public-experiment-evidence"
            if inputs["audit_mode"] == "corrected-gt-reanalysis"
            else "private-audit-not-published"
        ),
        "limitations": [
            "The 12-page valid split is internal development, not the final PolOCRBench test.",
            "Fixed-threshold TP/FP/FN is diagnostic and is not COCO mAP.",
            "Rare and absent valid classes do not support reliable per-class conclusions.",
            "Overlay images and predictions are experiment evidence, not final benchmark results.",
        ],
    }
    (output / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "best-checkpoint-metrics.json").write_text(
        json.dumps(best_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8", newline="\n")
    (output / "hard-examples.json").write_text(
        json.dumps(hard_examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "dataset-provenance.json").write_text(
        json.dumps(inputs["provenance"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    artifacts = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "checksums.sha256").write_text(
        "".join(f"{file_digest(path)}  {path.relative_to(output).as_posix()}\n"
                for path in artifacts), encoding="utf-8", newline="\n")
    archive = Path(shutil.make_archive(
        str(work / "slayer-rfdetr-layout-audit-v1"), "zip", output))
    return archive, report

