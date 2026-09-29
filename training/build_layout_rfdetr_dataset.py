"""Build a private, checksum-verified RF-DETR layout training dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import shutil

from PIL import Image


SCHEMA = "slayer-layout-rfdetr-dataset-v1"
DEFAULT_SALT = "slayer-layout-rfdetr-private-v1"


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checksums(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in result or len(digest) != 64:
            raise ValueError("Invalid or duplicate candidate checksum entry")
        result[name] = digest
    return result


def _verify_candidate(candidate: Path) -> tuple[dict, dict]:
    required = {
        "adjudication.json", "annotations.coco.json", "annotations.jsonl",
        "review-evidence.json",
    }
    checksums = _load_checksums(candidate / "checksums.sha256")
    if set(checksums) != required:
        raise ValueError("Candidate checksum manifest does not cover expected artifacts")
    for name, expected in checksums.items():
        if file_digest(candidate / name) != expected:
            raise ValueError(f"Candidate checksum mismatch: {name}")
    report = json.loads((candidate / "adjudication.json").read_text(encoding="utf-8"))
    if (report.get("schema") != "slayer-layout-adjudication-report-v1" or
            report.get("release_status") != "private-candidate-not-published" or
            report.get("images_or_references_included") is not False):
        raise ValueError("Expected a private adjudicated layout candidate")
    coco = json.loads((candidate / "annotations.coco.json").read_text(encoding="utf-8"))
    if coco.get("info", {}).get("schema") != "slayer-layout-clean-candidate-v1":
        raise ValueError("Unexpected candidate COCO schema")
    return coco, report


def _collection(page_id: str) -> str:
    if not isinstance(page_id, str) or "__" not in page_id:
        raise ValueError(f"Page ID does not identify a collection: {page_id!r}")
    return page_id.split("__", 1)[0]


@dataclass(frozen=True)
class CollectionStats:
    pages: int
    categories: Counter


def _select_valid_collections(images: list[dict], annotations: list[dict],
                              valid_pages: int, salt: str) -> tuple[set[str], dict]:
    if not 0 < valid_pages < len(images):
        raise ValueError("valid_pages must leave at least one page in each split")
    image_by_id = {image["id"]: image for image in images}
    if len(image_by_id) != len(images):
        raise ValueError("Duplicate COCO image IDs")
    stats: dict[str, CollectionStats] = {}
    page_counts = Counter(_collection(image["page_id"]) for image in images)
    category_counts: dict[str, Counter] = {name: Counter() for name in page_counts}
    for annotation in annotations:
        image = image_by_id.get(annotation.get("image_id"))
        if image is None:
            raise ValueError("Annotation references an unknown image")
        category_counts[_collection(image["page_id"])][annotation["category_id"]] += 1
    for name in page_counts:
        stats[name] = CollectionStats(page_counts[name], category_counts[name])

    names = sorted(stats)
    if len(names) < 2:
        raise ValueError("Collection-disjoint split requires at least two collections")
    totals = Counter()
    category_collections: dict[int, set[str]] = {}
    for name, item in stats.items():
        totals.update(item.categories)
        for category_id, count in item.categories.items():
            if count:
                category_collections.setdefault(category_id, set()).add(name)
    dual_eligible = {
        category_id for category_id, collections in category_collections.items()
        if len(collections) >= 2
    }

    min_pages = min(item.pages for item in stats.values())
    max_size = min(len(names) - 1, max(1, math.ceil(valid_pages / min_pages) + 2))
    best = None
    target_ratio = valid_pages / len(images)
    for size in range(1, max_size + 1):
        for selected in combinations(names, size):
            pages = sum(stats[name].pages for name in selected)
            counts = Counter()
            for name in selected:
                counts.update(stats[name].categories)
            missing = sum(
                counts[category_id] == 0 or counts[category_id] == totals[category_id]
                for category_id in dual_eligible
            )
            balance = sum(
                abs(counts[category_id] - total * target_ratio) / math.sqrt(total)
                for category_id, total in totals.items()
            )
            tie = hashlib.sha256(
                (salt + "\0" + "\0".join(selected)).encode("utf-8")
            ).hexdigest()
            score = (abs(pages - valid_pages), missing, round(balance, 12), tie)
            if best is None or score < best[0]:
                best = (score, set(selected), pages, counts)
    if best is None:
        raise ValueError("Could not produce a collection-disjoint split")
    _, selected, pages, counts = best
    split_info = {
        "algorithm": "exhaustive collection subsets with page, coverage, balance, hash tie-break",
        "salt": salt,
        "requested_valid_pages": valid_pages,
        "actual_valid_pages": pages,
        "target_valid_ratio": target_ratio,
        "dual_split_eligible_category_ids": sorted(dual_eligible),
        "valid_category_counts": {str(key): counts[key] for key in sorted(totals)},
        "total_category_counts": {str(key): totals[key] for key in sorted(totals)},
    }
    return selected, split_info


def _validate_coco(coco: dict, images_dir: Path) -> dict[int, Path]:
    images = coco.get("images")
    annotations = coco.get("annotations")
    categories = coco.get("categories")
    if not all(isinstance(value, list) and value for value in
               (images, annotations, categories)):
        raise ValueError("COCO images, annotations and categories must be non-empty")
    category_ids = {item.get("id") for item in categories}
    if len(category_ids) != len(categories) or None in category_ids:
        raise ValueError("Invalid COCO category IDs")
    image_ids = set()
    page_ids = set()
    image_hashes = set()
    sources = {}
    root = images_dir.resolve()
    for image in images:
        image_id = image.get("id")
        page_id = image.get("page_id")
        basename = Path(str(image.get("file_name", ""))).name
        source = (images_dir / basename).resolve()
        if (image_id in image_ids or page_id in page_ids or not basename or
                source.parent != root or not source.is_file()):
            raise ValueError("Missing, duplicate or unsafe source image")
        digest = file_digest(source)
        if digest != image.get("sha256") or digest in image_hashes:
            raise ValueError(f"Image checksum mismatch or duplicate: {page_id}")
        with Image.open(source) as opened:
            if opened.width != image.get("width") or opened.height != image.get("height"):
                raise ValueError(f"Image dimensions mismatch: {page_id}")
            opened.verify()
        image_ids.add(image_id)
        page_ids.add(page_id)
        image_hashes.add(digest)
        sources[image_id] = source
    annotation_ids = set()
    for annotation in annotations:
        bbox = annotation.get("bbox")
        if (annotation.get("id") in annotation_ids or
                annotation.get("image_id") not in image_ids or
                annotation.get("category_id") not in category_ids or
                not isinstance(bbox, list) or len(bbox) != 4 or
                any(not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in bbox) or
                bbox[2] <= 0 or bbox[3] <= 0):
            raise ValueError("Invalid COCO annotation")
        annotation_ids.add(annotation["id"])
    return sources


def _split_coco(coco: dict, selected_ids: set, split: str) -> dict:
    source_images = [image for image in coco["images"] if image["id"] in selected_ids]
    image_ids = {image["id"] for image in source_images}
    source_annotations = [
        annotation for annotation in coco["annotations"]
        if annotation["image_id"] in image_ids
    ]
    remap = {image["id"]: index for index, image in enumerate(source_images, 1)}
    images = [{
        **image,
        "id": remap[image["id"]],
        "file_name": Path(image["file_name"]).name,
    } for image in source_images]
    annotations = [{
        **annotation,
        "id": index,
        "image_id": remap[annotation["image_id"]],
    } for index, annotation in enumerate(source_annotations, 1)]
    return {
        "info": {
            "description": f"SLAYER-OCR private RF-DETR {split} split",
            "version": "1",
            "schema": SCHEMA,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def _summary(coco: dict) -> dict:
    names = {item["id"]: item["name"] for item in coco["categories"]}
    counts = Counter(item["category_id"] for item in coco["annotations"])
    return {
        "pages": len(coco["images"]),
        "objects": len(coco["annotations"]),
        "collections": sorted({_collection(item["page_id"]) for item in coco["images"]}),
        "classes": {names[key]: counts[key] for key in sorted(names)},
    }


def build_dataset(candidate_dir: str | Path, images_dir: str | Path,
                  output_dir: str | Path, valid_pages: int = 12,
                  salt: str = DEFAULT_SALT) -> dict:
    candidate, images_dir, output = map(Path, (candidate_dir, images_dir, output_dir))
    if output.exists():
        raise FileExistsError(output)
    coco, adjudication = _verify_candidate(candidate)
    sources = _validate_coco(coco, images_dir)
    valid_collections, split_info = _select_valid_collections(
        coco["images"], coco["annotations"], valid_pages, salt)
    valid_ids = {
        image["id"] for image in coco["images"]
        if _collection(image["page_id"]) in valid_collections
    }
    train_ids = {image["id"] for image in coco["images"]} - valid_ids
    split_cocos = {
        "train": _split_coco(coco, train_ids, "train"),
        "valid": _split_coco(coco, valid_ids, "valid"),
    }

    output.mkdir(parents=True)
    manifest_rows = []
    for split, split_coco in split_cocos.items():
        split_dir = output / split
        split_dir.mkdir()
        source_id_by_page = {item["page_id"]: item["id"] for item in coco["images"]}
        annotations_by_image = Counter(
            item["image_id"] for item in split_coco["annotations"])
        classes_by_image: dict[int, Counter] = {}
        category_names = {item["id"]: item["name"] for item in coco["categories"]}
        for annotation in split_coco["annotations"]:
            classes_by_image.setdefault(annotation["image_id"], Counter())[
                category_names[annotation["category_id"]]
            ] += 1
        for image in split_coco["images"]:
            source_id = source_id_by_page[image["page_id"]]
            target = split_dir / image["file_name"]
            shutil.copyfile(sources[source_id], target)
            if file_digest(target) != image["sha256"]:
                raise RuntimeError("Copied image checksum mismatch")
            manifest_rows.append({
                "schema": SCHEMA,
                "page_id": image["page_id"],
                "collection": _collection(image["page_id"]),
                "split": split,
                "image": f"{split}/{image['file_name']}",
                "sha256": image["sha256"],
                "width": image["width"],
                "height": image["height"],
                "objects": annotations_by_image[image["id"]],
                "classes": dict(sorted(classes_by_image.get(image["id"], {}).items())),
            })
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(split_coco, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n")

    train_summary = _summary(split_cocos["train"])
    valid_summary = _summary(split_cocos["valid"])
    train_collections = set(train_summary["collections"])
    valid_collections = set(valid_summary["collections"])
    if train_collections & valid_collections:
        raise RuntimeError("Collection leakage detected")
    train_hashes = {item["sha256"] for item in split_cocos["train"]["images"]}
    valid_hashes = {item["sha256"] for item in split_cocos["valid"]["images"]}
    if train_hashes & valid_hashes:
        raise RuntimeError("Image hash leakage detected")

    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n"
                for row in sorted(manifest_rows, key=lambda row: row["page_id"])),
        encoding="utf-8", newline="\n")
    provenance = {
        "schema": SCHEMA,
        "release_status": "private-training-candidate-not-published",
        "source": {
            "candidate_coco_sha256": file_digest(candidate / "annotations.coco.json"),
            "candidate_adjudication_sha256": file_digest(candidate / "adjudication.json"),
            "review_input_sha256": adjudication["review_input_sha256"],
            "source_evidence_sha256": adjudication["source_evidence_sha256"],
            "source_code_revision": adjudication.get("source_code_revision"),
        },
        "split": split_info,
        "train": train_summary,
        "valid": valid_summary,
        "collection_overlap": sorted(train_collections & valid_collections),
        "image_hash_overlap": sorted(train_hashes & valid_hashes),
        "intended_use": "RF-DETR layout development only; valid is internal development, not final test.",
        "limitations": [
            "Only 60 pages from 22 collections are included.",
            "Labels combine teacher consensus with one human review pass and are not definitive ground truth.",
            "Rare classes may be absent from one split and do not support reliable per-class conclusions.",
            "No near-duplicate visual audit beyond exact image SHA-256 was performed.",
            "The final PolOCRBench test remains excluded and must stay closed during model selection.",
        ],
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    checksum_paths = sorted(
        (path for path in output.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(output).as_posix())
    (output / "checksums.sha256").write_text(
        "".join(
            f"{file_digest(path)}  {path.relative_to(output).as_posix()}\n"
            for path in checksum_paths
        ), encoding="utf-8", newline="\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--valid-pages", type=int, default=12)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    args = parser.parse_args()
    print(json.dumps(build_dataset(
        args.candidate, args.images, args.output, args.valid_pages, args.salt
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
