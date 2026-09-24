"""Build a line-level historical-print corpus from pinned IMPACT region data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import unicodedata
from urllib.parse import quote
from urllib.request import urlopen

from PIL import Image

from training.body_line_geometry import detect_lines


DATASET = "PiotrSty/impact-psnc-polish-ocr"
REVISION = "c7cb156fb95d2880699c33725bbaf1fbc1008fea"
METADATA_SHA256 = {
    "train": "dae1190a6de1f80468cda2e3682b0f0bf491683bf950e71365d9f86a09eba3da",
    "validation": "7c327af0b655813cd2b2e695aa9c9a6f31a2a2524e826e4b2cacec1717db53ad",
    "test": "b311a3c077deb15f95cb7f972fd351be15b5922b69ba8f49a2b3c0834a523435",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_line(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def private_use_count(text: str) -> int:
    return sum(unicodedata.category(character) == "Co" for character in text)


def safe_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe source path: {value}")
    return path


def _read_url(url: str, opener=urlopen) -> bytes:
    with opener(url, timeout=120) as response:
        return response.read()


def load_metadata(opener=urlopen) -> tuple[dict[str, list[dict]], dict[str, str]]:
    rows_by_split = {}
    observed = {}
    root = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/regions"
    for split, expected_hash in METADATA_SHA256.items():
        data = _read_url(f"{root}/{split}/metadata.jsonl", opener)
        observed[split] = digest(data)
        if observed[split] != expected_hash:
            raise ValueError(f"Metadata checksum mismatch: {split}")
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
        if not rows or any(row.get("split") != split for row in rows):
            raise ValueError(f"Invalid metadata split: {split}")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate region ID: {split}")
        rows_by_split[split] = rows
    return rows_by_split, observed


def load_holdout_collections(path: str | Path) -> tuple[set[str], str]:
    data = Path(path).read_bytes()
    manifest = json.loads(data)
    if manifest.get("dataset") != DATASET or manifest.get("revision") != REVISION:
        raise ValueError("Holdout provenance mismatch")
    collections = {row["collection"] for row in manifest["regions"]}
    if len(collections) != 12:
        raise ValueError("Expected 12 holdout collections")
    return collections, digest(data)


def _inventory(texts: list[str]) -> list[dict]:
    counts = Counter(character for text in texts for character in text if ord(character) > 127)
    return [
        {
            "character": character,
            "codepoint": f"U+{ord(character):04X}",
            "name": unicodedata.name(character, "UNNAMED"),
            "count": count,
        }
        for character, count in sorted(counts.items(), key=lambda item: ord(item[0]))
    ]


def build(
    rows_by_split: dict[str, list[dict]],
    holdout_collections: set[str],
    output: str | Path,
    *,
    opener=urlopen,
    segmenter=detect_lines,
    limits: dict[str, int] | None = None,
    provenance: dict | None = None,
) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    required = {"train", "validation", "test"}
    if set(rows_by_split) != required:
        raise ValueError(f"Expected metadata splits: {sorted(required)}")

    train_rows = [row for row in rows_by_split["train"]
                  if row["collection"] not in holdout_collections]
    validation_rows = list(rows_by_split["validation"])
    split_collections = {
        "train": {row["collection"] for row in train_rows},
        "validation": {row["collection"] for row in validation_rows},
        "test": {row["collection"] for row in rows_by_split["test"]},
        "geometry_holdout": set(holdout_collections),
    }
    labels = list(split_collections)
    for index, left in enumerate(labels):
        for right in labels[index + 1:]:
            if split_collections[left] & split_collections[right]:
                raise ValueError(f"Collection overlap: {left} / {right}")

    output.mkdir(parents=True)
    accepted = []
    quarantine = []
    accepted_texts = []
    root = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/regions"
    stats = {}
    source_hashes = {"train": set(), "validation": set()}
    line_hashes = {"train": set(), "validation": set()}

    for split, candidates in (("train", train_rows), ("validation", validation_rows)):
        if limits and limits.get(split) is not None:
            candidates = candidates[:limits[split]]
        pair_dir = output / split
        pair_dir.mkdir()
        counters = Counter(candidate_regions=len(candidates))
        accepted_regions = set()
        for position, row in enumerate(candidates, 1):
            source_path = safe_path(row["file_name"])
            if source_path.parts[0] != "images":
                raise ValueError(f"Unexpected image path: {source_path}")
            url = f"{root}/{split}/{quote(source_path.as_posix(), safe='/')}"
            image_data = _read_url(url, opener)
            if digest(image_data) != row["image_sha256"]:
                raise ValueError(f"Source checksum mismatch: {row['id']}")
            source_hashes[split].add(row["image_sha256"])
            with Image.open(io.BytesIO(image_data)) as source:
                source = source.convert("RGB")
                geometry = segmenter(source, follow_lines=False)
                boxes = geometry["boxes"]
                references = [normalize_line(line) for line in row["text"].splitlines()
                              if normalize_line(line)]
                if len(boxes) != len(references):
                    counters["regions_line_count_mismatch"] += 1
                    quarantine.append({
                        "id": row["id"],
                        "split": split,
                        "collection": row["collection"],
                        "reason": "detected/reference line count mismatch",
                        "detected_lines": len(boxes),
                        "reference_lines": len(references),
                    })
                    continue
                for line_index, (box, text) in enumerate(zip(boxes, references)):
                    line_id = f"{row['id']}__line{line_index:03d}"
                    pua = private_use_count(text)
                    if "\ufffd" in text or pua:
                        reason = "replacement character" if "\ufffd" in text else "private-use character"
                        counters[f"quarantined_{reason.replace('-', '_').replace(' ', '_')}_lines"] += 1
                        quarantine.append({
                            "id": line_id,
                            "split": split,
                            "collection": row["collection"],
                            "source_region_id": row["id"],
                            "reason": reason,
                            "private_use_characters": pua,
                        })
                        continue
                    target = pair_dir / line_id
                    crop = source.crop(tuple(box))
                    crop.save(target.with_suffix(".png"), format="PNG")
                    target.with_suffix(".txt").write_text(text + "\n", encoding="utf-8", newline="\n")
                    image_hash = digest(target.with_suffix(".png").read_bytes())
                    text_hash = digest((text + "\n").encode("utf-8"))
                    if image_hash in line_hashes[split]:
                        raise ValueError(f"Duplicate line image in {split}: {line_id}")
                    line_hashes[split].add(image_hash)
                    accepted.append({
                        "id": line_id,
                        "split": split,
                        "collection": row["collection"],
                        "page_id": row["page_id"],
                        "source_region_id": row["id"],
                        "source_image_sha256": row["image_sha256"],
                        "image_sha256": image_hash,
                        "text_sha256": text_hash,
                        "license": row["license"],
                    })
                    accepted_texts.append(text)
                    accepted_regions.add(row["id"])
                    counters["accepted_lines"] += 1
            if position % 25 == 0 or position == len(candidates):
                print(f"{split}: {position}/{len(candidates)} regions", flush=True)
        counters["accepted_regions"] = len(accepted_regions)
        stats[split] = dict(counters)

    if source_hashes["train"] & source_hashes["validation"]:
        raise ValueError("Exact source-image overlap between train and validation")
    if line_hashes["train"] & line_hashes["validation"]:
        raise ValueError("Exact line-image overlap between train and validation")
    if any(stats[split].get("accepted_lines", 0) == 0 for split in ("train", "validation")):
        raise ValueError("Both train and validation require accepted lines")

    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted),
        encoding="utf-8",
        newline="\n",
    )
    quarantine_path = output / "quarantine.jsonl"
    quarantine_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in quarantine),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema": "ocr-engine-historical-lines-v1",
        "scope": "training/development corpus; not benchmark or gold release",
        "dataset": DATASET,
        "revision": REVISION,
        "geometry": "automatic rectangle; reference line count used only to accept or quarantine a region",
        "normalization": "NFC and whitespace only; historical spelling retained",
        "label_policy": "U+FFFD and private-use lines quarantined; no modernization",
        "collections": {key: sorted(value) for key, value in split_collections.items()},
        "stats": stats,
        "quarantine_records": len(quarantine),
        "unicode_inventory": _inventory(accepted_texts),
        "manifest_sha256": digest(manifest_path.read_bytes()),
        "quarantine_sha256": digest(quarantine_path.read_bytes()),
        "provenance": provenance or {},
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def prepare(output: str | Path, holdout_manifest: str | Path, opener=urlopen, limits=None) -> dict:
    rows_by_split, metadata_hashes = load_metadata(opener)
    holdout_collections, holdout_hash = load_holdout_collections(holdout_manifest)
    return build(
        rows_by_split,
        holdout_collections,
        output,
        opener=opener,
        limits=limits,
        provenance={
            "metadata_sha256": metadata_hashes,
            "holdout_manifest_sha256": holdout_hash,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--holdout-manifest",
        default="experiments/2026-09-24/geometry-holdout-v1/manifest.json",
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    args = parser.parse_args()
    limits = {"train": args.train_limit, "validation": args.validation_limit}
    print(json.dumps(prepare(args.output, args.holdout_manifest, limits=limits),
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
