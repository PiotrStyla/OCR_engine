"""Recompute V2 anchors from word TSV and verify crops against returned native pixels."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import zipfile

from PIL import Image, ImageDraw, ImageFont

from training.audit_printed_replay_evidence import SOURCE_CHECKSUMS, verify_archive
from training.full_page_pilot import digest, read_rows, write_json
from training.mine_printed_replay import exact_anchors, parse_tsv
from training.prepare_printed_replay_sources import verify_source_package


V2_REVISION = "8b88ee5c8dd6cdbb429dbf6980077eaf35f32676"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_page(evidence, source, page, predictions, quarantine):
    page_id = source["id"]
    directory = evidence / "native-pages" / page_id
    with Image.open(directory / "page.png") as image:
        image.load()
        require(image.mode == "RGB", "Native page is not RGB")
        native = image.copy()
    width, height = native.size
    require(page["native_dimensions"] == [width, height] and page["split"] == source["split"],
            "Native dimensions/source split mismatch")
    lines = parse_tsv((directory / "tesseract.tsv").read_text(encoding="utf-8"), width, height)
    reference = (evidence / "source-package" / source["reference_file"]).read_text(encoding="utf-8")
    matched, rejected, coverage = exact_anchors(lines, reference, width, height, page_min_coverage=0)
    require(len(lines) == page["lines"] and len(matched) == page["candidates"] and
            math.isclose(coverage, page["anchor_coverage_before_page_gate"], rel_tol=0, abs_tol=1e-12),
            "Recomputed page counts/coverage mismatch")
    actual = {row["line_index"]: row for row in predictions}
    require(len(actual) == len(predictions) and set(actual) == {row["line_index"] for row in matched},
            "Candidate line coverage mismatch")
    excluded = {row["line_index"]: row for row in quarantine}
    require(len(excluded) == len(quarantine) and set(excluded) == {row["line_index"] for row in rejected},
            "Quarantined line coverage mismatch")
    # Compare complete rejected records, not just aggregate exclusion counts.
    for row in rejected:
        expected = json.loads(json.dumps({"page_id": page_id, **row}))
        require(excluded[row["line_index"]] == expected, "Quarantine differs from native TSV")
    native_hash = digest(directory / "page.png")
    for line in matched:
        row = actual[line["line_index"]]
        identifier = f"{page_id}-line-{line['line_index']:03d}"
        left, top, right, bottom = line["bbox"]
        crop_box = [max(0, left - 2), max(0, top - 1), min(width, right + 2), min(height, bottom + 1)]
        expected_fields = {
            "id": identifier, "page_id": page_id, "line_index": line["line_index"],
            "tesseract_key": list(line["key"]), "work_family": source["work_family"],
            "split": source["split"], "image": f"pairs/{source['split']}/{identifier}.png",
            "text_file": f"pairs/{source['split']}/{identifier}.txt", "text": line["text"],
            "bbox": line["bbox"], "crop_bbox": crop_box, "native_page_dimensions": [width, height],
            "native_page_sha256": native_hash, "source_span_normalized": line["source_span"],
            "min_word_confidence": line["min_word_confidence"],
            "label_status": "source-validated-text-automatic-exact-line-anchor",
            "crop_review_status": "not-human-reviewed", "eligible_for_training": False,
            "eligible_for_evaluation": False,
        }
        for key in ("reference_sha256", "source_revision", "source_url", "original_scan_sha1",
                    "scan_license", "transcription_license"):
            expected_fields[key] = source[key]
        require(all(row.get(key) == value for key, value in expected_fields.items()),
                "Candidate metadata/anchor differs from native TSV/source")
        scan_hash = row.get("original_scan_sha256", "")
        require(len(scan_hash) == 64 and all(c in "0123456789abcdef" for c in scan_hash),
                "Invalid original scan SHA256 receipt")
        image_path, text_path = evidence / row["image"], evidence / row["text_file"]
        require(digest(image_path) == row["image_sha256"] and digest(text_path) == row["text_sha256"],
                "Candidate file checksum mismatch")
        require(text_path.read_bytes() == line["text"].encode("utf-8"), "Candidate text bytes mismatch")
        with Image.open(image_path) as crop:
            expected_crop = native.crop(crop_box)
            require(crop.mode == native.mode and crop.size == expected_crop.size and
                    crop.tobytes() == expected_crop.tobytes(), "Crop pixels differ from native page")
    return {"page_id": page_id, "split": source["split"], "lines": len(lines),
            "candidates": len(matched), "coverage": coverage, "native_dimensions": [width, height]}


def contact_sheets(evidence, predictions, output, *, context=False):
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    paths = []
    for start in range(0, len(predictions), 9):
        rows = predictions[start:start + 9]
        sheet = Image.new("RGB", (1500, len(rows) * 190), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows):
            y = index * 190
            draw.text((12, y + 5), row["id"] + " | " + row["split"], fill="black", font=font)
            draw.text((12, y + 30), row["text"], fill="black", font=font)
            image_path = evidence / row["image"]
            if context:
                image_path = evidence / "native-pages" / row["page_id"] / "page.png"
            with Image.open(image_path) as image:
                crop = image.copy()
                if context:
                    left, top, right, bottom = row["crop_bbox"]
                    margin = bottom - top
                    context_box = [max(0, left - 12), max(0, top - margin),
                                   min(image.width, right + 12), min(image.height, bottom + margin)]
                    crop = image.crop(context_box)
                    ImageDraw.Draw(crop).rectangle(
                        [left - context_box[0], top - context_box[1],
                         right - context_box[0] - 1, bottom - context_box[1] - 1], outline="red")
                scale = min(3, 1476 / crop.width, 125 / crop.height)
                preview = crop.resize((round(crop.width * scale), round(crop.height * scale)))
                sheet.paste(preview, (12, y + 58))
            draw.line((0, y + 189, 1500, y + 189), fill="#cccccc")
        prefix = "context-sheet" if context else "contact-sheet"
        path = output / f"{prefix}-{len(paths) + 1:02d}.png"
        sheet.save(path)
        paths.append(path.name)
    return paths


def audit(archive_path, output):
    archive_path, output = Path(archive_path), Path(output)
    with zipfile.ZipFile(archive_path) as archive:
        checksums = verify_archive(archive)
        report = json.loads(archive.read("report.json"))
        environment = json.loads(archive.read("environment.json"))
        require(report["schema"] == "slayer-printed-replay-mining-v2" and
                report["page_min_coverage"] == 0 and environment["code_revision"] == V2_REVISION and
                environment["source_checksums_sha256"] == SOURCE_CHECKSUMS,
                "Unexpected V2 protocol/source lineage")
        runner = subprocess.check_output(["git", "show", V2_REVISION + ":training/mine_printed_replay.py"])
        require(hashlib.sha256(runner).hexdigest() == environment["runner_sha256"], "Pinned runner mismatch")
        output.mkdir(parents=True, exist_ok=False)
        evidence = output / "evidence"
        evidence.mkdir()
        for member in archive.infolist():
            target = evidence / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    shutil.copyfile(archive_path, output / archive_path.name)
    sources = verify_source_package(evidence / "source-package")
    require(digest(evidence / "source-package/checksums.json") == SOURCE_CHECKSUMS, "Pinned source mismatch")
    for name in ("checksums.json", "manifest.jsonl", "source-policy.json"):
        require((evidence / ("source-" + name)).read_bytes() ==
                (evidence / "source-package" / name).read_bytes(), "Duplicate source receipt mismatch")
    predictions, quarantine = read_rows(evidence / "manifest.jsonl"), read_rows(evidence / "quarantine.jsonl")
    pages = {r["page_id"]: r for r in report["pages"]}
    require(len(pages) == len(report["pages"]) and set(pages) == {r["id"] for r in sources},
            "Page report coverage mismatch")
    require(all(r["page_id"] in pages for r in predictions + quarantine), "Unknown page")
    page_results = [verify_page(evidence, source, pages[source["id"]],
                   [r for r in predictions if r["page_id"] == source["id"]],
                   [r for r in quarantine if r["page_id"] == source["id"]]) for source in sources]
    require(len({r["id"] for r in predictions}) == len(predictions), "Duplicate candidate ID")
    require(len({r["image_sha256"] for r in predictions}) == len(predictions), "Duplicate candidate image")
    expected_pairs = {r[key] for r in predictions for key in ("image", "text_file")}
    require({p.relative_to(evidence).as_posix() for p in (evidence / "pairs").rglob("*") if p.is_file()}
            == expected_pairs, "Pair file coverage mismatch")
    expected_native = {f"native-pages/{s['id']}/{name}" for s in sources
                       for name in ("page.png", "tesseract.tsv", "tesseract.stderr.txt")}
    require({p.relative_to(evidence).as_posix() for p in (evidence / "native-pages").rglob("*") if p.is_file()}
            == expected_native, "Native evidence coverage mismatch")
    scan_receipts = {}
    for row in predictions:
        old = scan_receipts.setdefault(row["work_family"], row["original_scan_sha256"])
        require(old == row["original_scan_sha256"], "Inconsistent original scan receipt")
    counts = dict(Counter(r["reason"] for r in quarantine))
    splits = dict(Counter(r["split"] for r in predictions))
    require(counts == report["quarantine_reasons"] and splits == report["splits"] and
            len(predictions) == report["candidates"], "Aggregate result mismatch")
    require(report["eligible_for_training"] is False and report["eligible_for_evaluation"] is False,
            "Unexpected training/evaluation promotion")
    sheets = contact_sheets(evidence, predictions, output)
    context = contact_sheets(evidence, predictions, output, context=True)
    result = {"schema": "slayer-printed-replay-v2-audit-v1", "archive_sha256": digest(archive_path),
        "payloads_verified": len(checksums), "code_revision": V2_REVISION,
        "source_checksums_sha256": SOURCE_CHECKSUMS, "pages": len(sources),
        "lines_recomputed_from_word_tsv": sum(r["lines"] for r in page_results),
        "candidate_pairs_verified": len(predictions), "candidate_splits": splits,
        "pixel_exact_native_crops": len(predictions), "exclusion_reasons_reproduced": counts,
        "page_results": page_results, "original_scan_sha256_receipts": scan_receipts,
        "contact_sheets": sheets, "context_sheets": context,
        "visual_review_status": "pending-agent-inspection",
        "training_ready": False, "evaluation_ready": False, "baseline_changed": False,
        "decision": "Preserve 25 candidate lines and one probe; scale independent replay sources before training.",
        "limitations": ["Exact-anchor selection favors easy lines.",
            "Original DjVu decoding and Tesseract execution were not independently rerun locally.",
            "Returned native pixels/TSV verify derivation, not independent ground truth.",
            "Two work families and one probe line are not a representative benchmark.",
            "Upstream pagequality=4 is not project human review; model pretraining exposure unknown."],
        "auditor_sha256": digest(__file__)}
    write_json(output / "audit.json", result)
    write_json(output / "checksums.json", {p.name: digest(p) for p in output.iterdir() if p.is_file()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.output), indent=2))
