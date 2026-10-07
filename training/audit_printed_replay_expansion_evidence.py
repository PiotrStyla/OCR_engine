"""Audit the frozen eight-family expansion without trusting archive-supplied pins."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from training.audit_printed_replay_v2_evidence import audit, require
from training.full_page_pilot import digest, read_rows, write_json
from PIL import Image, ImageDraw, ImageFont


REVISION = "c8ae68ae741d2c30d95317c2c0bb84d347d79a16"
SOURCE_CHECKSUMS = "53e3c81ec06e7a1c75f0b3a3553d97252716a5ccfe495e510b61d540f030b5ad"
WORKS_PATH = "experiments/2026-10-07/printed-replay-expansion-v1/works.json"
WORKS_SHA256 = "ee01b99977b100d3eb3a309c0e59fb9f3d762017d74b66ccdac048a4789f4b93"


def diagnose_conflicts(quarantine):
    pages = {}
    for row in quarantine:
        if row["reason"] == "anchor-order-or-line-overlap":
            pages.setdefault(row["page_id"], []).append(row)
    results = []
    for page_id, rows in pages.items():
        rows = sorted(rows, key=lambda row: row["line_index"])
        conflicts = []
        for left, right in zip(rows, rows[1:]):
            order = left["source_span"][1] > right["source_span"][0]
            overlap = max(0, left["bbox"][3] - right["bbox"][1])
            if order or overlap:
                conflicts.append({"line_indices": [left["line_index"], right["line_index"]],
                    "source_order_conflict": order, "vertical_extent_overlap_px": overlap,
                    "bboxes": [left["bbox"], right["bbox"]]})
        require(bool(conflicts), "Page-gate rejection without a reproducible conflict")
        results.append({"page_id": page_id, "rejected_exact_anchors": len(rows), "conflicts": conflicts})
    return {"schema": "slayer-printed-replay-geometry-diagnostic-v1", "pages": results,
        "rejected_exact_anchors": sum(page["rejected_exact_anchors"] for page in results),
        "source_order_conflicts": sum(c["source_order_conflict"] for p in results for c in p["conflicts"]),
        "bbox_extent_conflicts": sum(c["vertical_extent_overlap_px"] > 0 for p in results for c in p["conflicts"]),
        "rescued_pairs": 0, "training_ready": False,
        "limitation": "Rectangle extent overlap is not proof of ink collision or an incorrect transcription."}


def conflict_sheets(evidence, diagnostic, output):
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    names = []
    for start in range(0, len(diagnostic["pages"]), 4):
        pages = diagnostic["pages"][start:start + 4]
        sheet = Image.new("RGB", (1500, 280 * len(pages)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, page in enumerate(pages):
            conflict = page["conflicts"][0]
            y = index * 280
            draw.text((12, y + 5), page["page_id"] + " | lines " + str(conflict["line_indices"]), font=font, fill="black")
            draw.text((12, y + 30), "Vertical extent overlap: " + str(conflict["vertical_extent_overlap_px"]) + " px", font=font, fill="black")
            with Image.open(evidence / "native-pages" / page["page_id"] / "page.png") as image:
                boxes = conflict["bboxes"]
                bounds = [max(0, min(b[0] for b in boxes) - 10), max(0, min(b[1] for b in boxes) - 40),
                          min(image.width, max(b[2] for b in boxes) + 10), min(image.height, max(b[3] for b in boxes) + 40)]
                crop = image.crop(bounds)
                painter = ImageDraw.Draw(crop)
                for box, color in zip(boxes, ("red", "blue")):
                    painter.rectangle([box[0] - bounds[0], box[1] - bounds[1], box[2] - bounds[0] - 1,
                                       box[3] - bounds[1] - 1], outline=color, width=2)
                scale = min(2, 1476 / crop.width, 210 / crop.height)
                sheet.paste(crop.resize((round(crop.width * scale), round(crop.height * scale))), (12, y + 60))
        name = f"conflict-context-{len(names) + 1:02d}.png"
        sheet.save(output / name)
        names.append(name)
    return names


def add_geometry_diagnostic(output):
    output = Path(output)
    evidence = output / "evidence"
    result = diagnose_conflicts(read_rows(evidence / "quarantine.jsonl"))
    result["context_sheets"] = conflict_sheets(evidence, result, output)
    result["archive_sha256"] = json.loads((output / "audit.json").read_text(encoding="utf-8"))["archive_sha256"]
    write_json(output / "geometry-diagnostic.json", result)
    write_json(output / "checksums.json", {p.name: digest(p) for p in output.iterdir()
                                           if p.is_file() and p.name != "checksums.json"})
    return result


def expansion_audit(archive_path, output):
    config = subprocess.check_output(["git", "show", REVISION + ":" + WORKS_PATH])
    require(hashlib.sha256(config).hexdigest() == WORKS_SHA256, "Pinned expansion selection mismatch")
    works = json.loads(config)["works"]
    result = audit(archive_path, output, code_revision=REVISION, source_checksums=SOURCE_CHECKSUMS,
                   works=works, sample_review=True, result_schema="slayer-printed-replay-expansion-v1-audit-v1")
    add_geometry_diagnostic(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(expansion_audit(Path(args.archive), Path(args.output)), indent=2))
