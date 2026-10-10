"""Ink-level collision check for printed replay crops (CPU-only, diagnostic).

The geometry gate measures word-box rectangles; rectangles overlap by 2-3 px
on the six residual pairs while the printed ink of the two lines may still be
physically separated. This module answers the question a reviewer needs before
any crop decision: does neighbor ink actually enter the crop?

Two measurements per case, both on the frozen native page pixels:

- ``ink_min_gap_px`` for a pair: the minimum vertical distance between the ink
  bands of the two lines, column by column over the x-range where both lines
  have ink. Positive = ink separated (croppable with care), non-positive = ink
  interleaves (genuinely unsafe).
- ``neighbor_ink_pixels`` for an accepted crop: how many pixels of neighboring
  lines' ink fall inside the crop rectangle.

Ink is thresholded per case inside the union region of the two lines, so scan
shading does not decide the outcome. Measurement only: no crop is promoted,
rejected or relabeled.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.printed_replay_geometry import parse_words


SCHEMA = "slayer-printed-replay-ink-check-v1"


def ink_mask(gray, region_box):
    """Adaptive ink threshold inside the case region (median vs darkest)."""
    left, top, right, bottom = [int(v) for v in region_box]
    region = gray[top:bottom, left:right]
    if region.size == 0:
        raise ValueError("Empty ink region")
    threshold = (float(np.median(region)) + float(region.min())) / 2.0
    return gray <= threshold


def line_ink(mask, boxes):
    """Ink pixels of one line: thresholded ink inside its word boxes."""
    canvas = np.zeros_like(mask)
    for x0, y0, x1, y1 in boxes:
        canvas[y0:y1, x0:x1] = True
    return canvas & mask


def ink_gap(mask, boxes_a, boxes_b):
    """Whitespace rows between the two lines' ink, column by column.

    Positive = ink separated by that many rows; negative = that many rows of
    ink overlap. Only columns where both lines have ink are measured.
    """
    ink_a = line_ink(mask, boxes_a)
    ink_b = line_ink(mask, boxes_b)
    shared = np.nonzero(ink_a.any(axis=0) & ink_b.any(axis=0))[0]
    if shared.size == 0:
        return {"ink_min_gap_px": None, "columns_checked": 0, "columns_in_contact": 0,
                "verdict": "no-shared-ink-columns"}
    gaps = []
    for x in shared:
        col_a = np.nonzero(ink_a[:, x])[0]
        col_b = np.nonzero(ink_b[:, x])[0]
        gaps.append(int(col_b[0]) - int(col_a[-1]) - 1)
    minimum = min(gaps)
    contact = sum(1 for gap in gaps if gap <= 0)
    return {"ink_min_gap_px": minimum, "columns_checked": len(gaps),
            "columns_in_contact": contact,
            "verdict": "ink-separated" if minimum > 0 else "ink-contact"}


def crop_intrusion(gray, crop_box, neighbor_boxes, threshold):
    """Count neighbor ink pixels inside the crop rectangle."""
    left, top, right, bottom = crop_box
    inside = 0
    for x0, y0, x1, y1 in neighbor_boxes:
        cx0, cy0 = max(left, x0), max(top, y0)
        cx1, cy1 = min(right, x1), min(bottom, y1)
        if cx1 > cx0 and cy1 > cy0:
            block = gray[cy0:cy1, cx0:cx1]
            inside += int(np.count_nonzero(block <= threshold))
    return inside


def run(ink_root, geometry_root, output):
    ink_root, geometry_root, output = Path(ink_root), Path(geometry_root), Path(output)
    evidence = ink_root / "evidence"
    report = json.loads((geometry_root / "evidence" / "report.json").read_text(encoding="utf-8"))
    pairs = read_rows(geometry_root / "evidence" / "pairs.jsonl")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cases, crop_cases = [], []
    for page in report["pages"]:
        directory = evidence / "native-pages" / page["page_id"]
        with Image.open(directory / "page.png") as image:
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
        height, width = gray.shape
        lines = parse_words((directory / "tesseract.tsv").read_text(encoding="utf-8"), width, height)
        boxes = {index: [word["bbox"] for word in line["words"]] for index, line in enumerate(lines)}
        for conflict in page["conflicts"]:
            index_a, index_b = conflict["line_indices"]
            union = [min(box[0] for box in boxes[index_a] + boxes[index_b]),
                     min(box[1] for box in boxes[index_a] + boxes[index_b]),
                     max(box[2] for box in boxes[index_a] + boxes[index_b]),
                     max(box[3] for box in boxes[index_a] + boxes[index_b])]
            region = [max(0, union[0]), max(0, union[1]), min(width, union[2]), min(height, union[3])]
            mask = ink_mask(gray, region)
            result = ink_gap(mask, boxes[index_a], boxes[index_b])
            cases.append({"page_id": page["page_id"], "line_indices": conflict["line_indices"],
                          "box_overlap_px": conflict["slanted_overlap_px"],
                          "ink_region_threshold": float((np.median(gray[region[1]:region[3],
                                                                    region[0]:region[2]])
                                                         + gray[region[1]:region[3],
                                                               region[0]:region[2]].min()) / 2.0),
                          **result})
            print("INK_PAIR", page["page_id"], conflict["line_indices"], result["verdict"],
                  result["ink_min_gap_px"], flush=True)
        page_pairs = [row for row in pairs if row["page_id"] == page["page_id"]]
        for row in page_pairs:
            if not row["intruding_neighbor_lines"]:
                continue
            threshold = float((np.median(gray) + int(gray.min())) / 2.0)
            neighbor_boxes = [box for index in row["intruding_neighbor_lines"] for box in boxes[index]]
            inside = crop_intrusion(gray, row["crop_bbox"], neighbor_boxes, threshold)
            crop_cases.append({"id": row["id"], "page_id": page["page_id"],
                               "neighbor_lines": row["intruding_neighbor_lines"],
                               "neighbor_ink_pixels_in_crop": inside,
                               "neighbor_box_overlap_px2": row["neighbor_word_box_overlap_px2"],
                               "verdict": "ink-inside-crop" if inside else "boxes-only"})
    write_rows(output / "ink-pairs.jsonl", cases)
    write_rows(output / "ink-crops.jsonl", crop_cases)
    write_json(output / "report.json", {
        "schema": SCHEMA, "measures": ["ink_min_gap_px (pair, shared columns)",
                                        "neighbor_ink_pixels_in_crop (crop)"],
        "geometry_report_sha256": digest(geometry_root / "evidence" / "report.json"),
        "pairs_checked": len(cases), "crops_checked": len(crop_cases),
        "verdicts": {"ink-separated": sum(1 for c in cases if c["verdict"] == "ink-separated"),
                      "ink-contact": sum(1 for c in cases if c["verdict"] == "ink-contact"),
                      "other": sum(1 for c in cases if c["verdict"] not in ("ink-separated", "ink-contact")),
                      "ink-inside-crop": sum(1 for c in crop_cases if c["verdict"] == "ink-inside-crop"),
                      "boxes-only": sum(1 for c in crop_cases if c["verdict"] == "boxes-only")},
        "cases": cases, "crop_cases": crop_cases,
        "promoted_pairs": 0, "eligible_for_training": False,
        "decision": "Ink evidence for human crop review; nothing promoted.",
        "limitations": ["Ink threshold is adaptive per case and measured, not calibrated.",
                        "Per-column band gaps approximate the distance along the line normal.",
                        "No transcription label was changed and no crop was accepted or rejected here."]})
    write_json(output / "checksums.json", {p.relative_to(output).as_posix(): digest(p)
                                           for p in sorted(output.rglob("*"))
                                           if p.is_file() and p.name != "checksums.json"})
    return output / "report.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ink-root", required=True, help="Reproduced expansion audit root (native pages)")
    parser.add_argument("--geometry-root", required=True, help="Geometry V2 output directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.ink_root, args.geometry_root, args.output)
