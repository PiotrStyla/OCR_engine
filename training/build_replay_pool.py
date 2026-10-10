"""Assemble the printed-replay candidate pool (CPU-only, candidates only).

Cut rule approved 2026-10-10 (user, conversation): crops are cut at the ink
band of the line, not at the padded word-box rectangle. This module renders one
crop per anchor for the eight conflict pages:

- 43 anchors accepted by the geometry V2 gate (frozen text/anchor/confidence
  gates, exact shared-axis geometry);
- 56 anchors behind the 6 ink-separated box overlaps, approved for ink-gap
  cuts (see reference-review/ink-check evidence).

Every crop is machine-verified to contain zero neighboring ink pixels, upright
(regularized to the page median axis), and carries its provenance. Nothing is
eligible for training: the pool is a candidate set pending the remaining human
gates (rare PUA spots, U+FFFD, errata, second-reviewer sign-off).
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.mine_printed_replay import parse_tsv
from training.printed_replay_geometry import (affine, apply_affine, frame_lines, line_slope_degrees,
                                              parse_words, rotate_point, strip_words)
from training.printed_replay_geometry_v2 import run_page
from training.printed_replay_ink_check import ink_mask, line_ink


SCHEMA = "slayer-replay-pool-candidate-v1"
CONTENT_FLAGS = {"may-nad-rio-de-la-plata-0557": "publisher-backmatter",
                 "may-nad-rio-de-la-plata-0563": "publisher-backmatter"}


def ink_band(gray, word_boxes):
    """Vertical ink extent of one line inside its word boxes (or None if blank)."""
    ink = line_ink(gray, word_boxes)
    rows = np.nonzero(ink.any(axis=1))[0]
    if rows.size == 0:
        return None
    return int(rows[0]), int(rows[-1])


def cut_crop(gray, word_boxes, margin=1):
    """Crop box at the line's ink band with a small margin; blank lines use boxes."""
    xs = [box[0] for box in word_boxes] + [box[2] for box in word_boxes]
    ys = [box[1] for box in word_boxes] + [box[3] for box in word_boxes]
    left, right = min(xs), max(xs)
    region = [min(xs), min(ys), max(xs), max(ys)]
    band = ink_band(ink_mask(gray, region), word_boxes)
    if band is None:
        band = (min(ys), max(ys) - 1)
    return [left, max(0, band[0] - margin), right, min(gray.shape[0], band[1] + 1 + margin)]


def render_pool_crop(frame_image, gray, crop_box, neighbor_boxes):
    """Render the crop and verify zero neighboring ink inside it."""
    left, top, right, bottom = crop_box
    crop = frame_image.crop(crop_box)
    threshold = float((np.median(gray) + int(gray.min())) / 2.0)
    inside = 0
    for x0, y0, x1, y1 in neighbor_boxes:
        cx0, cy0 = max(left, x0), max(top, y0)
        cx1, cy1 = min(right, x1), min(bottom, y1)
        if cx1 > cx0 and cy1 > cy0:
            inside += int(np.count_nonzero(gray[cy0:cy1, cx0:cx1] <= threshold))
    return crop, inside


def run(audit_root, geometry_root, output):
    audit_root, geometry_root, output = Path(audit_root), Path(geometry_root), Path(output)
    source_evidence = audit_root / "evidence"
    sources = read_rows(source_evidence / "source-manifest.jsonl")
    diagnostic = json.loads((audit_root / "geometry-diagnostic.json").read_text(encoding="utf-8"))
    affected = {page["page_id"] for page in diagnostic["pages"]}
    if output.exists():
        raise FileExistsError(output)
    (output / "crops").mkdir(parents=True)
    (output / "texts").mkdir(parents=True)
    manifest, report_pages = [], []
    for source in sorted(sources, key=lambda row: row["id"]):
        if source["id"] not in affected:
            continue
        page_id = source["id"]
        directory = source_evidence / "native-pages" / page_id
        with Image.open(directory / "page.png") as image:
            image.load()
            native = image.convert("RGB")
        width, height = native.size
        tsv = (directory / "tesseract.tsv").read_text(encoding="utf-8")
        lines = parse_words(tsv, width, height)
        if strip_words(lines) != parse_tsv(tsv, width, height):
            raise ValueError("Word-level parsing diverges from the frozen miner")
        reference = (source_evidence / "source-package" / source["reference_file"]).read_text(encoding="utf-8")
        outcome = run_page(native, lines, reference)
        rule = "geometry-v2-accepted" if not outcome["gated_by_overlap"] else "ink-gap-approved"
        anchors = [(row, rule) for row in outcome["candidates"]]
        page_ink = np.asarray(native.convert("L"), dtype=np.uint8)
        theta = outcome["axis_degrees"]
        frame_image = native.rotate(theta, center=(width / 2, height / 2),
                                    resample=Image.Resampling.BICUBIC)
        frame_gray = np.asarray(frame_image.convert("L"), dtype=np.uint8)
        framed_lines = frame_lines(lines, width / 2, height / 2, theta)
        boxes = {index: [word["bbox"] for word in line["words"]] for index, line in enumerate(framed_lines)}
        verified = 0
        for row, rule in anchors:
            index = row["line_index"]
            word_boxes = boxes[index]
            crop_box = cut_crop(frame_gray, word_boxes)
            neighbors = [box for other, box_list in boxes.items() if other != index for box in box_list]
            crop, inside = render_pool_crop(frame_image, frame_gray, crop_box, neighbors)
            identifier = f"{page_id}-line-{index:03d}"
            crop.save(output / "crops" / (identifier + ".png"))
            (output / "texts" / (identifier + ".txt")).write_text(row["text"], encoding="utf-8")
            corners = [rotate_point(x, y, width / 2, height / 2, -theta) for x, y in
                       ((crop_box[0], crop_box[1]), (crop_box[2], crop_box[1]),
                        (crop_box[2], crop_box[3]), (crop_box[0], crop_box[3]))]
            manifest.append({"id": identifier, "page_id": page_id,
                             "work_family": source["work_family"], "split": source["split"],
                             "line_index": index, "cut_rule": rule,
                             "content_type": CONTENT_FLAGS.get(page_id, "unclassified-pending-review"),
                             "text": row["text"], "min_word_confidence": row["min_word_confidence"],
                             "source_span_normalized": row["source_span"],
                             "axis_degrees": theta, "crop_bbox": crop_box,
                             "affine_native_to_crop": affine(theta, width / 2, height / 2),
                             "native_corners": [[round(x, 4), round(y, 4)] for x, y in corners],
                             "neighbor_ink_pixels_in_crop": inside,
                             "ink_verified_clean": inside == 0,
                             "eligible_for_training": False, "eligible_for_evaluation": False,
                             "review_status": "candidate-pending-human-sign-off"})
            verified += inside == 0
        report_pages.append({"page_id": page_id, "anchors": len(anchors),
                             "rule": rule,
                             "geometry_v2_accepted": len(outcome["accepted"]),
                             "ink_verified_clean": verified,
                             "content_type": CONTENT_FLAGS.get(page_id, "unclassified-pending-review")})
        print("POOL", page_id, len(anchors), "kotwic | czyste atramentowo:", verified, flush=True)
    write_rows(output / "manifest.jsonl", manifest)
    write_json(output / "report.json", {
        "schema": SCHEMA,
        "cut_rule": "line ink band + 1 px margin (approved 2026-10-10)",
        "input_geometry_sha256": digest(geometry_root / "evidence" / "report.json"),
        "totals": {"anchors": len(manifest),
                   "geometry_v2_accepted": sum(1 for r in manifest if r["cut_rule"] == "geometry-v2-accepted"),
                   "ink_gap_approved": sum(1 for r in manifest if r["cut_rule"] == "ink-gap-approved"),
                   "ink_verified_clean": sum(1 for r in manifest if r["ink_verified_clean"]),
                   "content_types": dict(Counter(r["content_type"] for r in manifest))},
        "pages": report_pages,
        "eligible_for_training": False, "eligible_for_evaluation": False,
        "decision": "Candidate pool assembled; promotion needs the remaining human gates.",
        "open_gates": ["second-reviewer sign-off on the reference candidate",
                       "rare PUA spots (5 codepoints, 23 places)", "U+FFFD (86 spots)",
                       "errata page transcription", "content-type classification for non-backmatter pages"],
        "limitations": ["Ink verification is threshold-based, not a human inspection.",
                        "Content types are flagged only where documented earlier."]})
    write_json(output / "checksums.json", {p.relative_to(output).as_posix(): digest(p)
                                           for p in sorted(output.rglob("*"))
                                           if p.is_file() and p.name != "checksums.json"})
    return output / "report.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.audit_root, args.geometry_root, args.output)
