"""CPU-only line-geometry variants over frozen printed-replay evidence.

Compares unchanged native segmentation with deskew geometry handling while
reusing the exact text, anchor and confidence gates of the frozen miner.
Nothing produced here is eligible for training or evaluation.
"""
import argparse
from collections import Counter
import csv
import importlib.metadata
import io
import json
import math
from pathlib import Path
import shutil
import subprocess

from PIL import Image

from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.mine_printed_replay import exact_anchors, parse_tsv


SCHEMA = "slayer-printed-replay-geometry-v1"
PAD = (2, 1, 2, 1)  # left, top, right, bottom — identical to the frozen miner
MAX_SKEW_DEGREES = 5.0


def rotate_point(x, y, cx, cy, degrees):
    """Map a native point into the frame of ``Image.rotate(degrees, center=...)``."""
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    dx, dy = x - cx, y - cy
    return cx + dx * cosine + dy * sine, cy - dx * sine + dy * cosine


def affine(degrees, cx, cy):
    """2x3 native->frame matrix matching :func:`rotate_point`."""
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    return [cosine, sine, cx - cx * cosine - cy * sine,
            -sine, cosine, cy + cx * sine - cy * cosine]


def apply_affine(matrix, x, y):
    return matrix[0] * x + matrix[1] * y + matrix[2], matrix[3] * x + matrix[4] * y + matrix[5]


def transform_bbox(bbox, cx, cy, degrees):
    left, top, right, bottom = bbox
    points = [rotate_point(x, y, cx, cy, degrees)
              for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]
    return [math.floor(min(p[0] for p in points)), math.floor(min(p[1] for p in points)),
            math.ceil(max(p[0] for p in points)), math.ceil(max(p[1] for p in points))]


def parse_words(text, width, height):
    """Group the word TSV into lines like :func:`parse_tsv`, keeping word boxes."""
    groups = {}
    for row in csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row["level"] != "5" or not row["text"].strip():
            continue
        key = tuple(int(row[k]) for k in ("page_num", "block_num", "par_num", "line_num"))
        left, top, w, h = [int(row[k]) for k in ("left", "top", "width", "height")]
        if min(left, top) < 0 or min(w, h) <= 0 or left + w > width or top + h > height:
            raise ValueError("Invalid Tesseract word geometry")
        confidence = float(row["conf"])
        if not 0 <= confidence <= 100:
            raise ValueError("Invalid word confidence")
        groups.setdefault(key, []).append({"text": row["text"], "bbox": [left, top, left + w, top + h],
                                          "confidence": confidence})
    result = []
    for key, words in groups.items():
        result.append({"key": key, "text": " ".join(w["text"] for w in words),
                       "bbox": [min(w["bbox"][0] for w in words), min(w["bbox"][1] for w in words),
                                max(w["bbox"][2] for w in words), max(w["bbox"][3] for w in words)],
                       "min_word_confidence": min(w["confidence"] for w in words), "words": words})
    return result


def strip_words(lines):
    return [{k: v for k, v in line.items() if k != "words"} for line in lines]


def line_slope_degrees(words):
    """Least-squares slope of word centers; positive means descending in image coords."""
    if len(words) < 2:
        return 0.0
    points = [((w["bbox"][0] + w["bbox"][2]) / 2, (w["bbox"][1] + w["bbox"][3]) / 2) for w in words]
    mean_x = sum(p[0] for p in points) / len(points)
    mean_y = sum(p[1] for p in points) / len(points)
    denominator = sum((p[0] - mean_x) ** 2 for p in points)
    if denominator == 0:
        return 0.0
    slope = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points) / denominator
    return math.degrees(math.atan(slope))


def page_skew_degrees(lines):
    slopes = sorted(line_slope_degrees(line["words"]) for line in lines if len(line["words"]) >= 2)
    if not slopes:
        return 0.0
    middle = len(slopes) // 2
    skew = slopes[middle] if len(slopes) % 2 else (slopes[middle - 1] + slopes[middle]) / 2
    return skew if abs(skew) <= MAX_SKEW_DEGREES else 0.0


def frame_lines(lines, cx, cy, degrees):
    """Line records with word boxes mapped into the rotated comparison frame.

    The line extent stays the union of its word boxes, so rotation cannot inflate
    a long axis-aligned rectangle into a larger one.
    """
    framed = []
    for index, line in enumerate(lines):
        words = [{**word, "bbox": transform_bbox(word["bbox"], cx, cy, degrees)} for word in line["words"]]
        bbox = [min(w["bbox"][0] for w in words), min(w["bbox"][1] for w in words),
                max(w["bbox"][2] for w in words), max(w["bbox"][3] for w in words)]
        framed.append({**line, "line_index": index, "bbox": bbox, "words": words})
    return framed


def overlap_report(rows):
    """Extent conflicts of adjacent anchors, computed outside the all-or-nothing gate."""
    conflicts = []
    for left, right in zip(rows, rows[1:]):
        order = left["source_span"][1] > right["source_span"][0]
        overlap = max(0, left["bbox"][3] - right["bbox"][1])
        if order or overlap:
            conflicts.append({"line_indices": [left["line_index"], right["line_index"]],
                              "source_order_conflict": order, "vertical_extent_overlap_px": overlap})
    return conflicts


def crop_geometry(bbox, width, height):
    left, top, right, bottom = bbox
    return [max(0, left - PAD[0]), max(0, top - PAD[1]),
            min(width, right + PAD[2]), min(height, bottom + PAD[3])]


def neighbor_intrusion(crop_box, line, lines):
    """Rectangular overlap of the crop with every other line's word boxes."""
    left, top, right, bottom = crop_box
    inside = {}
    for other in lines:
        if other["line_index"] == line["line_index"]:
            continue
        rects = []
        for word in other["words"]:
            x0, y0, x1, y1 = word["bbox"]
            rect = [max(left, x0), max(top, y0), min(right, x1), min(bottom, y1)]
            if rect[2] > rect[0] and rect[3] > rect[1]:
                rects.append(rect)
        if rects:
            inside[other["line_index"]] = {"px2": sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects),
                                           "rects": rects}
    return inside


def run_frame(native, lines, reference, *, degrees, page_min_coverage=0.0):
    """Run the frozen miner gates in one comparison frame; pixels stay untouched."""
    width, height = native.size
    cx, cy = width / 2, height / 2
    framed = frame_lines(lines, cx, cy, degrees)
    matched, rejected, coverage = exact_anchors(framed, reference, width, height,
                                                page_min_coverage=page_min_coverage)
    gated = sorted((r for r in rejected if r["reason"] == "anchor-order-or-line-overlap"),
                   key=lambda row: row["line_index"])
    return {"rotation_degrees": degrees, "affine_native_to_frame": affine(degrees, cx, cy),
            "center": [cx, cy], "matched": matched, "rejected": rejected,
            "anchor_coverage": coverage, "conflicts": overlap_report(gated),
            "gated_by_overlap": bool(gated),
            "line_boxes": [[line["line_index"], line["bbox"]] for line in framed],
            "framed_lines": framed}


def render_pair(frame_image, line, framed_line, degrees, width, height, *, image_dir, text_dir, identifier):
    """Save a crop from the comparison frame and record its native coordinate mapping."""
    cx, cy = width / 2, height / 2
    crop_box = crop_geometry(framed_line["bbox"], width, height)
    crop = frame_image.crop(crop_box)
    image_path = image_dir / (identifier + ".png")
    text_path = text_dir / (identifier + ".txt")
    crop.save(image_path)
    text_path.write_text(line["text"], encoding="utf-8")
    corners_frame = [(crop_box[0], crop_box[1]), (crop_box[2], crop_box[1]),
                     (crop_box[2], crop_box[3]), (crop_box[0], crop_box[3])]
    corners = [rotate_point(x, y, cx, cy, -degrees) for x, y in corners_frame]
    matrix = affine(degrees, cx, cy)
    roundtrip = max(math.dist(apply_affine(matrix, *rotate_point(x, y, cx, cy, -degrees)), (x, y))
                    for x, y in corners_frame)
    return {"image": image_path, "text_file": text_path, "crop_bbox": crop_box,
            "native_corners": [[round(x, 4), round(y, 4)] for x, y in corners],
            "affine_native_to_crop": matrix, "roundtrip_error_px": roundtrip}


def page_variants(native, lines, reference, page_id, source, *, output, produce_crops):
    width, height = native.size
    skew = page_skew_degrees(lines)
    variants = {}
    for name, degrees in (("native", 0.0), ("page-deskew", skew)):
        result = run_frame(native, lines, reference, degrees=degrees)
        cx, cy = width / 2, height / 2
        frame_image = native if degrees == 0.0 else native.rotate(
            degrees, center=(cx, cy), resample=Image.Resampling.BICUBIC)
        framed = {line["line_index"]: line for line in result["framed_lines"]}
        records = []
        if produce_crops:
            image_dir = output / "crops" / name / page_id
            text_dir = output / "texts" / name / page_id
            image_dir.mkdir(parents=True, exist_ok=True)
            text_dir.mkdir(parents=True, exist_ok=True)
            for line in result["matched"]:
                identifier = f"{page_id}-line-{line['line_index']:03d}"
                rendered = render_pair(frame_image, line, framed[line["line_index"]], degrees,
                                       width, height, image_dir=image_dir, text_dir=text_dir,
                                       identifier=identifier)
                inside = neighbor_intrusion(rendered["crop_bbox"], framed[line["line_index"]],
                                            result["framed_lines"])
                records.append({"id": identifier, "page_id": page_id,
                                "work_family": source["work_family"], "split": source["split"],
                                "line_index": line["line_index"], "frame": name,
                                "text": line["text"], "min_word_confidence": line["min_word_confidence"],
                                "source_span_normalized": line["source_span"],
                                "native_bbox": lines[line["line_index"]]["bbox"],
                                "frame_bbox": framed[line["line_index"]]["bbox"],
                                "rotation_degrees": degrees,
                                "affine_native_to_crop": rendered["affine_native_to_crop"],
                                "native_corners": rendered["native_corners"],
                                "roundtrip_error_px": rendered["roundtrip_error_px"],
                                "crop_bbox": rendered["crop_bbox"],
                                "image": rendered["image"].relative_to(output).as_posix(),
                                "text_file": rendered["text_file"].relative_to(output).as_posix(),
                                "image_sha256": digest(rendered["image"]),
                                "text_sha256": digest(rendered["text_file"]),
                                "intruding_neighbor_lines": sorted(inside),
                                "intrusion_rects": [{"line_index": key, "rect": rect}
                                                    for key, value in sorted(inside.items())
                                                    for rect in value["rects"]],
                                "neighbor_word_box_overlap_px2": sum(v["px2"] for v in inside.values()),
                                "crop_review_status": "not-human-reviewed",
                                "label_status": "source-validated-text-automatic-exact-line-anchor",
                                "eligible_for_training": False, "eligible_for_evaluation": False})
        variants[name] = {"rotation_degrees": degrees,
                          "rejected_reasons": dict(Counter(r["reason"] for r in result["rejected"])),
                          "accepted_anchors": len(result["matched"]),
                          "anchor_coverage": result["anchor_coverage"],
                          "conflicts": result["conflicts"], "gated_by_overlap": result["gated_by_overlap"],
                          "line_boxes": result["line_boxes"],
                          "pairs": records}
    return {"page_id": page_id, "split": source["split"], "work_family": source["work_family"],
            "page_skew_degrees": skew, "native_lines": len(lines),
            "pairs_with_intrusion": {name: sum(1 for r in v["pairs"] if r["intruding_neighbor_lines"])
                                     for name, v in variants.items()},
            "variants": variants}


def crop_sheet(output, page_id, name, pairs, limit=8):
    """Montage of crops with intruding neighbor rectangles drawn inside."""
    from PIL import ImageDraw, ImageFont
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    rows = pairs[:limit]
    if not rows:
        return None
    sheet = Image.new("RGB", (1500, 150 * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        y = index * 150
        draw.text((12, y + 4), f"{row['id']} | conf {row['min_word_confidence']:.0f} | "
                               f"intrusion px2 {row['neighbor_word_box_overlap_px2']}", font=font, fill="black")
        with Image.open(output / row["image"]) as image:
            crop = image.copy()
        painter = ImageDraw.Draw(crop)
        for rect in row["intrusion_rects"]:
            painter.rectangle([rect["rect"][0] - row["crop_bbox"][0], rect["rect"][1] - row["crop_bbox"][1],
                               rect["rect"][2] - row["crop_bbox"][0] - 1, rect["rect"][3] - row["crop_bbox"][1] - 1],
                              outline="red", width=1)
        scale = min(3, 1476 / crop.width, 115 / crop.height)
        sheet.paste(crop.resize((round(crop.width * scale), round(crop.height * scale))), (12, y + 28))
    path = output / f"crop-sheet-{name}-{page_id}.png"
    sheet.save(path)
    return path.name


def geometry_sheet(output, page_id, variants, native):
    """Native vs deskew line boxes over page pixels, around the first native conflict."""
    from PIL import ImageDraw, ImageFont
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    conflict = variants["native"]["conflicts"][0] if variants["native"]["conflicts"] else None
    boxes = dict(variants["native"]["line_boxes"])
    if conflict is None:
        return None
    indices = conflict["line_indices"]
    focus = [boxes[i] for i in indices]
    bounds = [max(0, min(b[0] for b in focus) - 10), max(0, min(b[1] for b in focus) - 40),
              min(native.width, max(b[2] for b in focus) + 10), min(native.height, max(b[3] for b in focus) + 40)]
    sheet = Image.new("RGB", (1500, 2 * 240 + 40), "white")
    draw = ImageDraw.Draw(sheet)
    cx, cy = native.width / 2, native.height / 2
    for panel, name in enumerate(("native", "page-deskew")):
        y = panel * 240
        degrees = variants[name]["rotation_degrees"]
        draw.text((12, y + 4), f"{page_id} | frame {name} ({degrees:+.3f} deg) | native-conflict lines {indices}",
                  font=font, fill="black")
        draw.text((12, y + 24), "conflicts: " + str(len(variants[name]["conflicts"])) +
                  " | accepted anchors: " + str(variants[name]["accepted_anchors"]), font=font, fill="black")
        frame = native if degrees == 0.0 else native.rotate(degrees, center=(cx, cy),
                                                            resample=Image.Resampling.BICUBIC)
        image = frame.crop(bounds)
        painter = ImageDraw.Draw(image)
        for line_index, box in variants[name]["line_boxes"]:
            color = "red" if line_index in indices else "blue"
            painter.rectangle([box[0] - bounds[0], box[1] - bounds[1],
                               box[2] - bounds[0] - 1, box[3] - bounds[1] - 1], outline=color, width=2)
        scale = min(2, 1476 / image.width, 180 / image.height)
        sheet.paste(image.resize((round(image.width * scale), round(image.height * scale))), (12, y + 46))
    path = output / f"geometry-sheet-{page_id}.png"
    sheet.save(path)
    return path.name


def summarize(pages, name, native_name="native"):
    conflict_pages = [p for p in pages if p["role"] == "conflict-page"]
    accepted = sum(len(p["variants"][name]["pairs"]) for p in conflict_pages)
    baseline = sum(len(p["variants"][native_name]["pairs"]) for p in conflict_pages)
    return {"accepted_anchors_conflict_pages": accepted,
            "rejected_exact_anchors_conflict_pages": sum(
                p["variants"][name]["rejected_reasons"].get("anchor-order-or-line-overlap", 0)
                for p in conflict_pages),
            "extent_conflicts_conflict_pages": sum(len(p["variants"][name]["conflicts"])
                                                   for p in conflict_pages),
            "source_order_conflicts": sum(c["source_order_conflict"] for p in conflict_pages
                                          for c in p["variants"][name]["conflicts"]),
            "additional_accepted_over_native": accepted - baseline,
            "pairs_with_neighbor_intrusion": sum(1 for p in pages for r in p["variants"][name]["pairs"]
                                                 if r["intruding_neighbor_lines"]),
            "max_roundtrip_error_px": max((r["roundtrip_error_px"] for p in pages
                                           for r in p["variants"][name]["pairs"]), default=0.0)}


def run(audit_root, output, *, control_pages=3):
    audit_root, output = Path(audit_root), Path(output)
    input_evidence = audit_root / "evidence"
    sources = read_rows(input_evidence / "source-manifest.jsonl")
    diagnostic = json.loads((audit_root / "geometry-diagnostic.json").read_text(encoding="utf-8"))
    affected = {page["page_id"] for page in diagnostic["pages"]}
    control = sorted({source["id"] for source in sources} - affected)[:control_pages]
    selected = sorted(affected | set(control))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    evidence = output / "evidence"
    evidence.mkdir()
    pages, pairs = [], []
    for source in sorted(sources, key=lambda row: row["id"]):
        if source["id"] not in selected:
            continue
        directory = input_evidence / "native-pages" / source["id"]
        with Image.open(directory / "page.png") as image:
            image.load()
            native = image.convert("RGB")
        width, height = native.size
        tsv = (directory / "tesseract.tsv").read_text(encoding="utf-8")
        lines = parse_words(tsv, width, height)
        if strip_words(lines) != parse_tsv(tsv, width, height):
            raise ValueError("Word-level parsing diverges from the frozen miner")
        reference = (input_evidence / "source-package" / source["reference_file"]).read_text(encoding="utf-8")
        result = page_variants(native, lines, reference, source["id"], source,
                               output=evidence, produce_crops=True)
        result["role"] = "conflict-page" if source["id"] in affected else "control-page"
        result["native_page_sha256"] = digest(directory / "page.png")
        result["tesseract_tsv_sha256"] = digest(directory / "tesseract.tsv")
        result["sheets"] = [name for name in (
            geometry_sheet(evidence, source["id"], result["variants"], native)
            if result["role"] == "conflict-page" else None,
            *(crop_sheet(evidence, source["id"], variant, result["variants"][variant]["pairs"])
              for variant in ("native", "page-deskew"))) if name]
        pages.append(result)
        for variant in result["variants"].values():
            pairs.extend(variant["pairs"])
        print("GEOMETRY_READY", source["id"], result["role"],
              {name: (v["accepted_anchors"], len(v["conflicts"])) for name, v in result["variants"].items()},
              flush=True)
    write_rows(evidence / "pairs.jsonl", pairs)
    write_json(evidence / "report.json", {
        "schema": SCHEMA, "audit_root": str(audit_root),
        "input_geometry_diagnostic_sha256": digest(audit_root / "geometry-diagnostic.json"),
        "input_audit_sha256": digest(audit_root / "audit.json"),
        "input_evidence_checksums_sha256": digest(input_evidence / "checksums.json"),
        "selected_pages": selected, "conflict_pages": sorted(affected), "control_pages": control,
        "page_min_coverage": 0.0, "padding": list(PAD), "max_skew_degrees": MAX_SKEW_DEGREES,
        "variants": {name: summarize(pages, name) for name in ("native", "page-deskew")},
        "pages": pages, "pairs_written": len(pairs), "promoted_pairs": 0,
        "eligible_for_training": False, "eligible_for_evaluation": False,
        "baseline_changed": False,
        "decision": "Geometry handling is measured, not promoted; the frozen miner, gates and baseline are unchanged.",
        "limitations": [
            "Intrusion is rectangular word-box overlap, not ink collision.",
            "Deskew uses a single median page angle and cannot model curved or warped lines.",
            "Crops are not human reviewed and no transcription label was changed.",
            "Exact-anchor selection still favors easy lines.",
        ]})
    write_json(evidence / "environment.json", {
        "python_pillow": importlib.metadata.version("pillow"),
        "module_sha256": digest(__file__),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "native_mapping": "rotate_point/affine checked against PIL.Image.rotate in tests"})
    write_json(evidence / "checksums.json", {p.relative_to(evidence).as_posix(): digest(p)
                                             for p in sorted(evidence.rglob("*"))
                                             if p.is_file() and p.name != "checksums.json"})
    shutil.copyfile(evidence / "report.json", output / "report.json")
    archive = Path(shutil.make_archive(str(output / "printed-replay-geometry-v1-evidence"), "zip", evidence))
    print("EVIDENCE_ZIP", archive, flush=True)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--control-pages", type=int, default=3)
    args = parser.parse_args()
    run(args.audit_root, args.output, control_pages=args.control_pages)
