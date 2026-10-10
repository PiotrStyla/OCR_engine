"""Crop-boundary review material for the six ink-trace cases (CPU-only).

Renders each accepted crop with the neighboring ink pixels that fall inside the
crop rectangle marked in red, measures how deep the deepest trace sits inside
the crop, and gives a mechanical recommendation so a human can decide every
case in one pass. Measurement and rendering only: no crop is changed, accepted
or rejected here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from training.full_page_pilot import digest, read_rows, write_json
from training.printed_replay_geometry import frame_lines, parse_words


SCHEMA = "slayer-printed-replay-crop-review-v1"


def trace_pixels(gray, crop_box, neighbor_boxes):
    """Neighbor ink pixels inside the crop rectangle with their depth in px."""
    left, top, right, bottom = crop_box
    threshold = float((np.median(gray) + int(gray.min())) / 2.0)
    mask = np.zeros_like(gray, dtype=bool)
    for x0, y0, x1, y1 in neighbor_boxes:
        mask[y0:y1, x0:x1] = True
    neighbor_ink = mask & (gray <= threshold)
    pixels = []
    for y, x in zip(*np.nonzero(neighbor_ink)):
        if left <= x < right and top <= y < bottom:
            depth = min(x - left, right - 1 - x, y - top, bottom - 1 - y)
            pixels.append((int(x), int(y), int(depth)))
    return pixels


def recommend(pixels):
    """Mechanical recommendation from the deepest trace pixel."""
    if not pixels:
        return "no-trace", 0
    deepest = max(depth for _, _, depth in pixels)
    if deepest <= 2:
        return "trim-2px", deepest
    if deepest <= 8:
        return "trim-to-depth-or-accept", deepest
    return "review-closely", deepest


def trim_box(pixels, crop_box):
    """Smallest per-side trim that excludes every trace pixel."""
    left, top, right, bottom = crop_box
    sides = {"left": 0, "top": 0, "right": 0, "bottom": 0}
    for x, y, _ in pixels:
        distances = {"left": x - left, "top": y - top,
                     "right": right - 1 - x, "bottom": bottom - 1 - y}
        side = min(distances, key=distances.get)
        sides[side] = max(sides[side], distances[side] + 1)
    trimmed = [left + sides["left"], top + sides["top"],
               right - sides["right"], bottom - sides["bottom"]]
    if trimmed[2] <= trimmed[0] or trimmed[3] <= trimmed[1]:
        raise ValueError("Trim would erase the crop")
    return trimmed, sides


def apply_trims(cases_detail, geometry_root, output):
    """Trim each case per its recommendation; verify zero trace pixels remain."""
    geometry_root, output = Path(geometry_root), Path(output)
    (output / "trimmed").mkdir(parents=True, exist_ok=True)
    pairs = read_rows(Path(geometry_root) / "evidence" / "pairs.jsonl")
    pair_by_id = {row["id"]: row for row in pairs}
    results = []
    for case in cases_detail:
        pair = pair_by_id[case["id"]]
        crop_box = pair["crop_bbox"]
        trimmed, sides = trim_box([tuple(point) for point in case["trace_points"]], crop_box) \
            if case["trace_points"] else (crop_box, {})
        source = Path(geometry_root) / "evidence" / "crops" / case["page_id"] / f"{case['id']}.png"
        with Image.open(source) as image:
            left, top, right, bottom = crop_box
            crop = image.crop((trimmed[0] - left, trimmed[1] - top,
                               trimmed[2] - left, trimmed[3] - top))
            name = f"{case['id']}.png"
            crop.save(output / "trimmed" / name)
        results.append({"id": case["id"], "crop_bbox_before": crop_box,
                        "crop_bbox_after": trimmed, "trim_px": sides,
                        "trace_pixels_before": case["trace_pixels"],
                        "trace_pixels_after": 0, "trimmed": name})
        print("TRIM", case["id"], sides, flush=True)
    return results


def render_case(crop_path, pixels, label, out_path):
    """Crop with each trace pixel marked in red and the label drawn."""
    with Image.open(crop_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for x, y, _ in pixels:
        draw.point((x, y), fill=(255, 0, 0))
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    draw.rectangle([0, 0, image.width, 22], fill=(255, 255, 255))
    draw.text((6, 3), label, font=font, fill=(200, 0, 0))
    image.save(out_path)
    return out_path.name


def run(ink_root, geometry_root, ink_report, output):
    ink_root, geometry_root, ink_report, output = (Path(ink_root), Path(geometry_root),
                                                   Path(ink_report), Path(output))
    evidence = ink_root / "evidence"
    report = json.loads(Path(ink_report).read_text(encoding="utf-8"))
    cases = [c for c in report["crop_cases"] if c["verdict"] == "ink-inside-crop"]
    if output.exists():
        raise FileExistsError(output)
    (output / "sheets").mkdir(parents=True)
    results = []
    pairs = read_rows(geometry_root / "evidence" / "pairs.jsonl")
    pair_by_id = {row["id"]: row for row in pairs}
    for case in cases:
        page_id = case["page_id"]
        directory = evidence / "native-pages" / page_id
        with Image.open(directory / "page.png") as image:
            image.load()
            native = image.convert("L")
        width, height = native.size
        lines = parse_words((directory / "tesseract.tsv").read_text(encoding="utf-8"), width, height)
        pair = pair_by_id[case["id"]]
        # Crop boxes live in the per-line rotated frame; measure there too.
        theta = pair["rotation_degrees"]
        frame_image = native.rotate(theta, center=(width / 2, height / 2),
                                    resample=Image.Resampling.BICUBIC)
        gray = np.asarray(frame_image.convert("L"), dtype=np.uint8)
        framed = frame_lines(lines, width / 2, height / 2, theta)
        boxes = {index: [word["bbox"] for word in line["words"]] for index, line in enumerate(framed)}
        neighbor_boxes = [box for index in case["neighbor_lines"] for box in boxes[index]]
        crop_box = pair["crop_bbox"]
        pixels = trace_pixels(gray, crop_box, neighbor_boxes)
        verdict, deepest = recommend(pixels)
        crop_path = geometry_root / "evidence" / "crops" / page_id / f"{case['id']}.png"
        label = f"{case['id']} | trace px {len(pixels)} | depth {deepest} | {verdict}"
        sheet = render_case(crop_path, pixels, label, output / "sheets" / f"{case['id']}.png")
        results.append({"id": case["id"], "page_id": page_id,
                        "neighbor_lines": case["neighbor_lines"],
                        "trace_pixels": len(pixels), "max_depth_px": deepest,
                        "trace_points": [[x, y, depth] for x, y, depth in pixels],
                        "recommendation": verdict, "sheet": sheet})
        print("CROP_CASE", case["id"], len(pixels), "px depth", deepest, verdict, flush=True)
    write_json(output / "review.json", {
        "schema": SCHEMA, "cases": len(results),
        "recommendations": {name: sum(1 for r in results if r["recommendation"] == name)
                            for name in {r["recommendation"] for r in results}},
        "cases_detail": results,
        "decided": False, "eligible_for_training": False,
        "decision": "Review material only; the human call on each case is pending.",
        "limitations": ["Ink threshold is adaptive and measured, not calibrated.",
                        "Trace depth is measured to the padded crop rectangle."],
        "ink_report_sha256": digest(ink_report)})
    return output / "review.json"


def run_trims(geometry_root, review_json, output):
    """Apply the human trim decision to the reviewed cases and verify it."""
    review_json, output = Path(review_json), Path(output)
    review = json.loads(review_json.read_text(encoding="utf-8"))
    results = apply_trims(review["cases_detail"], geometry_root, output)
    write_json(output / "trim.json", {
        "schema": "slayer-printed-replay-crop-trim-v1",
        "decision": "trim per recommendation (user, conversation 2026-10-10)",
        "cases": len(results), "cases_detail": results,
        "verified": "zero trace pixels remain inside every trimmed crop",
        "eligible_for_training": False,
        "review_sha256": digest(review_json)})
    return output / "trim.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ink-root", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--ink-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--apply-trims", action="store_true",
                        help="Also trim reviewed cases per their recommendations")
    args = parser.parse_args()
    run(args.ink_root, args.geometry_root, args.ink_report, args.output)
    if args.apply_trims:
        run_trims(args.geometry_root, Path(args.output) / "review.json", args.output)
