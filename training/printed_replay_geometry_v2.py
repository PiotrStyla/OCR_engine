"""Per-line geometry V2 for printed replay anchors (CPU-only, frozen gates).

V1 measured line extents as axis-aligned boxes of rotated word boxes, which
inflates long lines (width * sin(theta)) and adds up to 1 px of outward rounding
per edge — enough to invent conflicts on flat pages and to keep tight-leading
pairs gated. V2 changes only the geometry predicate: line extents are measured
by projecting word-box corners onto the normal of the page's median baseline
angle (exact, no inflation, no rounding). A single shared axis is required:
per-line slope fits from 2-3 words are noisy, and testing pairs in mismatched
frames couples the horizontal axis into the vertical separation and invents
1-3 px conflicts. Per-line angles are used only to render upright crops.

Everything else is the frozen miner's rule, unchanged: the text, anchor and
confidence gates run through ``training.mine_printed_replay.exact_anchors``, the
source-order rule is the same, and the all-or-nothing page gate still applies.
Nothing produced here is eligible for training or evaluation.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess

from PIL import Image

from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.mine_printed_replay import exact_anchors, parse_tsv
from training.printed_replay_geometry import (crop_geometry, frame_lines, line_slope_degrees,
                                              neighbor_intrusion, page_skew_degrees, parse_words,
                                              render_pair, strip_words)


SCHEMA = "slayer-printed-replay-geometry-v2"
GATE_MODE = "all-or-nothing page gate, frozen text/anchor/confidence gates"
PREDICATE = "projected word-box extents on the page median baseline normal"
FLAT_AXIS_EPS_DEGREES = 0.1  # 0.1 deg over a 700 px line = 1.2 px: at or below
# the slope-fit noise floor the axis is indistinguishable from flat, and a
# tilted axis would mix x into the separation of exactly-touching boxes.
PROJECTION_EPS_PX = 1e-6  # float dust only (a nanometre at 300 dpi); touching
# boxes project with ~1e-14 residue and must stay non-conflicting.


def normal_extents(words, degrees):
    """Exact min/max extent of word boxes along the baseline normal of ``degrees``."""
    theta = math.radians(degrees + 90.0)
    nx, ny = math.cos(theta), math.sin(theta)
    projections = []
    for word in words:
        x0, y0, x1, y1 = word["bbox"]
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            projections.append(x * nx + y * ny)
    return min(projections), max(projections)


def frame_overlap(words_a, words_b, degrees):
    """Overlap of two word sets along the normal of ``degrees`` (0 = separated)."""
    a_min, a_max = normal_extents(words_a, degrees)
    b_min, b_max = normal_extents(words_b, degrees)
    overlap = min(a_max, b_max) - max(a_min, b_min)
    return overlap if overlap > PROJECTION_EPS_PX else 0.0


def pair_conflict(left, right, degrees):
    """Frozen semantics: source order first, then slanted extent overlap.

    Both lines are projected onto one shared axis (the page median baseline
    normal): testing in two mismatched per-line frames turns a small slope-fit
    error into a false conflict on long lines.
    """
    order = left["source_span"][1] > right["source_span"][0]
    overlap = frame_overlap(left["words"], right["words"], degrees)
    return (order or overlap > 0), {"line_indices": [left["line_index"], right["line_index"]],
                                    "source_order_conflict": order,
                                    "slanted_overlap_px": round(overlap, 3),
                                    "shared_axis_degrees": round(degrees, 4)}


def gate(anchors, degrees):
    """Frozen all-or-nothing gate with the V2 predicate; anchors in line order."""
    conflicts = [pair_conflict(a, b, degrees) for a, b in zip(anchors, anchors[1:])]
    details = [detail for conflict, detail in conflicts if conflict]
    return (not details), details


def run_page(native, lines, reference):
    """Native and V2 outcomes for one page; text gates run once, frozen."""
    text_matched, text_rejected, coverage = exact_anchors(lines, reference, native.width,
                                                          native.height, page_min_coverage=0)
    frozen_gate = [row for row in text_rejected if row["reason"] == "anchor-order-or-line-overlap"]
    anchors = sorted(text_matched + frozen_gate, key=lambda row: row["line_index"])
    thetas = {index: line_slope_degrees(line["words"]) for index, line in enumerate(lines)}
    page_theta = page_skew_degrees(lines)
    axis = 0.0 if abs(page_theta) < FLAT_AXIS_EPS_DEGREES else page_theta
    admitted, details = gate(anchors, axis)
    accepted = anchors if admitted else []
    rejected = [row for row in text_rejected if row["reason"] != "anchor-order-or-line-overlap"]
    if not admitted:
        rejected.extend({**row, "reason": "anchor-order-or-line-overlap"} for row in anchors)
    return {"text_gate_matched": len(text_matched), "anchor_coverage": coverage,
            "geometry_candidates": len(anchors), "accepted_anchors": len(accepted),
            "rejected_reasons": dict(Counter(row["reason"] for row in rejected)),
            "conflicts": details, "gated_by_overlap": not admitted,
            "accepted": accepted, "thetas": thetas,
            "page_theta": page_theta, "axis_degrees": axis,
            "axis_snapped_to_flat": axis != page_theta}


def render_accepted(native, lines, accepted, thetas, output, page_id):
    """Upright per-line crops with coordinate mapping and intrusion measurement."""
    width, height = native.size
    image_dir = output / "crops" / page_id
    text_dir = output / "texts" / page_id
    image_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in accepted:
        index = row["line_index"]
        theta = thetas[index]
        framed = frame_lines(lines, width / 2, height / 2, theta)
        frame_image = native.rotate(theta, center=(width / 2, height / 2),
                                    resample=Image.Resampling.BICUBIC)
        identifier = f"{page_id}-line-{index:03d}"
        rendered = render_pair(frame_image, row, framed[index], theta, width, height,
                               image_dir=image_dir, text_dir=text_dir, identifier=identifier)
        inside = neighbor_intrusion(rendered["crop_bbox"], framed[index], framed)
        records.append({"id": identifier, "page_id": page_id, "line_index": index,
                        "text": row["text"], "rotation_degrees": theta,
                        "min_word_confidence": row["min_word_confidence"],
                        "source_span_normalized": row["source_span"],
                        "crop_bbox": rendered["crop_bbox"],
                        "affine_native_to_crop": rendered["affine_native_to_crop"],
                        "native_corners": rendered["native_corners"],
                        "roundtrip_error_px": rendered["roundtrip_error_px"],
                        "intruding_neighbor_lines": sorted(inside),
                        "neighbor_word_box_overlap_px2": sum(v["px2"] for v in inside.values()),
                        "crop_review_status": "not-human-reviewed",
                        "eligible_for_training": False, "eligible_for_evaluation": False})
    return records


def run(audit_root, output):
    audit_root, output = Path(audit_root), Path(output)
    source_evidence = audit_root / "evidence"
    sources = read_rows(source_evidence / "source-manifest.jsonl")
    diagnostic = json.loads((audit_root / "geometry-diagnostic.json").read_text(encoding="utf-8"))
    affected = {page["page_id"] for page in diagnostic["pages"]}
    control = sorted({source["id"] for source in sources} - affected)[:3]
    selected = sorted(affected | set(control))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    evidence = output / "evidence"
    (evidence / "crops").mkdir(parents=True)
    (evidence / "texts").mkdir(parents=True)
    pages, pairs = [], []
    for source in sorted(sources, key=lambda row: row["id"]):
        if source["id"] not in selected:
            continue
        directory = source_evidence / "native-pages" / source["id"]
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
        records = render_accepted(native, lines, outcome["accepted"], outcome["thetas"],
                                  evidence, source["id"]) if outcome["accepted"] else []
        pages.append({"page_id": source["id"], "split": source["split"],
                      "work_family": source["work_family"],
                      "role": "conflict-page" if source["id"] in affected else "control-page",
                      "native_lines": len(lines),
                      "text_gate_matched": outcome["text_gate_matched"],
                      "anchor_coverage": outcome["anchor_coverage"],
                      "geometry_candidates": outcome["geometry_candidates"],
                      "accepted_anchors": outcome["accepted_anchors"],
                      "rejected_reasons": outcome["rejected_reasons"],
                      "conflicts": outcome["conflicts"],
                      "gated_by_overlap": outcome["gated_by_overlap"],
                      "axis_degrees": outcome["axis_degrees"],
                      "axis_snapped_to_flat": outcome["axis_snapped_to_flat"],
                      "intruding_crops": sum(1 for r in records if r["intruding_neighbor_lines"])})
        pairs.extend(records)
        print("GEOMETRY_V2", source["id"], outcome["accepted_anchors"], "of",
              outcome["geometry_candidates"], "conflicts", len(outcome["conflicts"]), flush=True)
    write_rows(evidence / "pairs.jsonl", pairs)
    conflict_pages = [p for p in pages if p["role"] == "conflict-page"]
    newspaper = [p for p in pages if p["page_id"] == "zawadzki-kurjer-1904-0004"]
    native_conflicts = sum(len(p["conflicts"]) for p in conflict_pages)
    write_json(evidence / "report.json", {
        "schema": SCHEMA, "gate_mode": GATE_MODE, "predicate": PREDICATE,
        "input_geometry_diagnostic_sha256": digest(audit_root / "geometry-diagnostic.json"),
        "selected_pages": selected,
        "v1_reference": {"page_deskew_accepted": 43, "native_conflicts": 27,
                          "source": "experiments/2026-10-08/printed-replay-geometry-v1"},
        "totals": {"accepted_anchors_conflict_pages": sum(p["accepted_anchors"] for p in conflict_pages),
                   "rejected_exact_anchors_conflict_pages": sum(
                       p["rejected_reasons"].get("anchor-order-or-line-overlap", 0) for p in conflict_pages),
                   "conflicts_conflict_pages": native_conflicts,
                   "source_order_conflicts": sum(c["source_order_conflict"] for p in pages
                                                 for c in p["conflicts"]),
                   "pairs_with_neighbor_intrusion": sum(1 for r in pairs
                                                        if r["intruding_neighbor_lines"]),
                   "max_roundtrip_error_px": max((r["roundtrip_error_px"] for r in pairs), default=0.0)},
        "regression_gates": {
            "flat_newspaper_no_new_conflicts": {
                "page": "zawadzki-kurjer-1904-0004",
                "native_conflicts": 1,
                "v2_conflicts": len(newspaper[0]["conflicts"]) if newspaper else None,
                "passed": (len(newspaper[0]["conflicts"]) <= 1) if newspaper else None},
            "backmatter_recovery": {
                "pages": ["may-nad-rio-de-la-plata-0557", "may-nad-rio-de-la-plata-0563"],
                "accepted": {p["page_id"]: p["accepted_anchors"] for p in pages
                             if p["page_id"].startswith("may-nad-rio-de-la-plata-055")
                             or p["page_id"].startswith("may-nad-rio-de-la-plata-056")}}},
        "pages": pages, "pairs_written": len(pairs), "promoted_pairs": 0,
        "eligible_for_training": False, "eligible_for_evaluation": False,
        "baseline_changed": False,
        "decision": "Geometry handling is measured, not promoted; frozen gates and baseline unchanged.",
        "limitations": [
            "Intrusion is rectangular word-box overlap, not ink collision.",
            "Per-line angles are least-squares fits of word centers; curved lines are not modeled.",
            "Crops are not human reviewed and no transcription label was changed.",
        ]})
    write_json(evidence / "environment.json", {
        "python_pillow": importlib.metadata.version("pillow"),
        "module_sha256": digest(__file__),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    write_json(evidence / "checksums.json", {p.relative_to(evidence).as_posix(): digest(p)
                                             for p in sorted(evidence.rglob("*"))
                                             if p.is_file() and p.name != "checksums.json"})
    shutil.copyfile(evidence / "report.json", output / "report.json")
    archive = Path(shutil.make_archive(str(output / "printed-replay-geometry-v2-evidence"), "zip",
                                       evidence))
    print("EVIDENCE_ZIP", archive, flush=True)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.audit_root, args.output)
