"""Audit V1's empty replay result and preserve exact rejected anchors, not crops."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile

from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.mine_printed_replay import exact_anchors
from training.prepare_printed_replay_sources import verify_source_package


V1_REVISION = "794f48feba605da4664d09610c735eb84c04e9c7"
SOURCE_CHECKSUMS = "b8125235588355b426eb0ccc56b1b71992aa301e7b3a6afa45d0a20520dcd982"


def verify_archive(archive):
    members = archive.infolist()
    names = [m.filename for m in members]
    if len(names) != len(set(names)) or sum(m.file_size for m in members) > 256_000_000:
        raise ValueError("Duplicate/oversized evidence archive")
    for member in members:
        name = member.orig_filename
        path = PurePosixPath(name)
        if (not name or name != member.filename or path.is_absolute() or ".." in path.parts or
                "\\" in name or ":" in name or
                (member.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError("Unsafe evidence member")
    payloads = {m.filename for m in members if not m.is_dir()} - {"checksums.json"}
    checksums = json.loads(archive.read("checksums.json"))
    if set(checksums) != payloads:
        raise ValueError("Evidence checksum coverage mismatch")
    for name, expected in checksums.items():
        if hashlib.sha256(archive.read(name)).hexdigest() != expected:
            raise ValueError("Evidence payload checksum mismatch: " + name)
    return checksums


def audit(archive_path, output):
    archive_path, output = Path(archive_path), Path(output)
    with zipfile.ZipFile(archive_path) as archive:
        checksums = verify_archive(archive)
        report = json.loads(archive.read("report.json"))
        environment = json.loads(archive.read("environment.json"))
        if (report["schema"] != "slayer-printed-replay-mining-v1" or
                environment["code_revision"] != V1_REVISION or
                environment["source_checksums_sha256"] != SOURCE_CHECKSUMS):
            raise ValueError("Unexpected V1 protocol/source lineage")
        runner = subprocess.check_output(["git", "show", V1_REVISION + ":training/mine_printed_replay.py"])
        if hashlib.sha256(runner).hexdigest() != environment["runner_sha256"]:
            raise ValueError("Pinned runner hash mismatch")
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
    if digest(evidence / "source-package/checksums.json") != SOURCE_CHECKSUMS:
        raise ValueError("Pinned source checksums mismatch")
    for name in ("checksums.json", "manifest.jsonl", "source-policy.json"):
        if (evidence / ("source-" + name)).read_bytes() != (evidence / "source-package" / name).read_bytes():
            raise ValueError("Duplicate source receipt mismatch")
    predictions = read_rows(evidence / "manifest.jsonl")
    quarantine = read_rows(evidence / "quarantine.jsonl")
    pages = {r["page_id"]: r for r in report["pages"]}
    by_source = {r["id"]: r for r in sources}
    if set(pages) != set(by_source) or len(pages) != len(report["pages"]):
        raise ValueError("Page report coverage mismatch")
    if any(r["page_id"] not in pages for r in predictions + quarantine):
        raise ValueError("Unknown evidence page")
    reconstructed, reconsidered, page_results = [], [], []
    for page_id, source in by_source.items():
        page = pages[page_id]
        rows = [r for r in predictions + quarantine if r["page_id"] == page_id]
        indexed = {r.get("line_index", -1): r for r in rows}
        if len(indexed) != len(rows) or set(indexed) != set(range(page["lines"])):
            raise ValueError("Incomplete/duplicate line coverage")
        width, height = page["native_dimensions"]
        if min(width, height) <= 0 or page["split"] != source["split"]:
            raise ValueError("Invalid dimensions/split")
        lines = []
        for index in range(page["lines"]):
            row = indexed[index]
            left, top, right, bottom = row["bbox"]
            confidence = row["min_word_confidence"]
            if not (0 <= left < right <= width and 0 <= top < bottom <= height and
                    math.isfinite(confidence) and 0 <= confidence <= 100):
                raise ValueError("Invalid returned geometry/confidence")
            lines.append({k: row[k] for k in ("key", "text", "bbox", "min_word_confidence")})
        reference = (evidence / "source-package" / source["reference_file"]).read_text(encoding="utf-8")
        accepted, rejected, coverage = exact_anchors(lines, reference, width, height)
        if (len(accepted) != page["candidates"] or not math.isclose(coverage,
                page["anchor_coverage_before_page_gate"], abs_tol=1e-12)):
            raise ValueError("Recomputed V1 page result differs")
        original_rejected = {r["line_index"]: r for r in quarantine if r["page_id"] == page_id}
        for row in rejected:
            actual = original_rejected[row["line_index"]]
            if row["reason"] != actual["reason"] or row.get("source_span") != actual.get("source_span"):
                raise ValueError("V1 exclusion reason/source span mismatch")
        reconstructed.extend({"page_id": page_id, **r} for r in rejected)
        candidates, _, _ = exact_anchors(lines, reference, width, height, page_min_coverage=0)
        reconsidered.extend({"page_id": page_id, "work_family": source["work_family"],
            "split": source["split"], "source_revision": source["source_revision"], **r,
            "status": "exact-anchor-only-crop-not-recovered", "eligible_for_training": False,
            "eligible_for_evaluation": False} for r in candidates)
        page_results.append({"id": page_id, "lines": len(lines), "v1_accepted": len(accepted),
                             "exact_line_candidates_without_page_gate": len(candidates), "coverage": coverage})
    counts = dict(Counter(r["reason"] for r in reconstructed))
    if counts != report["quarantine_reasons"] or report["candidates"] != len(predictions) or predictions:
        raise ValueError("V1 total counts/empty-result mismatch")
    if report["eligible_for_training"] or report["eligible_for_evaluation"] or report["splits"]:
        raise ValueError("Unexpected training/evaluation promotion")
    result = {"schema": "slayer-printed-replay-v1-audit-v1", "archive_sha256": digest(archive_path),
        "payloads_verified": len(checksums), "pages": len(sources), "lines": len(quarantine),
        "v1_accepted": 0, "exclusion_reasons_reproduced": counts,
        "exact_line_candidates_without_page_gate": len(reconsidered),
        "candidate_splits": dict(Counter(r["split"] for r in reconsidered)), "page_results": page_results,
        "native_images_present": False, "training_ready": False, "evaluation_ready": False,
        "decision": "Retain baseline; V1 made no replay corpus. Preserve exact anchors as diagnostic proposals only.",
        "limitations": "No native page pixels, full word TSV, original DjVu or weights returned. Confidence/geometry cannot be independently verified from pixels; no training promotion.",
        "auditor_sha256": digest(__file__)}
    write_rows(output / "exact-anchor-proposals.jsonl", reconsidered)
    write_json(output / "audit.json", result)
    write_json(output / "checksums.json", {p.name: digest(p) for p in output.iterdir() if p.is_file()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.output), indent=2))
