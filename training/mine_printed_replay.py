"""CPU-only, exact-anchor line candidates from original Wikisource scans."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import subprocess
import unicodedata

from training.full_page_pilot import digest, write_json, write_rows
from training.prepare_printed_replay_sources import download, verify_source_package


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def parse_tsv(text, width, height):
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
        groups.setdefault(key, []).append((row["text"], left, top, left + w, top + h, confidence))
    result = []
    for key, words in groups.items():
        result.append({"key": key, "text": " ".join(w[0] for w in words),
                       "bbox": [min(w[1] for w in words), min(w[2] for w in words),
                                max(w[3] for w in words), max(w[4] for w in words)],
                       "min_word_confidence": min(w[5] for w in words)})
    return result


def exact_anchors(lines, reference, width, height, *, page_min_coverage=0.4):
    if not 0 <= page_min_coverage <= 1:
        raise ValueError("Invalid page coverage threshold")
    source = normalize(reference)
    matched, rejected = [], []
    for index, line in enumerate(lines):
        text = normalize(line["text"])
        reason = None
        left, top, right, bottom = line["bbox"]
        if line["min_word_confidence"] < 90:
            reason = "word-confidence-below-90"
        elif len(text) < 20 or len(text.split()) < 3:
            reason = "anchor-too-short"
        elif bottom - top > height * 0.08 or right - left < 2 * (bottom - top):
            reason = "not-line-like"
        elif "\ufffd" in text or any(unicodedata.category(c) == "Co" for c in text):
            reason = "invalid-glyph"
        positions, start = [], 0
        while text and (found := source.find(text, start)) >= 0:
            end = found + len(text)
            if (found == 0 or source[found - 1].isspace()) and (end == len(source) or source[end].isspace()):
                positions.append(found)
            start = found + 1
        if reason is None and len(positions) != 1:
            reason = "no-unique-exact-source-anchor"
        if reason:
            rejected.append({"line_index": index, **line, "reason": reason})
        else:
            matched.append({"line_index": index, **line, "text": text,
                            "source_span": [positions[0], positions[0] + len(text)]})
    if any(a["source_span"][1] > b["source_span"][0] or
           a["bbox"][3] > b["bbox"][1] for a, b in zip(matched, matched[1:])):
        rejected.extend({**line, "reason": "anchor-order-or-line-overlap"} for line in matched)
        matched = []
    coverage = sum(len(line["text"]) for line in matched) / max(1, len(source))
    if coverage < page_min_coverage:
        rejected.extend({**line, "reason": "page-exact-anchor-coverage-below-40-percent"} for line in matched)
        matched = []
    return matched, rejected, coverage


def run(source_root, output, *, protocol="v1", works=None):
    if protocol not in ("v1", "v2"):
        raise ValueError("Unknown printed replay protocol")
    from PIL import Image
    source_root, output = Path(source_root), Path(output)
    pages = verify_source_package(source_root) if works is None else verify_source_package(source_root, works=works)
    if works is not None:
        policy = json.loads((source_root / "source-policy.json").read_text(encoding="utf-8"))
        if policy["works"] != works:
            raise ValueError("Explicit work configuration differs from frozen source policy")
    if not pages or {p["split"] for p in pages} != {"replay-candidate", "replay-probe"}:
        raise ValueError("Require nonempty candidate and probe sources")
    for command in ("ddjvu", "tesseract"):
        if not shutil.which(command):
            raise RuntimeError(f"Missing {command}; use the CPU Colab notebook")
    output.mkdir(parents=True, exist_ok=False)
    originals = output / "originals"
    originals.mkdir()
    candidates, quarantine, page_reports = [], [], []
    for page in pages:
        original = originals / (page["work_family"] + ".djvu")
        if not original.exists():
            body = download(page["original_scan_url"], limit=100_000_000)
            if hashlib.sha1(body).hexdigest() != page["original_scan_sha1"]:
                raise ValueError("Original DjVu SHA1 mismatch")
            original.write_bytes(body)
        if hashlib.sha1(original.read_bytes()).hexdigest() != page["original_scan_sha1"]:
            raise ValueError("Cached original DjVu SHA1 mismatch")
        directory = output / "pages" / page["id"]
        directory.mkdir(parents=True)
        ppm = directory / "page.ppm"
        subprocess.run(["ddjvu", "-format=ppm", f"-page={page['scan_page']}",
                        "-subsample=1", str(original), str(ppm)], check=True, timeout=120)
        with Image.open(ppm) as image:
            image.load()
            width, height = image.size
            native = image.convert("RGB")
            native.save(directory / "page.png")
        process = subprocess.run(["tesseract", str(directory / "page.png"), "stdout",
                                  "-l", "pol", "--oem", "1", "--psm", "6", "tsv"],
                                 check=True, timeout=180, capture_output=True, text=True, encoding="utf-8")
        (directory / "tesseract.tsv").write_text(process.stdout, encoding="utf-8")
        (directory / "tesseract.stderr.txt").write_text(process.stderr, encoding="utf-8")
        reference = (source_root / page["reference_file"]).read_text(encoding="utf-8")
        lines = parse_tsv(process.stdout, width, height)
        matched, rejected, coverage = exact_anchors(lines, reference, width, height,
            page_min_coverage=0.4 if protocol == "v1" else 0)
        for line in matched:
            identifier = f"{page['id']}-line-{line['line_index']:03d}"
            pair_dir = output / "pairs" / page["split"]
            pair_dir.mkdir(parents=True, exist_ok=True)
            left, top, right, bottom = line["bbox"]
            crop_box = [max(0, left - 2), max(0, top - 1), min(width, right + 2), min(height, bottom + 1)]
            image_path, text_path = pair_dir / (identifier + ".png"), pair_dir / (identifier + ".txt")
            native.crop(crop_box).save(image_path)
            text_path.write_text(line["text"], encoding="utf-8")
            candidates.append({"id": identifier, "page_id": page["id"],
                "line_index": line["line_index"], "tesseract_key": line["key"],
                "work_family": page["work_family"], "split": page["split"],
                "image": image_path.relative_to(output).as_posix(), "image_sha256": digest(image_path),
                "text_file": text_path.relative_to(output).as_posix(), "text_sha256": digest(text_path),
                "text": line["text"], "bbox": line["bbox"], "crop_bbox": crop_box,
                "native_page_dimensions": [width, height], "native_page_sha256": digest(directory / "page.png"),
                "source_span_normalized": line["source_span"], "reference_sha256": page["reference_sha256"],
                "source_revision": page["source_revision"], "source_url": page["source_url"],
                "original_scan_sha1": page["original_scan_sha1"],
                "original_scan_sha256": digest(original), "min_word_confidence": line["min_word_confidence"],
                "label_status": "source-validated-text-automatic-exact-line-anchor",
                "crop_review_status": "not-human-reviewed", "scan_license": page["scan_license"],
                "transcription_license": page["transcription_license"],
                "eligible_for_training": False, "eligible_for_evaluation": False})
        quarantine.extend({"page_id": page["id"], **r} for r in rejected)
        page_reports.append({"page_id": page["id"], "split": page["split"], "lines": len(lines),
                             "candidates": len(matched), "anchor_coverage_before_page_gate": coverage,
                             "native_dimensions": [width, height]})
        print("LINES_READY", page["id"], len(matched), "of", len(lines), flush=True)
    write_rows(output / "manifest.jsonl", candidates)
    write_rows(output / "quarantine.jsonl", quarantine)
    write_json(output / "report.json", {"schema": "slayer-printed-replay-mining-" + protocol, "pages": page_reports,
        "page_min_coverage": 0.4 if protocol == "v1" else 0,
        "candidates": len(candidates), "splits": dict(Counter(r["split"] for r in candidates)),
        "quarantine_reasons": dict(Counter(r["reason"] for r in quarantine)),
        "eligible_for_training": False, "eligible_for_evaluation": False,
        "limitations": "Exact-anchor selection favors easy lines; not independently reviewed crops or a representative benchmark. No spelling/hyphen/long-s normalization."})
    write_json(output / "environment.json", {"python_pillow": importlib.metadata.version("pillow"),
        "tesseract": subprocess.check_output(["tesseract", "--version"], text=True),
        "ddjvu": subprocess.run(["ddjvu", "--help"], capture_output=True, text=True).stderr,
        "runner_sha256": digest(__file__), "source_checksums_sha256": digest(source_root / "checksums.json"),
        "code_revision": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]),
                                                   "rev-parse", "HEAD"], text=True).strip(),
        "created_at": datetime.now(timezone.utc).isoformat()})
    # Keep originals in the work directory; evidence has selected crops and receipts, not entire books.
    evidence = output / "evidence"
    evidence.mkdir()
    for name in ("manifest.jsonl", "quarantine.jsonl", "report.json", "environment.json"):
        shutil.copyfile(output / name, evidence / name)
    for name in ("checksums.json", "manifest.jsonl", "source-policy.json"):
        shutil.copyfile(source_root / name, evidence / ("source-" + name))
    shutil.copytree(source_root, evidence / "source-package")
    if protocol == "v2":
        for page in pages:
            target = evidence / "native-pages" / page["id"]
            target.mkdir(parents=True)
            for name in ("page.png", "tesseract.tsv", "tesseract.stderr.txt"):
                shutil.copyfile(output / "pages" / page["id"] / name, target / name)
    if (output / "pairs").exists():
        shutil.copytree(output / "pairs", evidence / "pairs")
    write_json(evidence / "checksums.json", {p.relative_to(evidence).as_posix(): digest(p)
        for p in sorted(evidence.rglob("*")) if p.is_file()})
    archive = Path(shutil.make_archive(str(output / ("printed-replay-pilot-" + protocol + "-evidence")), "zip", evidence))
    print("EVIDENCE_ZIP", archive, flush=True)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", choices=("v1", "v2"), default="v1")
    parser.add_argument("--works-config", help="Explicit work identities for a frozen expansion input")
    args = parser.parse_args()
    works = json.loads(Path(args.works_config).read_text(encoding="utf-8"))["works"] if args.works_config else None
    run(args.source_root, args.output, protocol=args.protocol, works=works)
