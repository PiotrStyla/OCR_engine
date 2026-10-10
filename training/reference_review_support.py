"""Review support for the frozen reference defects (CPU-only, evidence only).

The frozen references carry 624 private-use-area characters (10 codepoints),
86 U+FFFD replacement characters and one truncated page (the errata table).
This module gathers machine evidence for the human decisions on those 34 review
items without touching any frozen reference:

- for every PUA/U+FFFD occurrence it aligns the reference with each available
  system prediction and records what the systems printed at that exact place,
  which identifies the glyph the character stands for;
- it assembles the errata page transcriptions side by side as a draft for
  two-reviewer verification against the scan.

Policy reminder: model output never repairs a frozen reference. It informs the
mapping/transcription proposal; the human reviewers decide.
"""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path

from training.full_page_pilot import digest, read_rows, write_json


SCHEMA = "polocrbench-reference-review-support-v1"
PUA_RANGE = range(0xE000, 0xF900)


def defect_indices(text):
    """Character positions of PUA and U+FFFD defects in a reference."""
    pua = [i for i, c in enumerate(text) if ord(c) in PUA_RANGE]
    uffd = [i for i, c in enumerate(text) if c == "�"]
    return pua, uffd


def replacements_at(gt, pred, indices):
    """What ``pred`` prints where ``gt`` has the characters at ``indices``."""
    matcher = SequenceMatcher(None, gt, pred, autojunk=False)
    out = {}
    for index in indices:
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if i1 <= index < i2:
                out[index] = pred[j1:j2]
                break
    return out


def collect_evidence(manifest_rows, predictions):
    """Per-codepoint replacement evidence across systems."""
    evidence = {}
    uffd_spots = {}
    for row in manifest_rows:
        gt = row["text"]
        pua, uffd = defect_indices(gt)
        if not pua and not uffd:
            continue
        for name, table in predictions.items():
            pred = table.get(row["id"])
            if pred is None:
                continue
            for index, replacement in replacements_at(gt, pred, pua + uffd).items():
                code = gt[index]
                key = f"U+{ord(code):04X}"
                entry = evidence.setdefault(key, {"codepoint": key, "occurrences": 0,
                                                  "pages": set(), "replacements": {}, "_seen": set()})
                spot = (row["id"], index)
                if spot not in entry["_seen"]:
                    entry["_seen"].add(spot)
                    entry["occurrences"] += 1
                    entry["pages"].add(row["id"])
                counter = entry["replacements"].setdefault(name, Counter())
                counter[replacement if replacement else "<empty>"] += 1
                if code == "�":
                    sample = uffd_spots.setdefault(spot, {"page_id": row["id"], "gt_index": index,
                                                          "context": gt[max(0, index - 12):index + 12]})
                    sample[name] = replacement
    for entry in evidence.values():
        entry["pages"] = sorted(entry["pages"])
        entry.pop("_seen", None)
        entry["replacements"] = {name: dict(counter.most_common())
                                 for name, counter in entry["replacements"].items()}
    return evidence, sorted(uffd_spots.values(), key=lambda s: (s["page_id"], s["gt_index"]))


def errata_draft(manifest_rows, predictions, page_id="NA2_FT__434735"):
    """Side-by-side system transcriptions of the truncated errata page."""
    gt = next(row["text"] for row in manifest_rows if row["id"] == page_id)
    transcriptions = {name: table.get(page_id, "") for name, table in predictions.items()}
    lines = [f"# Draft: {page_id} (frozen GT is truncated: {gt!r})",
             "", "System transcriptions of the page, for two-reviewer verification",
             "against the scan. None of these replaces the frozen reference.", ""]
    for name, text in transcriptions.items():
        lines += [f"## {name} ({len(text)} chars)", "", "```", text[:1200], "```", ""]
    longest = max(transcriptions.items(), key=lambda item: len(item[1])) if transcriptions else None
    if longest:
        lines += ["## Proposed consensus draft (verification needed)",
                  "", f"Primary source: `{longest[0]}` (longest transcription).",
                  "Cross-check every line against the scan before any candidate is built.", "",
                  "```", longest[1], "```"]
    return "\n".join(lines) + "\n"


def run(manifest, predictions_dir, output, models):
    manifest, predictions_dir, output = Path(manifest), Path(predictions_dir), Path(output)
    rows = read_rows(manifest)
    predictions = {}
    for name, path in models.items():
        source = predictions_dir / path
        predictions[name] = {row["id"]: row.get("text") or "" for row in read_rows(source)}
    evidence, uffd_rows = collect_evidence(rows, predictions)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "errata-draft.md").write_text(errata_draft(rows, predictions), encoding="utf-8")
    write_json(output / "pua-evidence.json", {
        "schema": SCHEMA, "codepoints": len(evidence),
        "occurrences": sum(e["occurrences"] for e in evidence.values()),
        "systems": sorted(predictions), "by_codepoint": evidence,
        "eligible_for_training": False,
        "decision": "Evidence for the PUA mapping policy; the mapping table is a human decision."})
    write_json(output / "uffd-evidence.json", {
        "schema": SCHEMA, "spots": len(uffd_rows), "samples": uffd_rows[:40],
        "decision": "Evidence for the U+FFFD review; frozen references unchanged."})
    write_json(output / "checksums.json", {p.relative_to(output).as_posix(): digest(p)
                                           for p in sorted(output.rglob("*"))
                                           if p.is_file() and p.name != "checksums.json"})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions-dir", required=True, help="Directory holding per-model predictions.jsonl files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    models = {
        "tesseract-js": "experiments/2026-09-19/impact-tesseractjs-png/predictions.jsonl",
        "gpt-4o-mini": "experiments/2026-10-08/api-testA/gpt-4o-mini/predictions.jsonl",
        "qwen3-vl-235b-api": "experiments/2026-10-08/api-testA/qwen3-vl-235b/predictions.jsonl",
        "gpt-5.4": "experiments/2026-10-08/api-testA/gpt-5.4/predictions.jsonl",
        "qwen3-vl-4b": "experiments/2026-10-10/runpod-sota/qwen3vl-predictions.jsonl",
        "paddlevl-1.6": "experiments/2026-10-10/runpod-sota/paddlevl-predictions.jsonl",
    }
    run(args.manifest, args.predictions_dir, args.output, models)
