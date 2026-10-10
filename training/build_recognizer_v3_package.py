"""Assemble the recognizer v3 training package (CPU-only, candidates only).

Merges the two non-overlapping halves of the printed-replay data:

- 127 accepted pairs from the frozen expansion mining run (split
  ``replay-candidate``), rendered and verified by the mining evidence;
- 99 recovered anchors from the replay pool (43 geometry-V2 accepted plus 56
  ink-gap approved), cut at the ink band and ink-verified.

The 65 ``replay-probe`` pairs stay held out and are written to a separate file;
they never enter training. Every merged row keeps its provenance and stays
``eligible_for_training: False`` — the open human gates are listed in the
report. Image/text bytes are referenced from their evidence roots by path and
hash rather than duplicated.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from training.full_page_pilot import digest, read_rows, write_json, write_rows


SCHEMA = "slayer-recognizer-v3-package-v1"


def unify_accepted(row):
    return {"id": row["id"], "page_id": row["page_id"], "work_family": row["work_family"],
            "split": row["split"], "source": "expansion-mining-accepted",
            "cut_rule": "miner-padded-rectangle", "content_type": "unclassified-pending-review",
            "text": row["text"], "min_word_confidence": row["min_word_confidence"],
            "source_span_normalized": row["source_span_normalized"],
            "image": row["image"], "image_sha256": row["image_sha256"],
            "text_file": row["text_file"], "text_sha256": row["text_sha256"],
            "review_status": row["crop_review_status"],
            "eligible_for_training": False, "eligible_for_evaluation": False}


def unify_pool(row):
    return {"id": row["id"], "page_id": row["page_id"], "work_family": row["work_family"],
            "split": row["split"], "source": "replay-pool-recovered",
            "cut_rule": row["cut_rule"], "content_type": row["content_type"],
            "text": row["text"], "min_word_confidence": row["min_word_confidence"],
            "source_span_normalized": row["source_span_normalized"],
            "image": f"crops/{row['id']}.png", "image_sha256": digest(row["_crop"]),
            "text_file": f"texts/{row['id']}.txt", "text_sha256": digest(row["_text"]),
            "review_status": row["review_status"],
            "eligible_for_training": False, "eligible_for_evaluation": False}


def run(audit_root, pool_root, output):
    audit_root, pool_root, output = Path(audit_root), Path(pool_root), Path(output)
    accepted = [unify_accepted(row) for row in read_rows(audit_root / "evidence" / "manifest.jsonl")]
    pool_rows = read_rows(pool_root / "manifest.jsonl")
    merged = []
    for row in pool_rows:
        row["_crop"] = pool_root / "crops" / f"{row['id']}.png"
        row["_text"] = pool_root / "texts" / f"{row['id']}.txt"
        merged.append(unify_pool(row))
    candidates = [row for row in accepted if row["split"] == "replay-candidate"] + merged
    probe = [row for row in accepted if row["split"] == "replay-probe"]
    ids = [row["id"] for row in candidates + probe]
    if len(ids) != len(set(ids)):
        raise ValueError("Overlapping ids between accepted pairs and the pool")
    if any(row["eligible_for_training"] for row in candidates + probe):
        raise ValueError("Nothing may be eligible for training at package time")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_rows(output / "train-candidates.jsonl", candidates)
    write_rows(output / "probe-heldout.jsonl", probe)
    write_json(output / "report.json", {
        "schema": SCHEMA,
        "totals": {"train_candidates": len(candidates),
                   "from_expansion_mining": sum(1 for r in candidates if r["source"] == "expansion-mining-accepted"),
                   "from_replay_pool": sum(1 for r in candidates if r["source"] == "replay-pool-recovered"),
                   "probe_held_out": len(probe),
                   "work_families": dict(Counter(r["work_family"] for r in candidates)),
                   "content_types": dict(Counter(r["content_type"] for r in candidates)),
                   "cut_rules": dict(Counter(r["cut_rule"] for r in candidates))},
        "inputs": {"expansion_audit_root": str(audit_root), "replay_pool": str(pool_root),
                   "accepted_manifest_sha256": digest(audit_root / "evidence" / "manifest.jsonl"),
                   "pool_manifest_sha256": digest(pool_root / "manifest.jsonl")},
        "eligible_for_training": False, "eligible_for_evaluation": False,
        "decision": "Training package assembled as candidates; the human gates decide promotion.",
        "open_gates": ["second-reviewer sign-off on the reference candidate",
                       "rare PUA spots (5 codepoints, 23 places)", "U+FFFD (86 spots)",
                       "errata page transcription",
                       "content-type classification (64 pool + 127 mining rows unclassified)",
                       "10 pool crops flagged for ink review"],
        "limitations": ["Image/text bytes are referenced from the evidence roots by path and hash.",
                        "Protection sets (9 historical + 75 ordinary dev lines) come from the recognizer v2 package.",
                        "The 65 probe pairs are held out and never enter training."]})
    write_json(output / "checksums.json", {p.relative_to(output).as_posix(): digest(p)
                                           for p in sorted(output.rglob("*"))
                                           if p.is_file() and p.name != "checksums.json"})
    return output / "report.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True, help="Reproduced expansion audit root")
    parser.add_argument("--pool-root", required=True, help="Replay pool output directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.audit_root, args.pool_root, args.output)
