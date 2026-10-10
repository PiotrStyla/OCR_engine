"""Build a candidate reference version with the approved PUA mapping (CPU-only).

Decision recorded 2026-10-10 (user, conversation): map private-use characters
to their historical glyphs. This module builds the candidate manifest — it never
touches the frozen one — and recomputes every measured system's CER under the
candidate so the score effect of the policy is explicit.

Mapping is applied only for codepoints with strong six-system consensus; the
ambiguous ones and U+FFFD stay unchanged and are listed for human review. The
errata page stays unchanged (its draft still needs scan verification).
"""
import argparse
import json
from pathlib import Path

from jiwer import cer

from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.transcription_eval import normalize


SCHEMA = "polocrbench-reference-candidate-v2"

# Consensus of six systems on the frozen test A (see reference-review evidence).
S_LONG = chr(0x017F)  # long s

# Consensus of six systems on the frozen test A (see reference-review evidence).
MAPPING = {
    chr(0xEADA): S_LONG + "t",    # 416x: st 688 / ft 299 (long-s misreads)
    chr(0xEBA2): S_LONG + "i",    # 113x: si 162 / fi 61
    chr(0xF51E): S_LONG + "l",    # 51x: sl 46 / fl 29 (with stroke l in scan)
    chr(0xEEC5): "ct",            # 13x: ct 37
    chr(0xEBA7): S_LONG + S_LONG + "i",  # 8x: ssi 7 / long-s-si 4
}
UNMAPPED = {
    chr(0xEBA6): "16x: ff 36 / ss 22 / long-s 4 - ligature ambiguous, needs human look",
    chr(0xE5DC): "3x: n / n-acute / ni - needs human look",
    chr(0xF516): "2x: z (4 systems) - needs human look",
    chr(0xF50E): "1x: divergent readings",
    chr(0xEBA3): "1x: divergent readings",
}


def apply_mapping(text, mapping):
    """Replace mapped codepoints; everything else stays byte-identical."""
    out = []
    for char in text:
        out.append(mapping.get(char, char))
    return "".join(out)


def verify_candidate(frozen, candidate, mapping):
    """Only mapped codepoints may differ; assert it row by row."""
    if len(frozen) != len(candidate):
        raise ValueError("Candidate changed the row count")
    mapped_chars = set(mapping)
    for before, after in zip(frozen, candidate):
        if before["id"] != after["id"] or before["sha256"] != after["sha256"]:
            raise ValueError("Candidate changed row identity")
        expected = apply_mapping(before["text"], mapping)
        if after["text"] != expected:
            raise ValueError(f"Candidate text differs beyond the mapping: {before['id']}")
    spots = sum(row["text"].count(char) for row in frozen for char in mapped_chars)
    return {"rows": len(frozen), "mapped_spots": spots, "unmapped_codepoints": len(UNMAPPED)}


def score_impact(frozen_rows, candidate_rows, predictions):
    """CER of each system under frozen vs candidate references."""
    frozen_by_id = {row["id"]: row["text"] for row in frozen_rows}
    candidate_by_id = {row["id"]: row["text"] for row in candidate_rows}
    impact = {}
    for name, table in predictions.items():
        ids = sorted(set(frozen_by_id) & set(table))
        refs_frozen = [normalize(frozen_by_id[i]) for i in ids]
        refs_candidate = [normalize(candidate_by_id[i]) for i in ids]
        hyps = [normalize(table[i]) for i in ids]
        impact[name] = {"cer_frozen": cer(refs_frozen, hyps),
                        "cer_candidate": cer(refs_candidate, hyps),
                        "delta": cer(refs_candidate, hyps) - cer(refs_frozen, hyps),
                        "pages": len(ids)}
    return impact


def run(manifest, predictions_dir, output, models, mapping=MAPPING):
    manifest, predictions_dir, output = Path(manifest), Path(predictions_dir), Path(output)
    frozen = read_rows(manifest)
    candidate = [{**row, "text": apply_mapping(row["text"], mapping)} for row in frozen]
    verification = verify_candidate(frozen, candidate, mapping)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_rows(output / "history_testA_manifest.candidate-v2.jsonl", candidate)
    predictions = {}
    for name, path in models.items():
        predictions[name] = {row["id"]: row.get("text") or "" for row in read_rows(predictions_dir / path)}
    impact = score_impact(frozen, candidate, predictions)
    write_json(output / "mapping.json", {
        "schema": SCHEMA, "decision": "map PUA to historical glyphs (user, 2026-10-10)",
        "mapping": {f"U+{ord(k):04X}": v for k, v in mapping.items()},
        "unmapped": {f"U+{ord(k):04X}": v for k, v in UNMAPPED.items()},
        "policy": "historical glyphs preserved; no normalization to modern spelling"})
    write_json(output / "impact.json", {
        "schema": SCHEMA, "verification": verification, "systems": impact,
        "effect": "candidate references score every system; deltas show the mapping cost",
        "frozen_manifest_sha256": digest(manifest),
        "eligible_for_training": False, "published": False,
        "note": "Candidate only; the frozen manifest is unchanged and publication needs review."})
    write_json(output / "checksums.json", {p.relative_to(output).as_posix(): digest(p)
                                           for p in sorted(output.rglob("*"))
                                           if p.is_file() and p.name != "checksums.json"})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions-dir", required=True)
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
