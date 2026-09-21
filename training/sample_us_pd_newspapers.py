"""Small text-only pilot; existing OCR is not ground truth."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata
from urllib.parse import urlencode
from urllib.request import urlopen


DATASET = "PleIAs/US-PD-Newspapers"


def fetch(url):
    with urlopen(url, timeout=90) as response:
        payload = response.read(8_000_001)
    if len(payload) > 8_000_000:
        raise ValueError("Response exceeds sample budget")
    return payload


def analyze(text):
    return {
        "characters": len(text),
        "whitespace_tokens": len(text.split()),
        "replacement_characters": text.count("\ufffd"),
        "private_use_characters": sum(unicodedata.category(c) == "Co" for c in text),
        "unexpected_controls": sum(
            unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in text
        ),
        "nfc_changes_text": unicodedata.normalize("NFC", text) != text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    requests = []

    def capture(url, filename):
        payload = fetch(url)
        (args.output / filename).write_bytes(payload)
        requests.append({"url": url, "file": filename,
                         "sha256": hashlib.sha256(payload).hexdigest()})
        return json.loads(payload)

    metadata = capture("https://huggingface.co/api/datasets/" + DATASET, "hub-metadata.json")
    splits = capture("https://datasets-server.huggingface.co/splits?" + urlencode({"dataset": DATASET}), "splits.json")
    assert any(s["config"] == "default" and s["split"] == "train" for s in splits["splits"])
    records = []
    seen = set()
    for offset in (0, 100_000, 1_000_000):
        url = "https://datasets-server.huggingface.co/rows?" + urlencode({
            "dataset": DATASET, "config": "default", "split": "train",
            "offset": offset, "length": 4,
        })
        data = capture(url, f"rows-{offset}.json")
        if len(data["rows"]) != 4:
            raise ValueError("Incomplete sample")
        for item in data["rows"]:
            if item.get("truncated_cells"):
                raise ValueError("Viewer returned truncated cells")
            row = item["row"]
            if row["file_name"] in seen:
                raise ValueError("Duplicate page")
            seen.add(row["file_name"])
            text = row["text"]
            if not isinstance(text, str):
                raise ValueError("Non-text record")
            records.append({"row_index": item["row_idx"], "source": row,
                            "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            "normalized_text": unicodedata.normalize("NFC", text),
                            "normalization": "NFC only; no spelling or historical-language corrections",
                            "diagnostics": analyze(text)})
    after = capture("https://huggingface.co/api/datasets/" + DATASET, "hub-metadata-after.json")
    if after["sha"] != metadata["sha"]:
        raise ValueError("Dataset changed during capture")
    with (args.output / "sample.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "dataset": DATASET, "observed_hub_revision": metadata["sha"],
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "sampling": "4 consecutive rows at offsets 0, 100000, 1000000; convenience sample, not representative",
        "revision_caveat": "Viewer rows are not revision-pinned; raw responses and hashes are the snapshot of record",
        "pages": len(records), "newspaper_ids": sorted({r["source"]["id"] for r in records}),
        "characters": sum(r["diagnostics"]["characters"] for r in records),
        "whitespace_tokens": sum(r["diagnostics"]["whitespace_tokens"] for r in records),
        "nfc_changed_pages": sum(r["diagnostics"]["nfc_changes_text"] for r in records),
        "flagged_pages": sum(any(r["diagnostics"][k] for k in
                             ("replacement_characters", "private_use_characters", "unexpected_controls")) for r in records),
        "limits": "No images, OCR inference, human corrections or ground truth. No CER/WER. Zero Unicode flags does not imply accurate OCR.",
        "requests": requests,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# US-PD-Newspapers text pilot", "", report["limits"], "",
             f"Pages: {report['pages']}; characters: {report['characters']}; whitespace tokens: {report['whitespace_tokens']}.",
             "", report["sampling"], "", report["revision_caveat"], "",
             "Only NFC normalization is applied; original OCR is retained verbatim in sample.jsonl and API responses.", ""]
    for r in records:
        row = r["source"]
        lines += [f"## Row {r['row_index']}: {row['id']} / {row['date']} / page {row['page']}",
                  "", f"Source key: `{row['file_name']}`", "", str(r["diagnostics"]), ""]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / "checksums.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "requests"}, indent=2))


if __name__ == "__main__":
    main()
