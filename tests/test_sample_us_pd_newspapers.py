import hashlib
import json
import sys
from urllib.parse import parse_qs, urlparse

from training import sample_us_pd_newspapers as pilot


def test_unicode_diagnostics():
    result = pilot.analyze("e\u0301 \ufffd\ue000\x00\n\t")
    assert result["nfc_changes_text"]
    assert result["replacement_characters"] == 1
    assert result["private_use_characters"] == 1
    assert result["unexpected_controls"] == 1


def test_snapshot_preserves_raw_text_and_checksums(tmp_path, monkeypatch):
    raw = "Olde e\u0301 spelling\n"

    def fake_fetch(url):
        if "/api/datasets/" in url:
            data = {"sha": "revision"}
        elif "/splits?" in url:
            data = {"splits": [{"config": "default", "split": "train"}]}
        else:
            offset = int(parse_qs(urlparse(url).query)["offset"][0])
            data = {"rows": [{"row_idx": i, "truncated_cells": [], "row": {
                "file_name": f"page-{i}", "id": "paper", "date": "1898-01-01",
                "page": str(i), "text": raw}} for i in range(offset, offset + 4)]}
        return json.dumps(data).encode()

    output = tmp_path / "pilot"
    monkeypatch.setattr(pilot, "fetch", fake_fetch)
    monkeypatch.setattr(sys, "argv", ["pilot", "--output", str(output)])
    pilot.main()
    records = [json.loads(line) for line in (output / "sample.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 12
    assert all(record["source"]["text"] == raw for record in records)
    assert records[0]["normalized_text"] == "Olde \u00e9 spelling\n"
    for filename, expected in json.loads((output / "checksums.json").read_text()).items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
