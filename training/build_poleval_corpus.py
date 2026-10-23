"""Build a pinned Qwen3-VL corpus from the official PolEvalDS train split.

Layout matches what training.runpod_qwen3vl_histfinetune consumes:
pages.jsonl rows {id, text, image, image_sha256, split} plus
pages/{train,validation}/images/. Subtask A only (page text); images are
copied from the PolEvalDS clone and bound by sha256.

Split policy: deterministic components over source-page identity
(strona_bazowa) and identical image bytes, hashed to ~10% validation, so no
group or repeated scan ever straddles train/validation. No benchmark page may
enter the corpus (the runner re-verifies against the frozen test A hashes).
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poleval", type=Path, required=True,
                        help="PolEvalDS clone root (contains data/train)")
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if cli.output.exists():
        raise FileExistsError("fresh output directory required")
    train_dir = cli.poleval / "data/train"
    rows = []
    for family in ("real", "real_degraded", "real_handwritten", "synthetic/A_markdown"):
        folder = train_dir / family
        with (folder / "metadata.csv").open(encoding="utf-8", newline="") as handle:
            meta = {row["id"]: row for row in csv.DictReader(handle)}
        for pid, m in sorted(meta.items()):
            if m["zadanie"] != "A":
                raise ValueError(f"unexpected task in A family: {pid}")
            image = next((folder / (pid + ext) for ext in (".png", ".jpeg", ".jpg")
                          if (folder / (pid + ext)).exists()), None)
            if image is None:
                raise FileNotFoundError(pid)
            text = (folder / (pid + ".md")).read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"empty reference: {pid}")
            group = m.get("strona_bazowa") or pid
            rows.append({"id": pid, "family": family, "group": group, "image_src": image,
                         "text": text, "image_sha256": digest(image),
                         "typ_dokumentu": m.get("typ_dokumentu"), "okres": m.get("okres"),
                         "licencja": m.get("licencja"), "zniszczenia": m.get("zniszczenia")})
    # deterministic grouped split: rows linked by strona_bazowa OR identical
    # image bytes form one component that never straddles train/dev
    parent = {r["id"]: r["id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_group, by_sha = {}, {}
    for row in rows:
        if row["group"] in by_group:
            union(row["id"], by_group[row["group"]])
        by_group[row["group"]] = row["id"]
        if row["image_sha256"] in by_sha:
            union(row["id"], by_sha[row["image_sha256"]])
        by_sha[row["image_sha256"]] = row["id"]
    components = collections.defaultdict(list)
    for row in rows:
        components[find(row["id"])].append(row)
    out_rows = []
    for root, members in sorted(components.items()):
        key = sorted({m["image_sha256"] for m in members})
        bucket = int(hashlib.sha256("".join(key).encode()).hexdigest()[:8], 16) % 10
        split = "validation" if bucket == 0 else "train"
        for row in members:
            row["split"] = split
            out_rows.append(row)
    counts = collections.Counter(r["split"] for r in out_rows)
    if counts["validation"] == 0 or counts["train"] == 0:
        raise ValueError("empty split")
    train_shas = {r["image_sha256"] for r in out_rows if r["split"] == "train"}
    dev_shas = {r["image_sha256"] for r in out_rows if r["split"] == "validation"}
    if train_shas & dev_shas:
        raise ValueError("image straddles train and validation")

    (cli.output / "pages/train/images").mkdir(parents=True)
    (cli.output / "pages/validation/images").mkdir(parents=True)
    manifest = []
    for row in sorted(out_rows, key=lambda r: r["id"]):
        target = cli.output / f"pages/{row['split']}/images/{row['id']}{row['image_src'].suffix.lower()}"
        target.write_bytes(row["image_src"].read_bytes())
        if digest(target) != row["image_sha256"]:
            raise ValueError(f"copy checksum mismatch: {row['id']}")
        manifest.append({"id": row["id"], "text": row["text"],
                         "image": target.relative_to(cli.output).as_posix(),
                         "image_sha256": row["image_sha256"], "split": row["split"],
                         "family": row["family"], "typ_dokumentu": row["typ_dokumentu"],
                         "okres": row["okres"], "licencja": row["licencja"],
                         "zniszczenia": row["zniszczenia"],
                         "source": "slayerlabs/PolEvalDS data/train"})
    (cli.output / "pages.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest), encoding="utf-8")
    report = {"schema": "poleval-corpus-v1", "pages": len(manifest),
              "splits": dict(counts),
              "by_family": dict(collections.Counter(r["family"] for r in manifest)),
              "pages_manifest_sha256": digest(cli.output / "pages.jsonl"),
              "grouped_split_policy": "components over strona_bazowa links and identical "
                                      "image sha256; hash bucket 10% validation"}
    (cli.output / "corpus-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
