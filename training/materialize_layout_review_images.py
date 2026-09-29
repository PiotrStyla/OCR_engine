"""Materialize pinned private-review images from a teacher selection manifest."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath

from training.slayer_layout_teacher_pilot import sha256_file


def _dataset_path(file_name: str) -> str:
    path = PurePosixPath(file_name)
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("images",):
        raise ValueError(f"Unsafe dataset image path: {file_name}")
    return (PurePosixPath("pages/train") / path).as_posix()


def materialize(selection_path: str | Path, config_path: str | Path,
                output_dir: str | Path, downloader=None) -> dict:
    selection_path, config_path, output_dir = map(
        Path, (selection_path, config_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    rows = json.loads(selection_path.read_text(encoding="utf-8"))
    if (not isinstance(rows, list) or len(rows) != dataset["selected_pages"] or
            len({row.get("id") for row in rows}) != len(rows)):
        raise ValueError("Invalid or incomplete teacher selection")
    names = [Path(row["file_name"]).name for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate review image basename")
    if downloader is None:
        from huggingface_hub import hf_hub_download

        def downloader(repo, path, revision):
            return hf_hub_download(
                repo, path, repo_type="dataset", revision=revision)

    output_dir.mkdir(parents=True)
    manifest = []
    try:
        for index, row in enumerate(rows, 1):
            source_path = _dataset_path(row["file_name"])
            source = Path(downloader(
                dataset["repo"], source_path, dataset["revision"]))
            actual_hash = sha256_file(source)
            if actual_hash != row["image_sha256"]:
                raise ValueError(f"Image checksum mismatch: {row['id']}")
            target = output_dir / Path(row["file_name"]).name
            shutil.copyfile(source, target)
            manifest.append({
                "page_id": row["id"],
                "file_name": target.name,
                "sha256": actual_hash,
                "source_path": source_path,
            })
            print(f"Image {index}/{len(rows)}: {row['id']}", flush=True)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "schema": "slayer-layout-review-images-v1",
            "dataset_repo": dataset["repo"],
            "dataset_revision": dataset["revision"],
            "selection_sha256": sha256_file(selection_path),
            "images": manifest,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        checksums = output_dir / "checksums.sha256"
        checksums.write_text(
            "".join(f"{item['sha256']}  {item['file_name']}\n" for item in manifest)
            + f"{sha256_file(manifest_path)}  manifest.json\n",
            encoding="utf-8", newline="\n")
    except Exception:
        shutil.rmtree(output_dir)
        raise
    return {"images": len(manifest), "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(
        args.selection, args.config, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
