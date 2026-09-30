from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


CODE_SUFFIXES = {
    ".bat",
    ".cmd",
    ".css",
    ".dll",
    ".exe",
    ".html",
    ".ipynb",
    ".js",
    ".jsx",
    ".ps1",
    ".py",
    ".pyc",
    ".pyd",
    ".sh",
    ".so",
    ".ts",
    ".tsx",
    ".whl",
}

SKIP_DIR_NAMES = {
    ".cache",
    ".git",
    "__pycache__",
    "node_modules",
    "private-tools",
    "python-deps",
}

SKIP_DIR_PREFIXES = ("pytest-", "test-")

DATASET_DIRS = {
    "annotation-review-v1",
    "body-auto-geometry-bands-v1",
    "body-auto-geometry-v1",
    "body-auto-geometry-v2",
    "body-auto-geometry-v3",
    "body-dev-corrected-draft-v1",
    "body-dev-kaggle-v1",
    "body-dev-kaggle-v2",
    "body-dev-review-v1",
    "body-dev-review-v2",
    "geometry-holdout-reference-review-v1",
    "historical-lines-smoke-v1",
    "historical-lines-smoke-v2",
    "historical-lines-v1",
    "impact-historical-lines-source",
    "impact-print-v2",
    "layout-clean-candidate-v1",
    "layout-consensus-smoke-review-v1",
    "layout-consensus-smoke-review-v2",
    "layout-consensus-smoke-v2-inputs",
    "page-0007-20260920",
    "polocrbench-history-png-v1",
    "polocrbench-history-v1",
    "printed-dev-20260921",
    "slayer-layout-clean-candidate-v1-20260929",
    "slayer-layout-consensus-full-v3-20260929",
    "slayer-layout-rfdetr-private-v1-20260929",
    "slayer-ocr-hard-10-20260929-v1",
    "slayer-rfdetr-layout-gt-review-v1-20260930",
    "us-pd-newspapers-correction-20260920",
    "us-pd-newspapers-pilot-20260920",
}

MODEL_DIRS = {"historical-recognizer-v2-model-9c992c67"}

DESKTOP_ARCHIVE_TERMS = (
    "body",
    "doclayout",
    "ehri",
    "geometry",
    "historical",
    "kraken",
    "layout",
    "printed",
    "qwen",
    "slayer",
    "surya",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build code-free Hugging Face staging trees."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--desktop", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--visibility", choices=("private", "public"), default="public"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def is_code(path: Path) -> bool:
    return path.suffix.lower() in CODE_SUFFIXES


def should_skip(relative: Path) -> bool:
    for part in relative.parts[:-1]:
        lowered = part.lower()
        if lowered in SKIP_DIR_NAMES or lowered.startswith(SKIP_DIR_PREFIXES):
            return True
    return is_code(relative) or relative.name in {".gitignore", "CACHEDIR.TAG"}


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def stage_tree(
    source: Path,
    destination_root: Path,
    destination_prefix: Path,
    category: str,
    rows: list[dict],
    *,
    skip_onnx_payload: bool = False,
) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    files = [source] if source.is_file() else sorted(source.rglob("*"))
    for file_path in files:
        if not file_path.is_file():
            continue
        relative = Path(file_path.name) if source.is_file() else file_path.relative_to(source)
        if should_skip(relative):
            continue
        if skip_onnx_payload and file_path.suffix.lower() == ".onnx":
            continue
        destination = (
            destination_root / destination_prefix
            if source.is_file()
            else destination_root / destination_prefix / relative
        )
        method = link_or_copy(file_path, destination)
        rows.append(
            {
                "category": category,
                "source": str(file_path.resolve()),
                "path": destination.relative_to(destination_root).as_posix(),
                "bytes": file_path.stat().st_size,
                "sha256": sha256(file_path),
                "staging_method": method,
            }
        )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def history_rows(repo: Path) -> list[dict]:
    raw = git(repo, "log", "--all", "--date=iso-strict", "--pretty=format:%H%x09%ad%x09%s")
    rows = []
    for line in raw.splitlines():
        commit, committed_at, subject = line.split("\t", 2)
        changed = git(
            repo,
            "show",
            "--pretty=format:",
            "--name-only",
            "--diff-filter=ACMRT",
            commit,
        ).splitlines()
        artifact_files = [
            name
            for name in changed
            if name
            and not is_code(Path(name))
            and Path(name).parts[0] in {"audits", "benchmarks", "docs", "experiments"}
        ]
        rows.append(
            {
                "commit": commit,
                "committed_at": committed_at,
                "subject": subject,
                "artifact_files": artifact_files,
            }
        )
    return rows


def card(kind: str, head: str, generated_at: str, visibility: str) -> str:
    if kind == "models":
        frontmatter = "---\nlanguage:\n- pl\nlicense: other\n---"
        title = "SLAYER-OCR model registry"
        body = (
            "Model artifacts for reproducible SLAYER-OCR experiments. "
            "Every payload is accompanied by hashes and provenance. The public "
            "SLAYER Vision ONNX artifact remains at `PiotrSty/slayer-vision-onnx`."
        )
    elif kind == "datasets":
        frontmatter = (
            "---\ntask_categories:\n- image-to-text\nlanguage:\n- pl\nlicense: other\n---"
        )
        title = "SLAYER-OCR dataset registry"
        body = (
            "Scans, annotations, frozen benchmark inputs, development splits, "
            "review workspaces and clean layout candidates. Upstream provenance and "
            "license limitations remain attached to their source artifacts."
        )
    else:
        frontmatter = "---\nlanguage:\n- pl\nlicense: other\n---"
        title = "SLAYER-OCR experiment evidence registry"
        body = (
            "Evidence, metrics, predictions, environment records, result reports "
            "and the non-code artifact timeline recovered from GitHub. Failed and negative "
            "experiments are intentionally retained."
        )
    return (
        f"{frontmatter}\n\n# {title}\n\n{body}\n\n"
        f"- Source repository: `PiotrStyla/OCR_engine`\n"
        f"- Source commit: `{head}`\n"
        f"- Registry generated: `{generated_at}`\n"
        f"- Visibility: {visibility}\n"
        "- Integrity: see `MANIFEST.jsonl` and `MANIFEST.sha256`\n\n"
        "Historical Polish spelling and transcription variants are preserved. "
        "No normalization from historical glyphs or diacritics to modern spelling is implied.\n"
    )


def finalize(
    root: Path,
    kind: str,
    head: str,
    generated_at: str,
    visibility: str,
    rows: list[dict],
) -> dict:
    write_text(root / "README.md", card(kind, head, generated_at, visibility))
    rows = sorted(rows, key=lambda row: row["path"])
    write_jsonl(root / "SOURCE-MANIFEST.jsonl", rows)

    manifest_rows = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.name in {"MANIFEST.jsonl", "MANIFEST.sha256"}:
            continue
        manifest_rows.append(
            {
                "path": file_path.relative_to(root).as_posix(),
                "bytes": file_path.stat().st_size,
                "sha256": sha256(file_path),
            }
        )
    write_jsonl(root / "MANIFEST.jsonl", manifest_rows)
    manifest_hash = sha256(root / "MANIFEST.jsonl")
    write_text(root / "MANIFEST.sha256", f"{manifest_hash}  MANIFEST.jsonl\n")
    return {
        "files": len(manifest_rows),
        "bytes": sum(row["bytes"] for row in manifest_rows),
        "manifest_sha256": manifest_hash,
    }


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)

    roots = {kind: output / kind for kind in ("datasets", "models", "evidence")}
    rows = {kind: [] for kind in roots}
    for root in roots.values():
        root.mkdir()

    data_root = repo / "data"
    for directory in sorted(path for path in data_root.iterdir() if path.is_dir()):
        name = directory.name
        lowered = name.lower()
        if lowered in SKIP_DIR_NAMES or lowered.startswith(SKIP_DIR_PREFIXES):
            continue
        if name in MODEL_DIRS:
            stage_tree(directory, roots["models"], Path("models") / name, "model", rows["models"])
        elif name in DATASET_DIRS:
            stage_tree(directory, roots["datasets"], Path("data") / name, "dataset", rows["datasets"])
        elif name == "private-layout-evidence":
            model_dir = directory / "slayer-rfdetr-layout-v1-model-20260929"
            model_zip = directory / "slayer-rfdetr-layout-v1-model-20260929.zip"
            stage_tree(model_dir, roots["models"], Path("models") / model_dir.name, "model", rows["models"])
            stage_tree(model_zip, roots["models"], Path("archives") / model_zip.name, "model-archive", rows["models"])
            for item in sorted(directory.iterdir()):
                if item in {model_dir, model_zip}:
                    continue
                stage_tree(item, roots["evidence"], Path("data") / name / item.name, "evidence", rows["evidence"])
        else:
            stage_tree(
                directory,
                roots["evidence"],
                Path("data") / name,
                "evidence",
                rows["evidence"],
                skip_onnx_payload=name == "slayer-vision-onnx-ir9-audit",
            )

    stage_tree(repo / "benchmarks", roots["datasets"], Path("github-head") / "benchmarks", "benchmark", rows["datasets"])
    for name in ("docs", "experiments", "audits"):
        stage_tree(repo / name, roots["evidence"], Path("github-head") / name, "github-artifact", rows["evidence"])
    stage_tree(repo / "README.md", roots["evidence"], Path("github-head") / "README.md", "github-artifact", rows["evidence"])

    for archive in sorted(args.desktop.glob("*.zip")):
        lowered = archive.name.lower()
        if any(term in lowered for term in DESKTOP_ARCHIVE_TERMS):
            stage_tree(archive, roots["evidence"], Path("source-archives") / archive.name, "source-archive", rows["evidence"])

    head = git(repo, "rev-parse", "HEAD")
    generated_at = datetime.now(timezone.utc).isoformat()
    history = history_rows(repo)
    write_jsonl(roots["evidence"] / "git-history" / "commits.jsonl", history)
    write_text(roots["evidence"] / "git-history" / "refs.txt", git(repo, "show-ref") + "\n")
    write_json(
        roots["evidence"] / "git-history" / "source.json",
        {
            "repository": "https://github.com/PiotrStyla/OCR_engine",
            "head": head,
            "fetched_refs": True,
            "generated_at": generated_at,
            "commit_count": len(history),
        },
    )

    summary = {
        "schema": "slayer-ocr-hf-registry-v1",
        "generated_at": generated_at,
        "source_commit": head,
        "source_repository": "https://github.com/PiotrStyla/OCR_engine",
        "code_included": False,
        "visibility": args.visibility,
        "repositories": {},
    }
    registry_metadata = {
        key: value for key, value in summary.items() if key != "repositories"
    }
    for kind, root in roots.items():
        write_json(root / "REGISTRY.json", {**registry_metadata, "artifact_class": kind})
    for kind, root in roots.items():
        summary["repositories"][kind] = finalize(
            root, kind, head, generated_at, args.visibility, rows[kind]
        )
    write_json(output / "registry-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
