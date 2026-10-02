"""Archive code-free pilot inputs/audit in the existing public HF registry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from training.full_page_pilot import digest, read_rows, safe_relative, write_json


DATA_REPO = "PiotrSty/slayer-ocr-datasets"
EVIDENCE_REPO = "PiotrSty/slayer-ocr-experiment-evidence"
DATA_PREFIX = "data/full-page-pilot-review-v1-20261002"
EVIDENCE_PREFIX = "experiments/2026-10-02/full-page-reference-audit-v1"


def prepare(dataset, output, code_revision):
    dataset, output = Path(dataset), Path(output)
    if not re.fullmatch("[0-9a-f]{40}", code_revision):
        raise ValueError("Expected full Git commit SHA")
    rows = read_rows(dataset / "manifest.jsonl")
    audit = json.loads((dataset / "reference-audit.json").read_text(encoding="utf-8"))
    if len(rows) != 80 or audit["gold_pages"] != 0:
        raise ValueError("Expected the frozen 80-page unreviewed candidate pool")
    if digest(dataset / "manifest.jsonl") != audit["manifest_sha256"]:
        raise ValueError("Manifest/audit mismatch")
    files = [dataset / name for name in ("manifest.jsonl", "inference-inputs.jsonl",
                                       "reference-audit.json", "selection.json", "config.json")]
    for row in rows:
        image = dataset / safe_relative(row["image"])
        xml = dataset / safe_relative(row["pagexml_local"])
        if (digest(image) != row["sha256"] or digest(xml) != row["pagexml_sha256"]
                or row["reference_status"] != "source-unreviewed"
                or row["completeness_verified"] is not False or row["final_test"] is not False):
            raise ValueError(f'Candidate identity/status mismatch: {row["id"]}')
        files.extend((image, xml))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "slayer-full-page-reference-publication-v1", "code_revision": code_revision,
                "dataset_repo": DATA_REPO, "dataset_prefix": DATA_PREFIX,
                "source": json.loads((dataset / "config.json").read_text(encoding="utf-8"))["dataset"],
                "pages": len(rows), "gold_pages": 0, "status": "source-unreviewed",
                "review_decisions_included": False, "code_files_included": False,
                "files": {path.relative_to(dataset).as_posix(): digest(path) for path in files}}
    write_json(output / "publication-manifest.json", manifest)
    card = ("# Full-page historical reference candidates v1\n\n"
            "80 source-unreviewed pages: 65 train, 15 validation. This is not a gold benchmark.\n\n"
            "Source: PiotrSty/impact-psnc-polish-ocr, revision "
            "c7cb156fb95d2880699c33725bbaf1fbc1008fea. Source license: CC-BY-3.0. "
            "Per-page attribution URLs, collection IDs and source hashes are retained in manifest.jsonl. "
            "Images are frame-zero PNG conversions with no resizing; source transcripts are unchanged.\n\n"
            "Historical spelling is preserved. The source XML supplies region geometry, not line boxes. "
            "Completeness and annotation correctness remain unverified. Train pages may overlap "
            "recognizer training; exact/near-duplicate contamination against pretrained models has not been excluded. "
            "No independent test score or SOTA claim.\n\n"
            f"Code: https://github.com/PiotrStyla/OCR_engine/tree/{code_revision}\n")
    (output / "README.md").write_text(card, encoding="utf-8")
    return files, manifest


def publish(dataset, output, code_revision):
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    dataset, output = Path(dataset), Path(output)
    files, manifest = prepare(dataset, output, code_revision)
    api = HfApi()
    if api.whoami()["name"] != "PiotrSty":
        raise ValueError("Unexpected HF publishing account")
    revisions = {}
    for repo, prefix in ((DATA_REPO, DATA_PREFIX), (EVIDENCE_REPO, EVIDENCE_PREFIX)):
        info = api.repo_info(repo, repo_type="dataset")
        if info.private:
            raise ValueError("Existing registry repository must be public")
        if any(path.startswith(prefix + "/") for path in api.list_repo_files(repo, repo_type="dataset")):
            raise ValueError("Versioned prefix already exists; do not overwrite published artifacts")
        revisions[repo] = info.sha
    operations = [CommitOperationAdd(path_in_repo=f"{DATA_PREFIX}/{path.relative_to(dataset).as_posix()}",
                                     path_or_fileobj=str(path)) for path in files]
    operations.extend(CommitOperationAdd(path_in_repo=f"{DATA_PREFIX}/{name}",
                                         path_or_fileobj=str(output / name))
                      for name in ("publication-manifest.json", "README.md"))
    data_commit = api.create_commit(DATA_REPO, repo_type="dataset", operations=operations,
                                    parent_commit=revisions[DATA_REPO],
                                    commit_message="Archive 80 unreviewed full-page reference candidates")
    evidence_files = {"reference-audit.json": dataset / "reference-audit.json",
                      "publication-manifest.json": output / "publication-manifest.json",
                      "README.md": output / "README.md"}
    write_json(output / "evidence-checksums.json", {name: digest(path) for name, path in evidence_files.items()})
    evidence_files["evidence-checksums.json"] = output / "evidence-checksums.json"
    evidence_commit = api.create_commit(EVIDENCE_REPO, repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=f"{EVIDENCE_PREFIX}/{name}", path_or_fileobj=str(path))
                    for name, path in evidence_files.items()], parent_commit=revisions[EVIDENCE_REPO],
        commit_message="Record full-page source-reference audit; no quality claims")
    downloaded = hf_hub_download(DATA_REPO, f"{DATA_PREFIX}/manifest.jsonl", repo_type="dataset",
                                 revision=data_commit.oid, local_dir=output / "verification")
    if digest(downloaded) != manifest["files"]["manifest.jsonl"]:
        raise ValueError("Published manifest checksum mismatch")
    result = {"dataset_revision": data_commit.oid, "evidence_revision": evidence_commit.oid,
              "dataset_url": f"https://huggingface.co/datasets/{DATA_REPO}/tree/{data_commit.oid}/{DATA_PREFIX}",
              "evidence_url": f"https://huggingface.co/datasets/{EVIDENCE_REPO}/tree/{evidence_commit.oid}/{EVIDENCE_PREFIX}",
              "manifest_download_verified": True}
    write_json(output / "publication-result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.publish:
        print(json.dumps(publish(args.dataset, args.output, args.code_revision), indent=2))
    else:
        _, manifest = prepare(args.dataset, args.output, args.code_revision)
        print(json.dumps({"dry_run": True, "pages": manifest["pages"], "files": len(manifest["files"])}, indent=2))
