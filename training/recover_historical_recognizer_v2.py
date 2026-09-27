"""Recover a v2 model archive rejected only by the floating-point boundary bug."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess


TRAINING_REVISION = "8b950b8f5c42af259269fb0bb5a706f1c6724148"
ABS_TOLERANCE = 1e-12


def digest(path: str | Path) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def promotion_decision(baseline: dict, candidate: dict, tokenizer: dict) -> dict:
    def delta(name: str) -> float:
        return candidate[name]["cer"] - baseline[name]["cer"]

    def within(name: str, limit: float) -> bool:
        return delta(name) <= limit + ABS_TOLERANCE

    decision = {
        "historical_validation_improved": delta("historical-validation") < 0,
        "real_lines_regression_within_1pp": within("real-lines-v1", 0.01),
        "ehri_regression_within_2pp": within("ehri-test", 0.02),
        "tokenizer_roundtrip_clean": not tokenizer["roundtrip_mismatches"],
        "no_target_truncation": tokenizer["over_128"] == 0,
        "comparison_abs_tolerance": ABS_TOLERANCE,
        "cer_deltas": {
            name: delta(name)
            for name in ("historical-validation", "real-lines-v1", "ehri-test")
        },
    }
    decision["all_gates_passed"] = all(
        value
        for key, value in decision.items()
        if key not in {"comparison_abs_tolerance", "cer_deltas", "all_gates_passed"}
    )
    return decision


def find_work(root: str | Path = "/content") -> Path:
    candidates = sorted(
        Path(root).glob("historical-recognizer-v2-*/historical-recognizer-v2-model"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No completed historical-recognizer-v2 model in this runtime. "
            "Run the updated v2 notebook from the start."
        )
    return candidates[0].parent


def recover(work: str | Path) -> tuple[Path, Path]:
    work = Path(work)
    model_dir = work / "historical-recognizer-v2-model"
    source_evidence = work / "evidence"
    baseline = json.loads((work / "baseline-metrics.json").read_text(encoding="utf-8"))
    candidate = json.loads((work / "candidate-metrics.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((work / "tokenizer-audit.json").read_text(encoding="utf-8"))
    replay = json.loads((work / "replay-manifest.json").read_text(encoding="utf-8"))
    training_run = json.loads((model_dir / "run.json").read_text(encoding="utf-8"))

    if training_run.get("source_commit") != TRAINING_REVISION:
        raise ValueError("Training revision mismatch")
    if any(replay["exact_image_overlap"].values()):
        raise ValueError("Replay manifest reports train/evaluation overlap")
    promotion = promotion_decision(baseline, candidate, tokenizer)
    if not promotion["all_gates_passed"]:
        raise ValueError(f"Promotion gates still fail: {promotion}")

    root_files = sorted(path for path in model_dir.iterdir() if path.is_file())
    if not root_files or not any(path.name.startswith("model") for path in root_files):
        raise FileNotFoundError("Merged model files are missing")

    package_dir = work / "historical-recognizer-v2-recovered-package"
    recovered_evidence = work / "evidence-recovered"
    if package_dir.exists() or recovered_evidence.exists():
        raise FileExistsError("Recovery outputs already exist; download the existing ZIP files")
    package_dir.mkdir()
    recovered_evidence.mkdir()
    for source in root_files:
        shutil.copyfile(source, package_dir / source.name)

    model_archive = Path(
        shutil.make_archive(
            str(work / "historical-recognizer-v2-model-recovered"),
            "zip",
            package_dir,
        )
    )
    model_sha256 = digest(model_archive)

    for source in sorted(source_evidence.iterdir()):
        if source.is_file() and source.name not in {
            "checksums.json",
            "environment.json",
            "promotion-gates.json",
        }:
            shutil.copyfile(source, recovered_evidence / source.name)
    (recovered_evidence / "promotion-gates.json").write_text(
        json.dumps(promotion, indent=2) + "\n", encoding="utf-8"
    )

    environment = json.loads(
        (source_evidence / "environment.json").read_text(encoding="utf-8")
    )
    environment.update(
        model_archive_created=True,
        model_archive_sha256=model_sha256,
        recovery_protocol="v1-boundary-tolerance-1e-12",
    )
    (recovered_evidence / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    try:
        recovery_checkout = subprocess.check_output(
            ["git", "-C", "/content/OCR_engine", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        recovery_checkout = None
    original_archive = work / "historical-recognizer-v2-evidence.zip"
    recovery = {
        "schema": "ocr-engine-historical-recognizer-v2-recovery",
        "training_revision": TRAINING_REVISION,
        "recovery_checkout": recovery_checkout,
        "reason": "EHRI delta at the inclusive 0.02 boundary was rejected by floating-point drift",
        "comparison_abs_tolerance": ABS_TOLERANCE,
        "original_evidence_zip_sha256": (
            digest(original_archive) if original_archive.is_file() else None
        ),
        "model_archive_sha256": model_sha256,
        "model_root_files": [path.name for path in root_files],
        "publication": "disabled",
    }
    (recovered_evidence / "recovery.json").write_text(
        json.dumps(recovery, indent=2) + "\n", encoding="utf-8"
    )
    checksums = {
        path.name: digest(path)
        for path in sorted(recovered_evidence.iterdir())
        if path.is_file()
    }
    (recovered_evidence / "checksums.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )
    evidence_archive = Path(
        shutil.make_archive(
            str(work / "historical-recognizer-v2-evidence-recovered"),
            "zip",
            recovered_evidence,
        )
    )
    return model_archive, evidence_archive


def main() -> None:
    model_archive, evidence_archive = recover(find_work())
    print("Recovered model:", model_archive)
    print("Recovered evidence:", evidence_archive)
    try:
        from google.colab import files
    except ImportError:
        return
    files.download(str(model_archive))
    files.download(str(evidence_archive))


if __name__ == "__main__":
    main()
