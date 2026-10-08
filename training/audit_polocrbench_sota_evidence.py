"""Audit a returned PolOCRBench SOTA measurement evidence ZIP against frozen inputs.

Verifies the pinned input receipt, the measurement-only flags, the frozen prompt
hash where a prompt applies, and recomputes every reported metric from the
returned per-page predictions using the frozen subtask A metric. Nothing is
trusted from ``score.json`` or ``summary.json`` except as values to compare.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile

from training.build_polocrbench_sota_colab import (IMPACT_REPOSITORY, IMPACT_REVISION,
                                                   IMPACT_SHA256)
from training.run_sota_benchmark import MODELS, PROTOCOL_VERSION
from training.run_vision_baseline import PROMPT_VERSION
from training.transcription_eval import evaluate
from training.validate_submission import ZERO_SHOT_PROMPTS


PAGES = 36
RECEIPT_MEMBERS = ("staged-verification.json", "summary.json", "receipt.json")
MODEL_MEMBERS = ("predictions.jsonl", "score.json", "run.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def model_members(models):
    return {f"{model}-{name}" for model in models for name in MODEL_MEMBERS}


def read_json(data, label):
    try:
        return json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {label}: {error}") from error


def close(left, right, label):
    require(isinstance(left, (int, float)) and isinstance(right, (int, float)), f"Non-numeric {label}")
    require(math.isclose(left, right, rel_tol=0, abs_tol=1e-12), f"Recomputed {label} differs: {left} != {right}")


def audit(archive, benchmark):
    archive, benchmark = Path(archive), Path(benchmark)
    manifest = benchmark / "manifest.jsonl"
    with zipfile.ZipFile(archive) as zipped:
        members = {info.filename for info in zipped.infolist() if not info.is_dir()}
        receipt = read_json(zipped.read("receipt.json"), "receipt.json") if "receipt.json" in members else None
        require(receipt is not None, "Missing receipt.json")
        models = list(receipt.get("models", []))
        require(models and all(model in MODELS for model in models), "Unknown model in receipt")
        require(members == model_members(models) | set(RECEIPT_MEMBERS),
                "Evidence member set differs from the protocol")
        require(receipt.get("protocol_version") == PROTOCOL_VERSION, "Protocol version mismatch")
        require(receipt.get("impact_repository") == IMPACT_REPOSITORY
                and receipt.get("impact_revision") == IMPACT_REVISION
                and receipt.get("impact_sha256") == IMPACT_SHA256, "Frozen input receipt mismatch")
        require(receipt.get("measurement_only") is True and receipt.get("training_performed") is False,
                "Evidence does not state a measurement-only run")
        require(isinstance(receipt.get("code_revision"), str)
                and len(receipt["code_revision"]) == 40, "Missing code revision receipt")

        verification = read_json(zipped.read("staged-verification.json"), "staged-verification.json")
        require(verification.get("pages_verified") == PAGES, "Staged page count mismatch")
        require(verification.get("archive_sha256") == IMPACT_SHA256, "Staged archive checksum mismatch")

        summary = read_json(zipped.read("summary.json"), "summary.json")
        require(set(summary) == set(models), "Summary model coverage mismatch")

        results = {}
        for model in models:
            run = read_json(zipped.read(f"{model}-run.json"), f"{model}-run.json")
            require(run.get("model") == model and run.get("protocol_version") == PROTOCOL_VERSION,
                    f"Run identity mismatch: {model}")
            require(run.get("measurement_only") is True and run.get("training_performed") is False
                    and run.get("promoted") is False, f"Run is not measurement-only: {model}")
            require(isinstance(run.get("package_versions"), dict) and run["package_versions"],
                    f"Missing package versions: {model}")
            if MODELS[model]["prompt"]:
                require(run.get("prompt_used") is True and run.get("prompt_version") == PROMPT_VERSION
                        and run.get("prompt_sha256") == ZERO_SHOT_PROMPTS[PROMPT_VERSION],
                        f"Frozen prompt receipt mismatch: {model}")
            else:
                require(run.get("prompt_used") is False and run.get("prompt_sha256") is None,
                        f"Unexpected prompt receipt: {model}")

            predictions = zipped.read(f"{model}-predictions.jsonl")
            scores = read_json(zipped.read(f"{model}-score.json"), f"{model}-score.json")
            recomputed = evaluate(manifest, _materialize(f"{model}", predictions))
            for key in ("cer_micro", "wer_micro", "structure_similarity"):
                close(recomputed[key], scores.get(key), f"{key} for {model}")
            require(recomputed["predictions_sha256"] == scores.get("predictions_sha256"),
                    f"Predictions digest mismatch: {model}")
            require(recomputed["manifest_sha256"] == scores.get("manifest_sha256"),
                    f"Manifest digest mismatch: {model}")
            require(recomputed["pages"] == scores.get("pages"), f"Page count mismatch: {model}")
            entry = summary[model]
            for key in ("cer_micro", "wer_micro", "structure_similarity"):
                close(recomputed[key], entry.get(key), f"summary {key} for {model}")
            results[model] = {"cer_micro": recomputed["cer_micro"], "wer_micro": recomputed["wer_micro"],
                              "structure_similarity": recomputed["structure_similarity"],
                              "errors_or_missing": recomputed["errors_or_missing"],
                              "pages": recomputed["pages"],
                              "package_versions": run["package_versions"]}
    return {"schema": "polocrbench-sota-evidence-audit-v1",
            "archive": archive.name, "archive_sha256": _digest(archive),
            "code_revision": receipt["code_revision"], "models": results,
            "frozen_input": {"repository": IMPACT_REPOSITORY, "revision": IMPACT_REVISION,
                             "sha256": IMPACT_SHA256, "pages": PAGES},
            "metrics_recomputed_from": "returned per-page predictions",
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "training_performed": False, "promoted": False,
            "limitations": [
                "Recomputation trusts returned predictions, not model inference.",
                "References contain U+FFFD and private-use characters; absolute CER has reference noise.",
                "Package versions identify the run but GPU wheel builds are not re-verified.",
            ]}


def _materialize(model, predictions):
    """Write returned predictions where the frozen metric can hash them."""
    import tempfile
    directory = Path(tempfile.mkdtemp(prefix="polocrbench-sota-audit-"))
    path = directory / f"{model}-predictions.jsonl"
    path.write_bytes(predictions)
    return path


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--benchmark", required=True, help="Staged benchmark: manifest.jsonl + images/")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(args.archive, args.benchmark)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "audit.json").write_text(text, encoding="utf-8")
    print(text)
