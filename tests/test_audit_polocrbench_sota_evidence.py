import json
from pathlib import Path
import shutil

import pytest

from tests.test_sota_benchmark import TEXTS, by_path, staged
from training import audit_polocrbench_sota_evidence as audit_module
from training.build_polocrbench_sota_colab import IMPACT_REPOSITORY, IMPACT_REVISION, IMPACT_SHA256
from training.run_sota_benchmark import PROTOCOL_VERSION, run
from training.run_vision_baseline import PROMPT_VERSION
from training.validate_submission import ZERO_SHOT_PROMPTS


def build_evidence(tmp_path, staged_dir, *, models=("paddlevl", "qwen3vl"), mutate=None):
    lookup = by_path([json.loads(line) for line in (staged_dir / "manifest.jsonl").read_text().splitlines()])
    for name in ("runs", "evidence"):
        shutil.rmtree(tmp_path / name, ignore_errors=True)
    (tmp_path / "evidence.zip").unlink(missing_ok=True)
    runs = tmp_path / "runs"
    summary = {}
    for model in models:
        report = run(model, staged_dir, runs / model,
                     predictor=lambda p: lookup[p.relative_to(staged_dir).as_posix()])
        summary[model] = {key: report[key] for key in ("cer_micro", "wer_micro", "structure_similarity")}
        summary[model]["errors_or_missing"] = 0
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for model in models:
        for name in ("predictions.jsonl", "score.json", "run.json"):
            shutil.copyfile(runs / model / name, evidence / f"{model}-{name}")
    (evidence / "receipt.json").write_text(json.dumps({
        "protocol_version": PROTOCOL_VERSION, "code_revision": "a" * 40,
        "impact_repository": IMPACT_REPOSITORY, "impact_revision": IMPACT_REVISION,
        "impact_sha256": IMPACT_SHA256, "models": list(models),
        "measurement_only": True, "training_performed": False}), encoding="utf-8")
    (evidence / "staged-verification.json").write_text(json.dumps({
        "pages_verified": 36, "archive_sha256": IMPACT_SHA256}), encoding="utf-8")
    (evidence / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if mutate:
        mutate(evidence)
    archive = tmp_path / "evidence.zip"
    shutil.make_archive(str(archive.with_suffix("")), "zip", evidence)
    return archive


@pytest.fixture()
def staged_dir(tmp_path):
    return tmp_path / "benchmark"


def test_clean_evidence_is_recomputed_and_accepted(tmp_path, staged_dir):
    staged(staged_dir, TEXTS)
    archive = build_evidence(tmp_path, staged_dir)
    report = audit_module.audit(archive, staged_dir)
    assert set(report["models"]) == {"paddlevl", "qwen3vl"}
    assert report["models"]["paddlevl"]["cer_micro"] == 0.0
    assert report["training_performed"] is False and report["promoted"] is False
    assert report["frozen_input"]["pages"] == 36
    assert report["metrics_recomputed_from"] == "returned per-page predictions"


def test_tampered_prediction_is_rejected(tmp_path, staged_dir):
    staged(staged_dir, TEXTS)

    def mutate(evidence):
        path = evidence / "qwen3vl-predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["text"] = "zmieniony tekst"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    archive = build_evidence(tmp_path, staged_dir, models=("qwen3vl",), mutate=mutate)
    with pytest.raises(ValueError, match="differs|digest"):
        audit_module.audit(archive, staged_dir)


def test_frozen_prompt_receipt_is_enforced(tmp_path, staged_dir):
    staged(staged_dir, TEXTS)

    def mutate(evidence):
        path = evidence / "qwen3vl-run.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["prompt_sha256"] = "f" * 64
        path.write_text(json.dumps(data), encoding="utf-8")

    archive = build_evidence(tmp_path, staged_dir, models=("qwen3vl",), mutate=mutate)
    with pytest.raises(ValueError, match="prompt receipt"):
        audit_module.audit(archive, staged_dir)


def test_training_flag_or_bad_input_pin_is_rejected(tmp_path, staged_dir):
    staged(staged_dir, TEXTS)

    def mutate_run(evidence):
        path = evidence / "paddlevl-run.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["training_performed"] = True
        path.write_text(json.dumps(data), encoding="utf-8")

    archive = build_evidence(tmp_path, staged_dir, models=("paddlevl",), mutate=mutate_run)
    with pytest.raises(ValueError, match="measurement-only"):
        audit_module.audit(archive, staged_dir)

    def mutate_receipt(evidence):
        path = evidence / "receipt.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["impact_sha256"] = "e" * 64
        path.write_text(json.dumps(data), encoding="utf-8")

    archive = build_evidence(tmp_path, staged_dir, models=("paddlevl",), mutate=mutate_receipt)
    with pytest.raises(ValueError, match="Frozen input receipt"):
        audit_module.audit(archive, staged_dir)


def test_member_set_and_summary_are_strict(tmp_path, staged_dir):
    staged(staged_dir, TEXTS)

    def add_extra(evidence):
        (evidence / "not-in-protocol.txt").write_text("x", encoding="utf-8")

    archive = build_evidence(tmp_path, staged_dir, models=("paddlevl",), mutate=add_extra)
    with pytest.raises(ValueError, match="member set"):
        audit_module.audit(archive, staged_dir)

    def break_summary(evidence):
        path = evidence / "summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["paddlevl"]["cer_micro"] = 0.5
        path.write_text(json.dumps(data), encoding="utf-8")

    archive = build_evidence(tmp_path, staged_dir, models=("paddlevl",), mutate=break_summary)
    with pytest.raises(ValueError, match="summary"):
        audit_module.audit(archive, staged_dir)


def test_receipt_requires_prompt_version_and_model_coverage(tmp_path, staged_dir):
    staged(staged_dir, TEXTS)

    def mutate(evidence):
        path = evidence / "receipt.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["models"] = ["gpt5"]
        path.write_text(json.dumps(data), encoding="utf-8")

    archive = build_evidence(tmp_path, staged_dir, models=("paddlevl",), mutate=mutate)
    with pytest.raises(ValueError, match="model"):
        audit_module.audit(archive, staged_dir)
