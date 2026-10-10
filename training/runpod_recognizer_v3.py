"""Recognizer v3: guarded LoRA training with the 226-pair replay package.

Runpod GPU entry point. Same experiment contract as recognizer V2: paired
variants, protected development slices, selection gates with an
unchanged-baseline fallback. Training here is an experiment — nothing is
promoted to production and no SOTA claim is made.

Training data slot replaces the 70-line reviewed set of V2 with the assembled
package: 127 pairs accepted by the expansion mining run plus 99 recovered
anchors from the replay pool (43 geometry-V2, 56 ink-gap approved). The 65
replay-probe pairs stay held out. The historical corpus, the synthetic replay
subsets and the protected development slices are the frozen V2 inputs.

Runpod start command (after the repo clone and the pinned installs):

  python3 -u /workspace/OCR_engine/training/runpod_recognizer_v3.py
"""
from datetime import datetime, timezone
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from training.audit_slayer_rfdetr_layout import safe_extract_zip
from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.merge_recognizer_reviewed_pool import verified_package
from training.protocol import pair_manifest
from training.reviewed_recognizer_selection import normalize, select, summarize
from training.run_reviewed_recognizer_colab import evaluate, preflight
from training.run_reviewed_recognizer_colab_v2 import package_model, run_logged
from training.slayer_vision_onnx_smoke import safe_extract_tar

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "experiments" / "2026-10-10" / "recognizer-v3" / "config.json"


def build_reviewed_v3(repo, expansion_root, cfg, work):
    """Materialize the 226-pair training directory; keep the probe held out."""
    spec = cfg["reviewed_v3"]
    out = work / "reviewed-v3"
    out.mkdir(parents=True)
    pool = Path(repo) / "experiments" / "2026-10-10" / "replay-pool"
    pool_rows = read_rows(pool / "manifest.jsonl")
    if len(pool_rows) != spec["from_pool"]:
        raise ValueError("Replay pool count drift")
    expansion = read_rows(Path(expansion_root) / "manifest.jsonl")
    kept = [row for row in expansion if row["split"] == spec["split"]]
    probe = [row for row in expansion if row["split"] != spec["split"]]
    if len(kept) != spec["from_expansion"] or len(probe) != spec["probe_held_out"]:
        raise ValueError("Expansion split count drift")
    if {row["id"] for row in pool_rows} & {row["id"] for row in kept}:
        raise ValueError("Overlapping ids between the pool and the accepted pairs")
    for row in pool_rows:
        shutil.copyfile(pool / "crops" / f"{row['id']}.png", out / (row["id"] + ".png"))
        shutil.copyfile(pool / "texts" / f"{row['id']}.txt", out / (row["id"] + ".txt"))
    for row in kept:
        shutil.copyfile(Path(expansion_root) / row["image"], out / (row["id"] + ".png"))
        shutil.copyfile(Path(expansion_root) / row["text_file"], out / (row["id"] + ".txt"))
    rows = pair_manifest(out)
    if len(rows) != spec["expected_pairs"]:
        raise ValueError("Reviewed v3 pair count drift")
    return out, rows, probe


def build_replays(counts, ordered, source, work):
    """Materialize deterministic synthetic replay subsets; 0 maps to None."""
    replays = {}
    for count in sorted(counts):
        if not 0 <= count <= len(ordered):
            raise ValueError("Invalid replay size")
        if count == 0:
            replays[count] = None
            continue
        replay = work / f"replay-{count}"
        replay.mkdir(parents=True)
        for row in ordered[:count]:
            for suffix in (".png", ".txt"):
                shutil.copyfile(Path(source) / (row["id"] + suffix), replay / (row["id"] + suffix))
        replays[count] = replay
    return replays


def run(output_root="/workspace", config_path=DEFAULT_CONFIG):
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import torch
    from transformers import TrOCRProcessor
    from huggingface_hub import hf_hub_download
    repo = Path(__file__).resolve().parents[1]
    config_path = Path(config_path)
    stem = config_path.resolve().parent.name
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    work = Path(output_root) / (stem + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    work.mkdir()
    evidence = work / "evidence"
    evidence.mkdir()
    model_archive = None
    try:
        shutil.copyfile(config_path, evidence / "experiment-config.json")
        write_json(evidence / "environment.json", dict(
            code_revision=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                                  text=True).strip(),
            python=sys.version, gpu=torch.cuda.get_device_name(0),
            packages={name: importlib.metadata.version(name) for name in
                      ("torch", "torchao", "transformers", "peft", "accelerate", "jiwer",
                       "huggingface_hub")}))
        dataset = Path(hf_hub_download("PiotrSty/slayer-ocr-datasets",
                                       "data/recognizer-reviewed-colab-input-v1-20261006/"
                                       "recognizer-reviewed-colab-input-v1-20261006.zip",
                                       repo_type="dataset", revision=cfg["dataset_revision"]))
        if digest(dataset) != cfg["dataset_sha256"]:
            raise ValueError("Frozen reviewed input ZIP hash mismatch")
        corpus = safe_extract_zip(dataset, work / "corpus")
        verified_package(corpus)
        source = json.loads((corpus / "config.json").read_text(encoding="utf-8"))
        synthetic = source["synthetic_replay"]
        archive = Path(hf_hub_download(synthetic["repo"], synthetic["filename"],
                                       repo_type="dataset", revision=synthetic["revision"]))
        if digest(archive) != cfg["synthetic_archive_sha256"]:
            raise ValueError("Frozen synthetic archive hash mismatch")
        synthetic_root = work / "synthetic-source"
        safe_extract_tar(archive, synthetic_root)
        matches = [path for path in synthetic_root.rglob("train")
                   if path.is_dir() and list(path.glob("*.png"))]
        if len(matches) != 1:
            raise ValueError("Expected exactly one synthetic training split")
        rows = pair_manifest(matches[0])
        if len(rows) != 2000:
            raise ValueError("Synthetic training count drift")
        ordered = sorted(rows, key=lambda row: digest_of(cfg["synthetic_order_prefix"] + row["id"]))
        replays = build_replays({variant["replay_count"] for variant in cfg["variants"]},
                                ordered, matches[0], work)
        expansion_zip = Path(hf_hub_download(cfg["expansion_evidence"]["repo"],
                                             cfg["expansion_evidence"]["filename"],
                                             repo_type="dataset",
                                             revision=cfg["expansion_evidence"]["revision"]))
        if digest(expansion_zip) != cfg["expansion_evidence"]["sha256"]:
            raise ValueError("Frozen expansion evidence ZIP hash mismatch")
        expansion_root = safe_extract_zip(expansion_zip, work / "expansion")
        reviewed_v3, reviewed_rows, probe = build_reviewed_v3(repo, expansion_root, cfg, work)
        write_rows(evidence / "reviewed-v3-manifest.jsonl", reviewed_rows)
        write_rows(evidence / "probe-heldout.jsonl", probe)
        development = {"historical-development": corpus / "development",
                       "ordinary-development": repo / "benchmarks" / "real-lines-v1" / "pairs"}
        for domain, directory in development.items():
            if not list(directory.glob("*.txt")):
                raise ValueError(f"Missing development slice: {domain}")
        base = snapshot_base(source, work)
        processor = TrOCRProcessor.from_pretrained(base)
        recipe = cfg["training"]
        token_audit = []
        for domain, directory in {**development, "reviewed-v3": reviewed_v3}.items():
            for path in sorted(directory.glob("*.txt")):
                text = path.read_text(encoding="utf-8").strip()
                ids = processor.tokenizer(text, truncation=False).input_ids
                decoded = processor.tokenizer.decode(ids, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
                if len(ids) > recipe["max_target_length"] or normalize(decoded) != normalize(text):
                    raise ValueError(f"Tokenizer length/round-trip failure: {domain}/{path.stem}")
                token_audit.append(dict(domain=domain, id=path.stem, tokens=len(ids)))
        write_rows(evidence / "tokenizer-audit.jsonl", token_audit)
        write_json(evidence / "adapter-preflight.json", preflight(base, reviewed_v3, processor, recipe))
        print("FULL_MODEL_ADAPTER_AND_EVAL_LOSS_PREFLIGHT_OK", flush=True)
        baseline_path = evidence / "baseline-predictions.jsonl"
        evaluate(base, development, processor, baseline_path)
        baseline = summarize(read_rows(baseline_path))
        write_json(evidence / "baseline-metrics.json", baseline)
        policy_path = evidence / "selection-policy.json"
        write_json(policy_path, dict(baseline=baseline, policy=cfg["selection"]))
        validation = work / "validation"
        validation.mkdir()
        for domain, directory in development.items():
            for path in directory.glob("*"):
                if path.suffix in (".png", ".txt"):
                    shutil.copyfile(path, validation / (domain + "__" + path.name))
        candidates = []
        for variant in cfg["variants"]:
            identifier = variant["id"]
            model = work / identifier
            print("VARIANT", identifier, flush=True)
            train_dirs = [str(corpus / "train")] * recipe["historical_repeats"] + [str(reviewed_v3)]
            if replays[variant["replay_count"]] is not None:
                train_dirs.append(str(replays[variant["replay_count"]]))
            command = [sys.executable, "-m", "training.train_trocr_pl", "--train-dir",
                       *train_dirs,
                       "--val-dir", str(validation), "--base", str(base), "--output", str(model),
                       "--epochs", str(recipe["epochs"]), "--batch-size", str(recipe["batch_size"]),
                       "--gradient-accumulation-steps", str(recipe["gradient_accumulation_steps"]),
                       "--lr", str(variant["learning_rate"]), "--lora-rank", str(recipe["lora_rank"]),
                       "--lora-alpha", str(recipe["lora_alpha"]),
                       "--max-target-length", str(recipe["max_target_length"]),
                       "--seed", str(recipe["seed"]), "--selection-policy", str(policy_path),
                       "--no-4bit"]
            run_logged(command, repo, evidence / (identifier + "-training.log"))
            predictions_path = evidence / (identifier + "-predictions.jsonl")
            evaluate(model, development, processor, predictions_path)
            candidate_rows = read_rows(predictions_path)
            if [(row["domain"], row["id"], row["reference"]) for row in candidate_rows] != \
               [(row["domain"], row["id"], row["reference"]) for row in read_rows(baseline_path)]:
                raise ValueError("Paired prediction identities/references changed")
            metrics = summarize(candidate_rows)
            candidates.append(dict(id=identifier, metrics=metrics))
            write_json(evidence / (identifier + "-metrics.json"), metrics)
            print("VARIANT_RESULT", identifier, json.dumps(metrics), flush=True)
            for name in ("run.json", "selection.json", "best_metrics.json", "trainer_state.json"):
                shutil.copyfile(model / name, evidence / (identifier + "-" + name))
        selection = select(baseline, candidates)
        selection.update(base_model=source["base_model"], base_revision=source["base_revision"],
                         dataset_revision=cfg["dataset_revision"], reviewed_v3_pairs=len(reviewed_rows),
                         probe_held_out=len(probe), production_promoted=False, sota_claim=False)
        write_json(evidence / "selection.json", selection)
        if not selection["baseline_retained"]:
            selected = work / "selected-model"
            selected.mkdir()
            for path in (work / selection["selected"]).iterdir():
                if path.is_file() and path.suffix in (".safetensors", ".json", ".txt", ".model"):
                    shutil.copyfile(path, selected / path.name)
            write_json(selected / "guarded-selection.json", selection)
            model_archive = work / (stem + "-model.zip")
            package_model(selected, model_archive)
        print("SELECTED", selection["selected"], flush=True)
    except Exception as exc:
        write_json(evidence / "failure.json", dict(type=type(exc).__name__, message=str(exc),
                                                   completed=False, production_promoted=False,
                                                   sota_claim=False))
        raise
    finally:
        write_json(evidence / "checksums.json",
                   {path.name: digest(path) for path in evidence.iterdir()
                    if path.is_file() and path.name != "checksums.json"})
        evidence_archive = Path(shutil.make_archive(str(work / (stem + "-evidence")), "zip", evidence))
        print("EVIDENCE_ZIP", evidence_archive, flush=True)
    return work


def digest_of(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snapshot_base(source, work):
    from huggingface_hub import snapshot_download
    base = snapshot_download(source["base_model"], revision=source["base_revision"],
                             local_dir=work / "base",
                             allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"])
    return Path(base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default="/workspace")
    args = parser.parse_args()
    print("RESULT_DIRECTORY", run(args.output_root, args.config), flush=True)
