"""SOTA measurement on frozen PolOCRBench test A (subtask A, zero-shot rules).

Runs document-understanding systems over the staged frozen test pages and
scores them with the frozen subtask A metric
(``training.transcription_eval``, v1.1). This is measurement only: no training,
no fine-tuning, no prompt edits and no promotion of any artifact.

Model adapters use published APIs only and load lazily, so this module imports
and tests on CPU without any model stack installed:

- ``qwen3vl``: ``transformers`` chat interface of Qwen3-VL with the frozen
  zero-shot prompt template for subtask A (greedy decoding).
- ``paddlevl``: the PaddleOCR-VL document-parsing pipeline (PaddleX or
  ``paddleocr.PaddleOCRVL``); no prompt, its own Markdown output is scored.

Predictions JSONL: {id, text, status: ok|error, elapsed_seconds}.
Usage:
  python -m training.run_sota_benchmark --model qwen3vl --benchmark STAGED \
      --output runs/qwen3vl [--limit 1] [--max-new-tokens 4096]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from training.run_vision_baseline import PROMPT_VERSION, load_templates
from training.transcription_eval import evaluate


PROTOCOL_VERSION = "polocrbench-sota-measurement-v1"
MODELS = {
    "qwen3vl": {"kind": "vlm", "source": "Qwen/Qwen3-VL-4B-Instruct",
                "prompt": True, "decoding": "greedy", "max_new_tokens": 4096},
    "paddlevl": {"kind": "pipeline", "source": "PaddleOCR-VL-1.6",
                 "prompt": False, "decoding": "pipeline-default", "max_new_tokens": 0},
}
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_PIXELS = 4194304  # profil 4MP sprawdzony na T4 (training/full_page_validation_4mp.py)
DEFAULT_MIN_PIXELS = 262144   # jw. (profil comparison-v5)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                          encoding="utf-8", newline="\n")


def extract_markdown(result):
    """Markdown text from a PaddleOCR-VL result (documented attribute/JSON shapes).

    An empty page is a valid prediction (the frozen metric scores it); only a
    result without usable markdown structure is an error.
    """
    markdown = result.get("markdown") if isinstance(result, dict) else getattr(result, "markdown", None)
    if isinstance(markdown, dict):
        text = markdown.get("markdown_texts")
    elif isinstance(markdown, str):
        text = markdown
    else:
        text = result.get("markdown_text") if isinstance(result, dict) else getattr(result, "markdown_text", None)
    if not isinstance(text, str):
        raise ValueError("PaddleOCR-VL result has no markdown text")
    return text.strip()


def model_image(path):
    """PNG/JPEG pass through; other formats (IMPACT TIFFs under .jpg names) are
    transcoded losslessly to PNG so vision loaders accept them. Decoded pixels
    are unchanged; the source file is never modified."""
    data = Path(path).read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff"):
        return Path(path)
    import io
    import tempfile
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        payload = image if image.mode in ("RGB", "L") else image.convert("RGB")
        target = Path(tempfile.mkdtemp(prefix="polocrbench-sota-image-")) / (Path(path).stem + ".png")
        payload.save(target, format="PNG")
    return target


def load_predictor(model, *, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, max_pixels=DEFAULT_MAX_PIXELS,
                   min_pixels=DEFAULT_MIN_PIXELS):
    """Real predictor for a model id; imports the model stack on first use."""
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    spec = MODELS[model]
    if spec["kind"] == "vlm":
        prompt = load_templates()["A"]
        import torch
        from PIL import Image
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        # Memory profile proven on Tesla T4 with 4MP pages
        # (training/full_page_pilot.py:load_qwen): 4-bit NF4, fp16, sdpa.
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=torch.float16,
                                          bnb_4bit_use_double_quant=True)
        engine = Qwen3VLForConditionalGeneration.from_pretrained(
            spec["source"], device_map={"": 0}, torch_dtype=torch.float16,
            attn_implementation="sdpa", quantization_config=quantization).eval()
        processor = AutoProcessor.from_pretrained(spec["source"],
                                                 min_pixels=min_pixels, max_pixels=max_pixels)
        eos = engine.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]

        def predict(path):
            with Image.open(path) as source:
                image = source.convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt").to(engine.device)
            with torch.inference_mode():
                outputs = engine.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                          eos_token_id=eos,
                                          pad_token_id=processor.tokenizer.pad_token_id)
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            ids = generated.tolist()
            ended = bool(ids and ids[-1] in eos)
            if not ended and len(ids) >= max_new_tokens:
                raise ValueError(f"Incomplete response: generation hit max_new_tokens={max_new_tokens}")
            return processor.decode(generated, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
        return predict

    try:
        from paddleocr import PaddleOCRVL
        pipeline, backend = PaddleOCRVL(pipeline_version="v1.6"), "paddleocr.PaddleOCRVL-1.6"
    except ImportError:
        from paddlex import create_pipeline
        pipeline, backend = create_pipeline(pipeline=spec["source"]), "paddlex:" + spec["source"]
    load_predictor.backend = backend

    def predict(path):
        results = list(pipeline.predict(str(path), max_new_tokens=max_new_tokens))
        texts = [extract_markdown(result) for result in results]
        return "\n\n".join(texts)
    return predict


def run(model, benchmark, output, *, predictor=None, limit=0, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        max_pixels=DEFAULT_MAX_PIXELS):
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if limit < 0:
        raise ValueError("Limit must be nonnegative")
    benchmark, output = Path(benchmark), Path(output)
    manifest = benchmark / "manifest.jsonl"
    rows = read_rows(manifest)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Staged manifest must contain unique nonempty records")
    selected = rows[:limit] if limit else rows
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    predict = predictor if predictor is not None else load_predictor(
        model, max_new_tokens=max_new_tokens, max_pixels=max_pixels, min_pixels=DEFAULT_MIN_PIXELS)
    predictions = []
    started = time.time()
    for row in selected:
        image = benchmark / row["image"]
        if not image.is_file():
            raise ValueError(f"Missing staged image: {row['id']}")
        page_started = time.time()
        try:
            text = predict(model_image(image))
            status, body = "ok", text
        except Exception as error:  # noqa: BLE001 - a failed page is scored, not fatal
            status, body = "error", f"{type(error).__name__}: {error}"
        predictions.append({"id": row["id"], "text": body if status == "ok" else "",
                            "status": status, "error": None if status == "ok" else body,
                            "elapsed_seconds": round(time.time() - page_started, 3)})
        print("PAGE_READY", row["id"], status, predictions[-1]["elapsed_seconds"], flush=True)
    write_rows(output / "predictions.jsonl", predictions)
    score = evaluate(manifest, output / "predictions.jsonl")
    (output / "score.json").write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    spec = MODELS[model]
    versions = {}
    for package in ("transformers", "torch", "bitsandbytes", "paddlepaddle", "paddlex", "paddleocr", "jiwer"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    run_report = {
        "protocol_version": PROTOCOL_VERSION, "model": model, "model_source": spec["source"],
        "kind": spec["kind"], "decoding": spec["decoding"], "prompt_used": spec["prompt"],
        "prompt_version": PROMPT_VERSION if spec["prompt"] else None,
        "prompt_sha256": digest(Path(__file__).resolve().parents[1] / "benchmarks" / "polocrbench"
                                / "prompts" / "zero_shot_prompt_v1.md") if spec["prompt"] else None,
        "max_new_tokens": max_new_tokens if spec["prompt"] else None,
        "max_pixels": max_pixels if spec["prompt"] else None,
        "min_pixels": DEFAULT_MIN_PIXELS if spec["prompt"] else None,
        "attn_implementation": "sdpa" if spec["prompt"] else None,
        "quantization": "bnb-nf4-4bit" if spec["prompt"] else None,
        "backend": getattr(load_predictor, "backend", None),
        "benchmark_manifest_sha256": digest(manifest),
        "pages_total": len(rows), "pages_predicted": len(selected),
        "elapsed_seconds": round(time.time() - started, 3),
        "package_versions": versions,
        "cer_micro": score["cer_micro"], "wer_micro": score["wer_micro"],
        "structure_similarity": score["structure_similarity"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "measurement_only": True, "training_performed": False, "promoted": False,
    }
    (output / "run.json").write_text(json.dumps(run_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT", model, f"CER {score['cer_micro']:.4f}", f"WER {score['wer_micro']:.4f}",
          f"structure {score['structure_similarity']:.4f}", flush=True)
    return run_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--benchmark", required=True, help="Staged directory: manifest.jsonl + images/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: first N pages only")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS,
                        help="Vision token budget per page (OOM safety on small GPUs)")
    args = parser.parse_args()
    run(args.model, args.benchmark, args.output, limit=args.limit,
        max_new_tokens=args.max_new_tokens, max_pixels=args.max_pixels)
