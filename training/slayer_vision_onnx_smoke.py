"""Reproducible two-page OCR diagnostic for the SLAYER Vision ONNX export."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import tarfile
import time
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


MODEL_REPO = "PiotrSty/slayer-vision-onnx"
MODEL_REVISION = "22b86dc311bb6c7213ddfc1924d5d90119f29c10"
BASE_LM = "SlayerLab/goLLeM-110M-PL-SFT-merged"
BASE_LM_REVISION = "a319aedb2b705f72c17823deef509225e59e3b0a"
VISION_MODEL = "google/siglip-base-patch16-224"
VISION_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
DATASET_REPO = "PiotrSty/impact-print-v2"
DATASET_REVISION = "a2480fde6f15284701458ff370b81cce50dc5c2d"
DATASET_FILE = "impact-print-v2-test.tar.gz"
DATASET_SHA256 = "0a9ffa126029703726fc5883a8279ddccf15763c4fdd483ed9b8cc034bdb42f0"
MODEL_FILES = (
    "embed_tokens.onnx",
    "embed_tokens.onnx.data",
    "lm_embeds.onnx",
    "lm_embeds.onnx.data",
    "manifest.json",
    "vision_projector.onnx",
    "vision_projector.onnx.data",
)
MAX_CONTEXT = 512
IMAGE_TOKENS = 196
DEFAULT_MAX_NEW_TOKENS = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_metric_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in unicodedata.normalize("NFC", text).splitlines()]
    return "\n".join(line for line in lines if line)


def decode_sequence(tokenizer, token_ids: list[int]) -> str:
    """Decode the complete sequence so byte-level tokens are not corrupted."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def preprocess_siglip(image):
    """Apply the pinned SigLIP 224px preprocessing without importing Transformers."""
    import numpy as np
    from PIL import Image

    image = image.convert("RGB").resize((224, 224), resample=Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    pixels = (pixels - np.float32(0.5)) / np.float32(0.5)
    return np.transpose(pixels, (2, 0, 1))[None, ...]


def validate_generation_limit(max_new_tokens: int) -> None:
    if not 1 <= max_new_tokens <= MAX_CONTEXT - IMAGE_TOKENS:
        raise ValueError(
            f"max_new_tokens must be in 1..{MAX_CONTEXT - IMAGE_TOKENS}; "
            "image embeddings already consume 196 positions"
        )


def normalized_tar_path(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    parts: list[str] = []
    for part in pure.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise ValueError(f"Unsafe archive member: {name}")
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        raise ValueError(f"Unsafe archive member: {name}")
    return Path(*parts)


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if sum(member.size for member in members) > 2_000_000_000:
            raise ValueError("Dataset archive is unexpectedly large")
        for member in members:
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsafe archive member: {member.name}")
            relative = normalized_tar_path(member.name)
            target = (output_root / relative).resolve()
            if not target.is_relative_to(output_root):
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            with source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)


def find_image(work_dir: Path, bench_dir: Path, record: dict) -> Path:
    relative = record["image"].replace("\\", "/").split("impact-corpus/")[-1]
    for base in (work_dir, bench_dir):
        for candidate in (base / "impact-corpus" / relative, base / relative):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(relative)


def _model_hashes(model_dir: Path) -> dict[str, str]:
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")
    return {name: sha256(model_dir / name) for name in MODEL_FILES}


def _make_sessions(model_dir: Path):
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            f"CUDAExecutionProvider unavailable: {available}. "
            "In Colab select Runtime > Change runtime type > T4 GPU and restart."
        )
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sessions = {
        "vision": ort.InferenceSession(
            str(model_dir / "vision_projector.onnx"), options, providers=providers
        ),
        "embed": ort.InferenceSession(
            str(model_dir / "embed_tokens.onnx"), options, providers=providers
        ),
        "lm": ort.InferenceSession(
            str(model_dir / "lm_embeds.onnx"), options, providers=providers
        ),
    }
    return sessions, available


def generate(sessions, pixel_values, tokenizer, max_new_tokens: int) -> dict:
    import numpy as np

    validate_generation_limit(max_new_tokens)
    image_embeds = sessions["vision"].run(
        None, {"pixel_values": pixel_values.astype(np.float32)}
    )[0]
    if image_embeds.shape != (1, IMAGE_TOKENS, 768):
        raise ValueError(f"Unexpected image embedding shape: {image_embeds.shape}")

    generated: list[int] = []
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")
    stopped_on_eos = False
    for _ in range(max_new_tokens):
        if generated:
            ids = np.asarray([generated], dtype=np.int64)
            token_embeds = sessions["embed"].run(None, {"ids": ids})[0]
            inputs_embeds = np.concatenate((image_embeds, token_embeds), axis=1)
        else:
            inputs_embeds = image_embeds
        attention_mask = np.ones(inputs_embeds.shape[:2], dtype=np.int64)
        logits = sessions["lm"].run(
            None,
            {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask},
        )[0]
        next_id = int(np.argmax(logits[0, -1]))
        if next_id == eos_token_id:
            stopped_on_eos = True
            break
        generated.append(next_id)

    return {
        "text": decode_sequence(tokenizer, generated),
        "token_ids": generated,
        "token_count": len(generated),
        "stopped_on_eos": stopped_on_eos,
    }


def _environment(available_providers: list[str]) -> dict:
    packages = (
        "huggingface_hub",
        "jiwer",
        "numpy",
        "onnxruntime-gpu",
        "Pillow",
        "tokenizers",
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "onnxruntime_providers": available_providers,
    }


def run(work_dir: Path = Path("/content/slayer-vision-onnx-smoke"), pages: int = 2,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download
    from jiwer import cer, wer
    from PIL import Image
    from tokenizers import Tokenizer

    validate_generation_limit(max_new_tokens)
    work_dir = Path(work_dir)
    if work_dir.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        work_dir = work_dir.with_name(f"{work_dir.name}-{suffix}")
    work_dir.mkdir(parents=True)
    output_dir = work_dir / "evidence"
    output_dir.mkdir()

    model_dir = Path(snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=list(MODEL_FILES),
    ))
    model_hashes = _model_hashes(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest = {
        "base_lm": BASE_LM,
        "vision_encoder": VISION_MODEL,
        "image_size": 224,
        "image_tokens": IMAGE_TOKENS,
        "lm_hidden": 768,
        "vocab_size": 32000,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Manifest mismatch for {key}: {manifest.get(key)!r}")

    tokenizer_path = hf_hub_download(
        BASE_LM, "tokenizer.json", revision=BASE_LM_REVISION
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if tokenizer.get_vocab_size(with_added_tokens=True) != 32000:
        raise ValueError("Unexpected tokenizer vocabulary size")
    if tokenizer.token_to_id("<|endoftext|>") != 0:
        raise ValueError("Unexpected tokenizer contract")

    processor_config_path = Path(hf_hub_download(
        VISION_MODEL, "preprocessor_config.json", revision=VISION_REVISION
    ))
    processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
    expected_processor = {
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "resample": 3,
        "rescale_factor": 1 / 255,
        "size": {"height": 224, "width": 224},
    }
    for key, expected in expected_processor.items():
        if processor_config.get(key) != expected:
            raise ValueError(f"Preprocessor mismatch for {key}: {processor_config.get(key)!r}")

    dataset_archive = Path(hf_hub_download(
        DATASET_REPO,
        DATASET_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
    ))
    if sha256(dataset_archive) != DATASET_SHA256:
        raise ValueError("Dataset archive checksum mismatch")
    safe_extract_tar(dataset_archive, work_dir)
    bench_dir = work_dir / "impact-print-v2"
    records = [
        json.loads(line)
        for line in (bench_dir / "test_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 36 or not 1 <= pages <= len(records):
        raise ValueError(f"Expected 36 records and pages in 1..36, got {len(records)}, {pages}")

    sessions, providers = _make_sessions(model_dir)
    predictions = []
    started = time.perf_counter()
    for record in records[:pages]:
        image_path = find_image(work_dir, bench_dir, record)
        if sha256(image_path) != record["sha256"]:
            raise ValueError(f"Image checksum mismatch: {record['id']}")
        page_started = time.perf_counter()
        error = None
        try:
            image = Image.open(image_path)
            pixel_values = preprocess_siglip(image)
            result = generate(sessions, pixel_values, tokenizer, max_new_tokens)
        except Exception as exc:  # Keep failed pages in the metric denominator.
            result = {"text": "", "token_ids": [], "token_count": 0, "stopped_on_eos": False}
            error = f"{type(exc).__name__}: {exc}"
        reference = normalize_metric_text(record["text"])
        prediction = normalize_metric_text(result["text"])
        row = {
            "id": record["id"],
            "image": record["image"],
            "image_sha256": record["sha256"],
            "reference": reference,
            "prediction": prediction,
            "token_ids": result["token_ids"],
            "token_count": result["token_count"],
            "stopped_on_eos": result["stopped_on_eos"],
            "elapsed_seconds": round(time.perf_counter() - page_started, 3),
            "error": error,
        }
        predictions.append(row)
        print(f"\n=== {record['id']} ===")
        print("REFERENCE:", reference[:1000])
        print("PREDICTION:", prediction[:1000])
        print("TOKENS:", row["token_count"], "EOS:", row["stopped_on_eos"], "ERROR:", error)

    references = [row["reference"] for row in predictions]
    hypotheses = [row["prediction"] for row in predictions]
    metrics = {
        "cer_micro": cer(references, hypotheses),
        "wer_micro": wer(references, hypotheses),
        "pages": pages,
        "failed_pages": sum(row["error"] is not None for row in predictions),
        "replacement_char_predictions": sum("\ufffd" in row["prediction"] for row in predictions),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "diagnostic-smoke" if pages == 2 else "full-frozen-test",
        "claim": "Not SOTA evidence; full-page capability and decoding are under test.",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "base_lm": BASE_LM,
        "base_lm_revision": BASE_LM_REVISION,
        "vision_model": VISION_MODEL,
        "vision_revision": VISION_REVISION,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_archive_sha256": DATASET_SHA256,
        "model_file_sha256": model_hashes,
        "pages": pages,
        "max_context": MAX_CONTEXT,
        "image_tokens": IMAGE_TOKENS,
        "max_new_tokens": max_new_tokens,
        "preprocessing": "Pinned SiglipImageProcessor; RGB; resize 224x224; rescale and normalize mean/std 0.5.",
        "decoding": "Greedy argmax; complete ID sequence decoded once; no post-correction; NFC/whitespace only for metrics.",
        "limitations": [
            "The model was described as image captioning, not faithful OCR.",
            "Image embeddings leave at most 316 LM positions for output tokens.",
            "The ONNX LM has no KV cache and recomputes the prefix for every token.",
            "The full page is resized to 224x224, which can erase small print.",
        ],
    }
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "run.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(_environment(providers), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    archive_path = work_dir / "slayer-vision-onnx-smoke-evidence.zip"
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            archive.write(path, path.name)
    print("\nMETRICS:", json.dumps(metrics, ensure_ascii=False, indent=2))
    print("EVIDENCE:", archive_path)
    return archive_path
