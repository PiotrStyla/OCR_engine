# PolOCRBench test A: pomiar SOTA (Runpod GPU pod).
# Qwen3-VL-4B + PaddleOCR-VL-1.6 na zamrożonych 36 stronach, prompt zero-shot v1.
# Pomiar, nie trening. Uruchamiane przez container start command poda:
#   curl -fsSL <raw-url-tego-pliku> -o /workspace/runpod_sota_testA.py && python3 /workspace/runpod_sota_testA.py
# Dowody leca na HF (PiotrSty/slayer-ocr-experiment-evidence), logi widac w stream-pod-logs.
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE_REVISION = "607ae333a6f8c61457aa0372eb941679f2c86088"
IMPACT_REPOSITORY = "PiotrSty/impact-print-v2"
IMPACT_REVISION = "a2480fde6f15284701458ff370b81cce50dc5c2d"
IMPACT_SHA256 = "0a9ffa126029703726fc5883a8279ddccf15763c4fdd483ed9b8cc034bdb42f0"
IMPACT_PATH = "impact-print-v2-test.tar.gz"
MODELS = ("qwen3vl", "paddlevl")
EVIDENCE_REPOSITORY = "PiotrSty/slayer-ocr-experiment-evidence"
EVIDENCE_PATH = "experiments/2026-10-08/polocrbench-sota-runpod"

WORK = Path("/workspace/polocrbench-sota-v1")
WORK.mkdir(parents=True, exist_ok=True)
python = sys.executable

print("STACK_PYTHON", sys.version, flush=True)
assert subprocess.check_output([python, "-c", "import torch; print(torch.cuda.is_available())"],
                               text=True).strip() == "True", "Pod bez GPU CUDA."
print("GPU", subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip(), flush=True)

repo = Path("/workspace/OCR_engine-sota")
if not repo.exists():
    subprocess.run(["git", "clone", "https://github.com/PiotrStyla/OCR_engine.git", str(repo)], check=True)
subprocess.run(["git", "-C", str(repo), "fetch", "origin", "main"], check=True)
subprocess.run(["git", "-C", str(repo), "checkout", "--detach", CODE_REVISION], check=True)
assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == CODE_REVISION
print("CODE_REVISION", CODE_REVISION, flush=True)

# Torch z obrazu zostaje nietkniony (kolizja NCCL przy instalacji Paddle obok).
subprocess.run([python, "-m", "pip", "install", "-q", "transformers==4.57.6", "accelerate==1.13.0",
                "bitsandbytes", "jiwer==4.0.0", "huggingface_hub==0.36.2"], check=True)
subprocess.check_call([python, "-c", "import torch; assert torch.cuda.is_available()"])
paddle_home = Path("/workspace/paddle-env")
paddle_python = str(paddle_home / "bin" / "python")
if not Path(paddle_python).exists():
    subprocess.run([python, "-m", "pip", "install", "-q", "uv"], check=True)
    subprocess.run([python, "-m", "uv", "venv", str(paddle_home)], check=True)
paddle_ok = False
for cu in ("cu126", "cu118"):
    attempt = subprocess.run([python, "-m", "uv", "pip", "install", "-q", "--python", paddle_python,
                             "paddlepaddle-gpu==3.3.1",
                             "--extra-index-url", f"https://www.paddlepaddle.org.cn/packages/stable/{cu}/",
                             "paddleocr[doc-parser]", "jiwer", "pillow"], capture_output=True, text=True)
    print("PADDLE_GPU_TRY", cu, "rc", attempt.returncode, flush=True)
    if attempt.returncode == 0:
        paddle_ok = True
        break
    print(attempt.stderr[-800:], flush=True)
if not paddle_ok:
    print("Kola GPU Paddle niedostepne; silnik CPU (poprawny, lecz wolny).", flush=True)
    subprocess.run([python, "-m", "uv", "pip", "install", "-q", "--python", paddle_python,
                    "paddlepaddle==3.3.1", "paddleocr[doc-parser]", "jiwer", "pillow"], check=True)
versions = subprocess.check_output([paddle_python, "-c",
    'import importlib.metadata as m\n'
    'def v(name):\n'
    '    try:\n'
    '        return m.version(name)\n'
    '    except m.PackageNotFoundError:\n'
    '        return None\n'
    'print({p: v(p) for p in ("paddleocr", "paddlex", "paddlepaddle", "paddlepaddle-gpu", "jiwer")})'],
    text=True).strip()
print("PADDLE_STACK", versions, flush=True)
subprocess.check_call([python, "-c", "import torch; assert torch.cuda.is_available()"])
print("TORCH_UNTOUCHED ok", flush=True)
PYTHONS = {"qwen3vl": python, "paddlevl": paddle_python}

from huggingface_hub import hf_hub_download
archive = Path(hf_hub_download(IMPACT_REPOSITORY, IMPACT_PATH, repo_type="dataset", revision=IMPACT_REVISION))
assert hashlib.sha256(archive.read_bytes()).hexdigest() == IMPACT_SHA256, "Checksum archiwum nie zgadza sie."
staged = WORK / "benchmark"
if not (staged / "manifest.jsonl").exists():
    subprocess.run([python, "-m", "training.stage_impact_benchmark", "--archive", str(archive),
                    "--output", str(staged)], cwd=repo, check=True)
rows = [json.loads(line) for line in (staged / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == 36, f"Oczekiwano 36 stron testu A, jest {len(rows)}."
print("INPUT_READY", staged, len(rows), "stron", flush=True)

for model in MODELS:
    out = WORK / "smoke" / model
    result = subprocess.run([PYTHONS[model], "-m", "training.run_sota_benchmark", "--model", model,
                             "--benchmark", str(staged), "--output", str(out), "--limit", "1"],
                            cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)
    assert result.returncode == 0, f"Smoke test nie przeszedl dla {model}."
    row = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["status"] == "ok" and row["text"].strip(), f"Pusty/bledny smoke dla {model}: {row}"
    print("SMOKE_OK", model, flush=True)

summary = {}
for model in MODELS:
    out = WORK / "runs" / model
    result = subprocess.run([PYTHONS[model], "-m", "training.run_sota_benchmark", "--model", model,
                             "--benchmark", str(staged), "--output", str(out)], cwd=repo)
    assert result.returncode == 0, f"Pomiar nie ukonczyl sie dla {model}."
    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    summary[model] = {key: score[key] for key in ("cer_micro", "wer_micro", "structure_similarity", "errors_or_missing")}
    print("MODEL_RESULT", model, summary[model], flush=True)

evidence = WORK / "evidence"
if evidence.exists():
    shutil.rmtree(evidence)
evidence.mkdir()
for model in MODELS:
    for name in ("predictions.jsonl", "score.json", "run.json"):
        shutil.copyfile(WORK / "runs" / model / name, evidence / f"{model}-{name}")
shutil.copyfile(staged / "verification.json", evidence / "staged-verification.json")
(evidence / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(evidence / "receipt.json").write_text(json.dumps({
    "protocol_version": "polocrbench-sota-measurement-v1", "code_revision": CODE_REVISION,
    "impact_repository": IMPACT_REPOSITORY, "impact_revision": IMPACT_REVISION,
    "impact_sha256": IMPACT_SHA256, "models": list(MODELS),
    "platform": "runpod-pod", "measurement_only": True, "training_performed": False},
    indent=2), encoding="utf-8")
zip_path = WORK / "polocrbench-sota-measurement-v1-evidence.zip"
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(str(zip_path.with_suffix("")), "zip", evidence)
print("GOTOWY_ZIP", zip_path, zip_path.stat().st_size, flush=True)

token = os.environ.get("HF_TOKEN", "")
if token:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    files = [evidence / name for name in sorted(os.listdir(evidence))]
    files.append(zip_path)
    info = api.upload_folder(repo_id=EVIDENCE_REPOSITORY, repo_type="dataset", path_in_repo=EVIDENCE_PATH,
                             folder_path=str(WORK),
                             allow_patterns=["polocrbench-sota-measurement-v1-evidence.zip", "evidence/*"],
                             commit_message="Add Runpod SOTA measurement evidence")
    print("HF_UPLOAD", EVIDENCE_REPOSITORY, EVIDENCE_PATH, "revision", info.oid, flush=True)
print("DONE", flush=True)
