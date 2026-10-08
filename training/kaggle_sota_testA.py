# PolOCRBench test A: pomiar SOTA (Kaggle, GPU T4, Internet ON).
# Qwen3-VL-4B + PaddleOCR-VL-1.6 na zamrożonych 36 stronach, prompt zero-shot v1.
# To pomiar, nie trening: bez strojenia wag, adapterów i korekt.
# Uzycie: wklej jako JEDNA komorka (lub wgryj kaggle_sota_testA.ipynb), Run All.
# Wynik: /kaggle/working/polocrbench-sota-v1-evidence.zip (panel Output po prawej).
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE_REVISION = "PENDING_PIN"  # przypinane po publikacji tego skryptu
IMPACT_REPOSITORY = "PiotrSty/impact-print-v2"
IMPACT_REVISION = "a2480fde6f15284701458ff370b81cce50dc5c2d"
IMPACT_SHA256 = "0a9ffa126029703726fc5883a8279ddccf15763c4fdd483ed9b8cc034bdb42f0"
IMPACT_PATH = "impact-print-v2-test.tar.gz"
MODELS = ("qwen3vl", "paddlevl")

WORK = Path("/kaggle/working/polocrbench-sota-v1")
WORK.mkdir(parents=True, exist_ok=True)
python = sys.executable

print("STACK_PYTHON", sys.version)
assert subprocess.check_output([python, "-c", "import torch; print(torch.cuda.is_available())"],
                               text=True).strip() == "True", "Wlacz GPU T4: Settings > Accelerator > GPU T4."

repo = Path("/kaggle/working/OCR_engine-sota")
if not repo.exists():
    subprocess.run(["git", "clone", "https://github.com/PiotrStyla/OCR_engine.git", str(repo)], check=True)
subprocess.run(["git", "-C", str(repo), "fetch", "origin", "main"], check=True)
subprocess.run(["git", "-C", str(repo), "checkout", "--detach", CODE_REVISION], check=True)
assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == CODE_REVISION
print("CODE_REVISION", CODE_REVISION)

# Torch z obrazu Kaggle zostaje nietkniete; dokladamy tylko potrzebne pakiety.
subprocess.run([python, "-m", "pip", "install", "-q", "transformers==4.57.6", "accelerate==1.13.0",
                "bitsandbytes", "jiwer==4.0.0", "huggingface_hub==0.36.2"], check=True)
paddle = subprocess.run([python, "-m", "pip", "install", "-q", "paddlepaddle-gpu==3.3.1",
                        "--extra-index-url", "https://www.paddlepaddle.org.cn/packages/stable/cu123/",
                        "paddleocr[doc-parser]"], capture_output=True, text=True)
if paddle.returncode != 0:
    print("Kola GPU Paddle (cu123) nie weszly, probuje cu118:")
    print(paddle.stderr[-1200:])
    paddle = subprocess.run([python, "-m", "pip", "install", "-q", "paddlepaddle-gpu==3.3.1",
                            "--extra-index-url", "https://www.paddlepaddle.org.cn/packages/stable/cu118/",
                            "paddleocr[doc-parser]"], capture_output=True, text=True)
if paddle.returncode != 0:
    print("Kola GPU Paddle niedostepne dla tego Pythona; silnik CPU (poprawny, lecz wolny):")
    print(paddle.stderr[-1200:])
    subprocess.run([python, "-m", "pip", "install", "-q", "paddlepaddle==3.3.1", "paddleocr[doc-parser]"], check=True)
versions = subprocess.check_output([python, "-c",
    'import importlib.metadata as m; print({p: m.version(p) for p in ("paddleocr", "paddlepaddle", "paddlex", "transformers", "torch", "bitsandbytes")})'],
    text=True).strip()
print("STACK_PACKAGES", versions)

from huggingface_hub import hf_hub_download
archive = Path(hf_hub_download(IMPACT_REPOSITORY, IMPACT_PATH, repo_type="dataset", revision=IMPACT_REVISION))
assert hashlib.sha256(archive.read_bytes()).hexdigest() == IMPACT_SHA256, "Checksum archiwum nie zgadza sie."
staged = WORK / "benchmark"
if not (staged / "manifest.jsonl").exists():
    subprocess.run([python, "-m", "training.stage_impact_benchmark", "--archive", str(archive),
                    "--output", str(staged)], cwd=repo, check=True)
rows = [json.loads(line) for line in (staged / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == 36, f"Oczekiwano 36 stron testu A, jest {len(rows)}."
print("INPUT_READY", staged, len(rows), "stron")

for model in MODELS:
    out = WORK / "smoke" / model
    result = subprocess.run([python, "-m", "training.run_sota_benchmark", "--model", model,
                             "--benchmark", str(staged), "--output", str(out), "--limit", "1"],
                            cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    assert result.returncode == 0, f"Smoke test nie przeszedl dla {model}. Wklej blad; nie uruchamiaj pelnego pomiaru."
    row = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["status"] == "ok" and row["text"].strip(), f"Pusty/bledny smoke dla {model}: {row}"
    print("SMOKE_OK", model)

summary = {}
for model in MODELS:
    out = WORK / "runs" / model
    result = subprocess.run([python, "-m", "training.run_sota_benchmark", "--model", model,
                             "--benchmark", str(staged), "--output", str(out)], cwd=repo)
    assert result.returncode == 0, f"Pomiar nie ukonczyl sie dla {model}."
    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    summary[model] = {key: score[key] for key in ("cer_micro", "wer_micro", "structure_similarity", "errors_or_missing")}
    print("MODEL_RESULT", model, summary[model])

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
    "measurement_only": True, "training_performed": False}, indent=2), encoding="utf-8")
zip_path = Path("/kaggle/working/polocrbench-sota-measurement-v1-evidence.zip")
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(str(zip_path.with_suffix("")), "zip", evidence)
print("GOTOWY_ZIP", zip_path, zip_path.stat().st_size)
print("Pobierz ZIP z panelu Output po prawej stronie i przekaz do audytu.")
