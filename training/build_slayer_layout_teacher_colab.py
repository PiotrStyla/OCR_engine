"""Generate the pinned Colab notebook for one SLAYER layout teacher run."""
from __future__ import annotations

import json
from pathlib import Path

from training.build_historical_recognizer_colab import code_cell


CODE_REVISION = "f3d2247094914f28e9f91b8bc8ee2cc8dd951b32"
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-09-28/slayer-layout-teacher-pilot-v1/config.json"
)
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def build(target: str | Path, teacher_id: str | None = None) -> None:
    if teacher_id is not None and teacher_id not in CONFIG["teachers"]:
        raise ValueError(f"Unknown teacher: {teacher_id}")
    if teacher_id is None:
        parameters = '''TEACHER_ID = "qwen3-vl-4b"  # @param ["qwen3-vl-4b", "doclayout-yolo", "surya-layout2"]
PAGES = 2  # @param {type:"integer"}

assert TEACHER_ID in {"qwen3-vl-4b", "doclayout-yolo", "surya-layout2"}
assert 1 <= PAGES <= 60
print(f"Teacher={TEACHER_ID} pages={PAGES}")
'''
    else:
        parameters = f'''TEACHER_ID = {teacher_id!r}
PAGES = 2

assert TEACHER_ID == {teacher_id!r}
assert PAGES == 2
print(f"Locked smoke run: teacher={{TEACHER_ID}} pages={{PAGES}}")
'''
    install = '''import subprocess
import sys

PACKAGES = {
    "qwen3-vl-4b": [
        "huggingface_hub==0.36.2", "pillow==11.3.0",
        "transformers==4.57.6", "tokenizers==0.22.2", "bitsandbytes==0.48.1",
        "accelerate==1.13.0", "sentencepiece==0.2.1",
    ],
    "doclayout-yolo": [
        "huggingface_hub==0.36.2", "pillow==11.3.0", "doclayout-yolo==0.0.4",
    ],
    "surya-layout2": [
        "surya-ocr==0.22.1", "transformers==5.12.1",
        "huggingface_hub==1.5.0", "pillow==10.4.0",
    ],
}
RESET = {
    "qwen3-vl-4b": ["torchao", "transformers", "tokenizers", "huggingface_hub"],
    "doclayout-yolo": [],
    "surya-layout2": ["transformers", "tokenizers", "huggingface_hub", "pillow"],
}
if RESET[TEACHER_ID]:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", *RESET[TEACHER_ID]],
        check=False,
    )
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir",
     *PACKAGES[TEACHER_ID]],
    check=True,
)
print("INSTALL_OK", TEACHER_ID)
'''
    setup = f'''import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

CODE_REVISION = {CODE_REVISION!r}
EXPECTED_CONFIG = {CONFIG!r}

repo = Path("/content/OCR_engine")
if not repo.exists():
    subprocess.run(
        ["git", "clone", "https://github.com/PiotrStyla/OCR_engine.git", str(repo)],
        check=True,
    )
subprocess.run(["git", "-C", str(repo), "fetch", "origin", CODE_REVISION], check=True)
subprocess.run(
    ["git", "-C", str(repo), "checkout", "--detach", CODE_REVISION], check=True
)
resolved = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
assert resolved == CODE_REVISION

config_path = repo / "experiments/2026-09-28/slayer-layout-teacher-pilot-v1/config.json"
CONFIG = json.loads(config_path.read_text(encoding="utf-8"))
assert CONFIG == EXPECTED_CONFIG
os.chdir(repo)
sys.path.insert(0, str(repo))

import torch
assert torch.cuda.is_available(), "Select a GPU runtime, then Run all."
expected_package = CONFIG["teachers"][TEACHER_ID]["package"].split("==")
assert importlib.metadata.version(expected_package[0]) == expected_package[1]
EXPECTED_RUNTIME = {{
    "qwen3-vl-4b": {{"huggingface_hub": "0.36.2", "Pillow": "11.3.0"}},
    "doclayout-yolo": {{"huggingface_hub": "0.36.2", "Pillow": "11.3.0"}},
    "surya-layout2": {{
        "huggingface_hub": "1.5.0", "Pillow": "10.4.0", "transformers": "5.12.1",
    }},
}}
for package, version in EXPECTED_RUNTIME[TEACHER_ID].items():
    assert importlib.metadata.version(package) == version, (package, version)
if TEACHER_ID == "qwen3-vl-4b":
    from transformers import AutoModelForImageTextToText
    assert AutoModelForImageTextToText is not None
print("PINNED_RUNTIME_OK", torch.cuda.get_device_name(0), CODE_REVISION)
'''
    inference = '''from training.slayer_layout_teacher_pilot import run

result_archive = run(TEACHER_ID, CONFIG, pages=PAGES, output_root="/content")
print("PRIVATE_EVIDENCE_READY", result_archive)
'''
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "colab": {"name": Path(target).name, "provenance": []},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "scope",
                "metadata": {},
                "source": [
                    f"# SLAYER-OCR layout teacher: {teacher_id or 'selectable pilot'}\n",
                    "Use a fresh GPU runtime for exactly one teacher. Start with the default "
                    "two-page smoke run. After all three smoke ZIPs pass consensus inspection, "
                    "repeat in three fresh runtimes with `PAGES = 60`.\n",
                    "The run uses pinned public train pages and pinned model revisions. Reference "
                    "text is never sent to a teacher. The downloaded private evidence ZIP contains "
                    "proposals, provenance and checksums, but no page images or reference text. "
                    "Nothing is published automatically.\n",
                ],
            },
            code_cell("parameters", parameters),
            code_cell("install", install),
            code_cell("setup", setup),
            code_cell("inference", inference),
            {
                "cell_type": "code",
                "id": "download",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "from google.colab import files\n",
                    "files.download(str(result_archive))\n",
                ],
            },
        ],
    }
    Path(target).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    directory = Path(__file__).parent
    build(directory / "colab_slayer_layout_teacher_pilot_v1.ipynb")
    build(directory / "colab_slayer_layout_doclayout_yolo_smoke_v1.ipynb",
          teacher_id="doclayout-yolo")
    build(directory / "colab_slayer_layout_surya_smoke_v1.ipynb",
          teacher_id="surya-layout2")
