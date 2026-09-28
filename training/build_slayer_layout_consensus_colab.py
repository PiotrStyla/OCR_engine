"""Generate the pinned Colab notebook for private teacher consensus."""
from __future__ import annotations

import json
from pathlib import Path

from training.build_historical_recognizer_colab import code_cell
from training.build_slayer_layout_teacher_colab import CONFIG


CONSENSUS_CODE_REVISION = "2c953866adc5ec7f8701072ab0a5eafdfd60a242"
POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-09-28/slayer-layout-consensus-policy-v2.json"
)
POLICY = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["consensus"]


def build(target: str | Path) -> None:
    setup = f'''import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

CODE_REVISION = {CONSENSUS_CODE_REVISION!r}
EXPECTED_CONFIG = {CONFIG!r}
EXPECTED_POLICY = {POLICY!r}

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
assert subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip() == CODE_REVISION
CONFIG = json.loads(
    (repo / "experiments/2026-09-28/slayer-layout-teacher-pilot-v1/config.json")
    .read_text(encoding="utf-8")
)
assert CONFIG == EXPECTED_CONFIG
POLICY = json.loads(
    (repo / "experiments/2026-09-28/slayer-layout-consensus-policy-v2.json")
    .read_text(encoding="utf-8")
)["consensus"]
assert POLICY == EXPECTED_POLICY
os.chdir(repo)
sys.path.insert(0, str(repo))
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
work = Path("/content") / ("slayer-layout-consensus-" + run_id)
print("PINNED_CONSENSUS_RUNTIME_OK", CODE_REVISION)
'''
    upload = '''from google.colab import files

uploaded = files.upload()
assert len(uploaded) == 3, "Upload exactly the three teacher evidence ZIPs."
upload_dir = work / "uploads"
upload_dir.mkdir(parents=True)
archives = []
for name, content in uploaded.items():
    assert Path(name).name == name and name.lower().endswith(".zip")
    target = upload_dir / name
    target.write_bytes(content)
    archives.append(target)
print("UPLOAD_OK", [path.name for path in archives])
'''
    consensus = '''from training.slayer_layout_consensus_colab import combine_archives

result_archive, summary = combine_archives(
    archives, work / "combined", CONFIG, POLICY, code_revision=CODE_REVISION
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("PRIVATE_CONSENSUS_READY", result_archive)
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
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "scope",
                "metadata": {},
                "source": [
                    "# SLAYER-OCR layout consensus v2\n",
                    "Run this notebook after obtaining one evidence ZIP from each pinned teacher: "
                    "Qwen3-VL, DocLayout-YOLO and Surya Layout. Upload exactly those three ZIPs.\n",
                    "The notebook verifies archive safety, checksums, model revisions, run identity, "
                    "page identity, frozen teacher configuration and consensus policy v2 before "
                    "building COCO weak labels, "
                    "a review queue and hard-example records. It never publishes the result.\n",
                ],
            },
            code_cell("setup", setup),
            code_cell("upload", upload),
            code_cell("consensus", consensus),
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
    build(Path(__file__).with_name("colab_slayer_layout_consensus_v2.ipynb"))
