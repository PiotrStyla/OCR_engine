import json
from pathlib import Path
import subprocess

from training.build_slayer_layout_teacher_colab import CODE_REVISION, CONFIG, build


def test_teacher_notebook_is_pinned_private_and_gpu_ready(tmp_path):
    target = tmp_path / "teacher.ipynb"
    build(target)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["cells"][-1]["id"] == "download"
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert CODE_REVISION in source
    assert 'PAGES = 2' in source
    assert 'files.download(str(result_archive))' in source
    assert 'from transformers import AutoModelForImageTextToText' in source
    assert 'AutoModelForMultimodalLM' not in source
    assert 'push_to_hub' not in source and 'upload_folder' not in source
    assert 'Reference text is never sent' in source
    for teacher_id, spec in CONFIG["teachers"].items():
        assert teacher_id in source
        assert spec["revision"] in source


def test_teacher_notebook_outputs_are_empty_and_code_compiles(tmp_path):
    target = tmp_path / "teacher.ipynb"
    build(target)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"<{cell['id']}>", "exec")


def test_code_revision_resolves_locally():
    root = Path(__file__).resolve().parents[1]
    resolved = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", f"{CODE_REVISION}^{{commit}}"], text=True
    ).strip()
    assert resolved == CODE_REVISION and len(CODE_REVISION) == 40
