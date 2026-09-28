import json
from pathlib import Path
import subprocess

from training.build_slayer_layout_qwen_retry_colab import (
    QWEN_RETRY_CODE_REVISION,
    TARGET,
    build_qwen_retry,
)


def _source(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return notebook, "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_qwen_retry_notebook_is_locked_and_matches_generator(tmp_path):
    generated = build_qwen_retry(tmp_path)
    notebook, source = _source(generated)
    assert generated.name == TARGET
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert "TEACHER_ID = 'qwen3-vl-4b'" in source
    assert "PAGES = 60" in source and "assert PAGES == 60" in source
    assert QWEN_RETRY_CODE_REVISION in source
    assert "code_revision=CODE_REVISION" in source
    assert 'run_metadata["code_revision"] == CODE_REVISION' in source
    assert 'run_metadata["pages_completed"] == PAGES' in source
    assert 'run_metadata["error_pages"] == 0' in source
    assert 'print("RUN_COMPLETE", run_complete' in source
    assert "qwen3-vl-4b-evidence-v3.zip" in source
    assert "# @param" not in source
    assert "push_to_hub" not in source and "upload_folder" not in source
    checked_in = Path(__file__).resolve().parents[1] / "training" / TARGET
    assert notebook == json.loads(checked_in.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), f"<{cell['id']}>", "exec")


def test_qwen_retry_runtime_contains_failure_safe_protocol():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output([
        "git", "-C", str(root), "show",
        f"{QWEN_RETRY_CODE_REVISION}:training/slayer_layout_teacher_pilot.py",
    ], text=True)
    assert "QWEN_RETRY_PROMPT" in source
    assert "TeacherInferenceError" in source
    assert "getattr(exc, 'raw_output', raw_text)" in source
