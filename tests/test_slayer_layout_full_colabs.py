import json
from pathlib import Path
import subprocess

from training.build_slayer_layout_full_colabs import (
    FULL_RUN_CODE_REVISION,
    TARGETS,
    build_all,
)


def _source(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return notebook, "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_full_notebooks_are_locked_to_one_teacher_and_60_pages(tmp_path):
    outputs = build_all(tmp_path)
    assert {path.name for path in outputs} == set(TARGETS.values())
    for teacher_id, name in TARGETS.items():
        notebook, source = _source(tmp_path / name)
        assert notebook["metadata"]["accelerator"] == "GPU"
        assert f"TEACHER_ID = {teacher_id!r}" in source
        assert f"assert TEACHER_ID == {teacher_id!r}" in source
        assert "PAGES = 60" in source and "assert PAGES == 60" in source
        assert FULL_RUN_CODE_REVISION in source
        assert "code_revision=CODE_REVISION" in source
        assert "# @param" not in source
        assert "push_to_hub" not in source and "upload_folder" not in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None and cell["outputs"] == []
                compile("".join(cell["source"]), f"<{cell['id']}>", "exec")


def test_checked_in_full_notebooks_match_generator(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for generated in build_all(tmp_path):
        checked_in = root / "training" / generated.name
        assert json.loads(generated.read_text(encoding="utf-8")) == json.loads(
            checked_in.read_text(encoding="utf-8")
        )


def test_pinned_full_runtime_records_code_revision():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output([
        "git", "-C", str(root), "show",
        f"{FULL_RUN_CODE_REVISION}:training/slayer_layout_teacher_pilot.py",
    ], text=True)
    assert "code_revision=None" in source
    assert "'code_revision': code_revision" in source
