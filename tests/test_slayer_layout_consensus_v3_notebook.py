import json
from pathlib import Path
import subprocess

from training.build_slayer_layout_consensus_v3_colab import (
    CONSENSUS_V3_CODE_REVISION,
    TARGET,
    build_v3,
)


def test_consensus_v3_notebook_is_pinned_and_matches_generator(tmp_path):
    generated = build_v3(tmp_path)
    notebook = json.loads(generated.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert generated.name == TARGET
    assert CONSENSUS_V3_CODE_REVISION in source
    assert "slayer-layout-consensus-policy-v3.json" in source
    assert "'allow_teacher_abstentions': True" in source
    assert "'max_teacher_error_pages': 4" in source
    assert "assert len(uploaded) == 3" in source
    assert "files.download(str(result_archive))" in source
    checked_in = Path(__file__).resolve().parents[1] / "training" / TARGET
    assert notebook == json.loads(checked_in.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), f"<{cell['id']}>", "exec")


def test_consensus_v3_runtime_supports_bounded_abstentions():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output([
        "git", "-C", str(root), "show",
        f"{CONSENSUS_V3_CODE_REVISION}:training/slayer_layout_consensus_colab.py",
    ], text=True)
    assert "allow_teacher_abstentions" in source
    assert "teacher-abstentions.jsonl" in source
    assert "teacher-abstention" in source
