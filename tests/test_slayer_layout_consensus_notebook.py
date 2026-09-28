import json
from pathlib import Path
import subprocess

from training.build_slayer_layout_consensus_colab import (
    CONSENSUS_CODE_REVISION,
    POLICY,
    build,
)
from training.build_slayer_layout_teacher_colab import CONFIG


def test_consensus_notebook_requires_three_private_archives(tmp_path):
    target = tmp_path / "consensus.ipynb"
    build(target)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    assert "accelerator" not in notebook["metadata"]
    assert notebook["cells"][-1]["id"] == "download"
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert CONSENSUS_CODE_REVISION in source
    assert "slayer-layout-consensus-policy-v2.json" in source
    assert "code_revision=CODE_REVISION" in source
    assert "assert len(uploaded) == 3" in source
    assert "consensus_module.combine_archives" in source
    assert "importlib.reload(consensus_module)" in source
    assert "files.download(str(result_archive))" in source
    assert "push_to_hub" not in source and "upload_folder" not in source
    for spec in CONFIG["teachers"].values():
        assert spec["revision"] in source
    assert str(POLICY["containment_threshold"]) in source


def test_consensus_notebook_outputs_are_empty_and_code_compiles(tmp_path):
    target = tmp_path / "consensus.ipynb"
    build(target)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"<{cell['id']}>", "exec")


def test_checked_in_v2_notebook_matches_generator(tmp_path):
    generated = tmp_path / "colab_slayer_layout_consensus_v2.ipynb"
    build(generated)
    checked_in = (
        Path(__file__).resolve().parents[1]
        / "training/colab_slayer_layout_consensus_v2.ipynb"
    )
    assert json.loads(generated.read_text(encoding="utf-8")) == json.loads(
        checked_in.read_text(encoding="utf-8")
    )


def test_pinned_consensus_runtime_supports_notebook_api():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"{CONSENSUS_CODE_REVISION}:training/slayer_layout_consensus_colab.py",
        ],
        text=True,
    )
    assert "def combine_archives(archives, output_root, config, consensus_policy=None," in source
    assert "code_revision=None" in source
