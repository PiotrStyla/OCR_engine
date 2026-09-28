import json

from training.build_slayer_layout_consensus_colab import build
from training.build_slayer_layout_teacher_colab import CODE_REVISION, CONFIG


def test_consensus_notebook_requires_three_private_archives(tmp_path):
    target = tmp_path / "consensus.ipynb"
    build(target)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    assert "accelerator" not in notebook["metadata"]
    assert notebook["cells"][-1]["id"] == "download"
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert CODE_REVISION in source
    assert "assert len(uploaded) == 3" in source
    assert "combine_archives" in source
    assert "files.download(str(result_archive))" in source
    assert "push_to_hub" not in source and "upload_folder" not in source
    for spec in CONFIG["teachers"].values():
        assert spec["revision"] in source


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
