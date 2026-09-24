import json
from pathlib import Path

from training.build_historical_recognizer_colab import (
    BASE_REVISION,
    CODE_REVISION,
    build,
)


def test_notebook_is_pinned_and_does_not_publish(tmp_path):
    target = tmp_path / "historical.ipynb"
    build(target)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["cells"][-1]["id"] == "download"
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert CODE_REVISION in source
    assert BASE_REVISION in source
    assert "accepted_lines'] == 258" in source
    assert "accepted_lines'] == 139" in source
    assert "roundtrip_mismatches" in source
    assert "gradient-accumulation-steps', '2'" in source
    assert "upload_folder" not in source
    assert "push_to_hub" not in source
    assert "files.download(str(evidence_zip))" in source


def test_all_plain_python_cells_compile(tmp_path):
    target = tmp_path / "historical.ipynb"
    build(target)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or cell["id"] == "install":
            continue
        compile("".join(cell["source"]), f"<{cell['id']}>", "exec")
