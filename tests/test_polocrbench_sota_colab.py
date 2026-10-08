import ast
from pathlib import Path

import nbformat
import pytest

from training import build_polocrbench_sota_colab as builder


def build_notebook(tmp_path, revision="a" * 40, **kwargs):
    target = tmp_path / "sota.ipynb"
    builder.build(target, revision, **kwargs)
    return nbformat.read(target, 4), target


def test_builder_requires_pushed_revision_and_valid_receipt(tmp_path):
    with pytest.raises(ValueError, match="revision"):
        builder.build(tmp_path / "x.ipynb", "not-a-revision")
    with pytest.raises(ValueError, match="checksum|revision"):
        builder.build(tmp_path / "x.ipynb", "a" * 40, input_spec={"sha256": "z" * 64})
    with pytest.raises(ValueError, match="checksum|revision"):
        builder.build(tmp_path / "x.ipynb", "a" * 40, input_spec={"revision": "b" * 39})
    with pytest.raises(ValueError, match="checksum|revision"):
        builder.build(tmp_path / "x.ipynb", "a" * 40, input_spec={"revision": "z" * 40})
    with pytest.raises(ValueError, match="model"):
        builder.build(tmp_path / "x.ipynb", "a" * 40, models=("gpt5",))
    with pytest.raises(ValueError, match="Duplicate"):
        builder.build(tmp_path / "x.ipynb", "a" * 40, models=("qwen3vl", "qwen3vl"))


def test_notebook_pins_frozen_input_and_code(tmp_path):
    notebook, _ = build_notebook(tmp_path)
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    ast.parse(code)
    assert builder.IMPACT_REVISION in code and builder.IMPACT_SHA256 in code
    assert "a" * 40 in code
    assert "stage_impact_benchmark" in code and "run_sota_benchmark" in code
    assert "IMPACT_REPOSITORY" in code and "repo_type='dataset'" in code
    assert "polocrbench-sota-measurement-v1-evidence.zip" in code
    assert "files.upload" not in code and "drive.mount" in code
    assert "torch.cuda.is_available()" in code
    assert all(cell.execution_count is None and not cell.outputs
               for cell in notebook.cells if cell.cell_type == "code")


def test_notebook_smokes_before_full_run_and_isolates_models(tmp_path):
    notebook, _ = build_notebook(tmp_path, models=("qwen3vl",))
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    code = "\n".join(code_cells)
    assert "MODELS = ('qwen3vl',)" in code
    smoke_index = next(i for i, source in enumerate(code_cells) if "'smoke'/" in source)
    full_index = next(i for i, source in enumerate(code_cells) if "runs'/model" in source)
    assert smoke_index < full_index
    assert code.count("training.run_sota_benchmark") == 2  # smoke loop + full loop
    assert code.count("subprocess.run(command, cwd=repo)") == 2  # one process per model per stage
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    assert "pomiar" in markdown.lower() and "nie trening" in markdown.lower()
