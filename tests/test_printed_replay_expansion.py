import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from training.build_printed_replay_colab import build
from training.prepare_printed_replay_sources import WORKS, validate_works


CONFIG = Path(__file__).resolve().parents[1] / "experiments/2026-10-07/printed-replay-expansion-v1/works.json"


def test_expansion_has_new_work_level_splits():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    works = config["works"]
    validate_works(works)
    assert len(works) == 8 and sum(len(w["pages"]) for w in works) == 44
    assert len([w for w in works if w["split"] == "replay-probe"]) == 2
    assert not {w["id"] for w in works} & {w["id"] for w in WORKS}
    assert config["eligible_for_training"] is config["eligible_for_evaluation"] is False


@pytest.mark.parametrize("field", ["file", "sha1"])
def test_scan_alias_cannot_cross_work_splits(field):
    works = deepcopy(WORKS)
    works[1][field] = works[0][field]
    with pytest.raises(ValueError, match="Repeated scan"):
        validate_works(works)


def test_expansion_notebook_requires_frozen_receipt(tmp_path):
    with pytest.raises(ValueError, match="receipt"):
        build(tmp_path / "missing.ipynb", "a" * 40, "expansion-v1")


def test_expansion_notebook_pins_inputs_and_is_cpu_only(tmp_path):
    import nbformat
    spec = {"revision": "b" * 40, "sha256": "c" * 64, "path": "data/source-expansion.zip",
            "works_config": CONFIG.relative_to(CONFIG.parents[3]).as_posix(),
            "works_config_sha256": "d" * 64, "pages": 34, "work_families": 8}
    target = tmp_path / "expansion.ipynb"
    build(target, "a" * 40, "expansion-v1", input_spec=spec)
    notebook = nbformat.read(target, 4)
    nbformat.validate(notebook)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    ast.parse(code)
    assert "'--works-config', str(WORKS_CONFIG)" in code
    assert spec["revision"] in code and spec["sha256"] in code
    assert spec["works_config_sha256"] in code and "hashlib.sha256(WORKS_CONFIG.read_bytes())" in code
    assert "printed-replay-expansion-v1-evidence.zip" in code
    assert "files.upload" not in code and "torch" not in code
    assert all(not c.outputs and c.execution_count is None for c in notebook.cells if c.cell_type == "code")
