import json
from pathlib import Path
import subprocess

from training.build_historical_full_page_ab_colab import CODE_REVISION, CONFIG, build


def generated(tmp_path):
    target = tmp_path / "full-page.ipynb"
    build(target)
    return target, json.loads(target.read_text(encoding="utf-8"))


def test_committed_notebook_matches_generator(tmp_path):
    target, _ = generated(tmp_path)
    committed = Path(__file__).resolve().parents[1] / "training/colab_historical_full_page_ab_v1.ipynb"
    assert target.read_bytes() == committed.read_bytes()


def test_notebook_contract_is_pinned_private_and_nonpublishing(tmp_path):
    _, notebook = generated(tmp_path)
    assert notebook["metadata"]["accelerator"] == "GPU"
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert CODE_REVISION in source
    assert CONFIG["dataset"]["revision"] in source
    assert CONFIG["dataset"]["archive_sha256"] in source
    for model in CONFIG["models"].values():
        assert model["revision"] in source
        assert model["model_sha256"] in source
    assert "userdata.get('HF_TOKEN')" in source
    assert "getpass.getpass" in source
    assert "SINGLE_SEGMENTATION_PASS_OK" in source
    assert "evaluation-manifest.jsonl" in source
    assert "resolved.relative_to(dataset_root).as_posix()" in source
    assert "not relative.startswith('../')" in source
    assert "models[label]" in source
    assert "raw_predictions_included': False" in source
    assert "private_token_recorded': False" in source
    assert "upload_folder" not in source
    assert "push_to_hub" not in source
    assert "files.download(str(evidence_zip))" in source


def test_code_revision_resolves_to_local_commit():
    root = Path(__file__).resolve().parents[1]
    resolved = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", f"{CODE_REVISION}^{{commit}}"], text=True
    ).strip()
    assert resolved == CODE_REVISION


def test_plain_python_cells_compile(tmp_path):
    _, notebook = generated(tmp_path)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or cell["id"] == "install":
            continue
        compile("".join(cell["source"]), f"<{cell['id']}>", "exec")
