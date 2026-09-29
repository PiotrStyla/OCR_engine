import ast
import json
from pathlib import Path


NOTEBOOK = Path("training/colab_slayer_rfdetr_layout_v1.ipynb")


def test_rfdetr_notebook_is_pinned_private_and_downloadable():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'rfdetr[train]==1.11.0' in source
    assert "b6570a7829fc96651c7a41db8cae35e59c888460" in source
    assert "f0a157c79756276e9a82e4f8010521190bf72dc8b459c81c3806e840681a0c33" in source
    assert "RFDETRSmall" in source
    assert "from rfdetr.config import RFDETRSmallConfig, TrainConfig" in source
    assert "set(RUN_CONFIG) - set(TrainConfig.model_fields)" in source
    assert "set(MODEL_CONFIG) - set(RFDETRSmallConfig.model_fields)" in source
    assert "'model': 'RFDETRSmall'" not in source
    assert "batch_size': 4" in source
    assert "grad_accum_steps': 4" in source
    assert "resolution': 512" in source
    assert "split='val'" in source
    assert "log_per_class_metrics=True" in source
    assert "checkpoint_best_total.pth" in source
    assert "files.download(str(evidence_zip))" in source
    assert "files.download(str(model_zip))" in source
    assert "huggingface_hub" not in source
    assert "github.com/PiotrStyla/OCR_engine/raw" not in source

    training_cell = next(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
        if "RUN_CONFIG =" in "".join(cell.get("source", [])))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(training_cell).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"MODEL_CONFIG", "RUN_CONFIG"}
    }
    assert assignments["MODEL_CONFIG"] == {"resolution": 512}
    assert "resolution" not in assignments["RUN_CONFIG"]


def test_rfdetr_notebook_validates_archive_before_training():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    checks = [
        "len(names) == len(set(names))",
        "mode != stat.S_IFLNK",
        "actual_dataset_sha256 == EXPECTED_DATASET_SHA256",
        "provenance['train']['pages'] == 48",
        "provenance['valid']['pages'] == 12",
        "provenance['collection_overlap'] == []",
        "actual_files == set(expected_files)",
        "sha256_file(path) == digest",
    ]
    for check in checks:
        assert check in source
