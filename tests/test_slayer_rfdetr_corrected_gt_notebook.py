import ast
import json
from pathlib import Path


NOTEBOOK = Path("training/colab_slayer_rfdetr_corrected_gt_v2.ipynb")


def notebook_source():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return notebook, "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])


def test_corrected_gt_notebook_is_pinned_and_downloadable():
    notebook, source = notebook_source()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert 'rfdetr[train]==1.11.0' in source
    assert "964691bd1ea1db7d51abfee30ec50e41628f83ca" in source
    assert "68578a6e008788fd41b6b9f64d31b3a9def82f25c615ead8fc97da5a9962fb0b" in source
    assert "ce5e68096866ab903d7722a75da1b0b5cf44215d597da49c7a8137859773fbde" in source
    assert "slayer-layout-rfdetr-dataset-v2" in source
    assert "dataset_sha256 = upload_corrected_dataset()" in source
    assert "'dataset': dataset_sha256" in source
    assert "corrected-gt-reanalysis" in source
    assert "prediction_threshold=0.05" in source
    assert "audit_threshold=0.25" in source
    assert "files.download(str(result_archive))" in source
    assert "huggingface_hub" not in source
    assert "git push" not in source


def test_corrected_gt_notebook_code_cells_compile():
    notebook, _ = notebook_source()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%pip"):
            continue
        ast.parse(source, filename=f"cell-{index}")
