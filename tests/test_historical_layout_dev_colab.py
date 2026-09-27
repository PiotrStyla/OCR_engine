import json
from pathlib import Path

from training.build_historical_layout_dev_colab import build


def test_layout_notebook_contract(tmp_path):
    target = tmp_path / "layout.ipynb"
    build(target)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert notebook["metadata"]["accelerator"] == "GPU"
    assert "5a6ab7b99ad56a15812c91df353866ae7cc745eb" in source
    assert "536fc014963c2c55626e89f9bd07500826cb930f05ca3f8848352ca80c8856b1" in source
    assert "pages/validation/metadata.jsonl" in source
    assert "history_testA_manifest" not in source
    assert "impact-print-v2-test.tar.gz" not in source
    assert source.count("recognizer.recognize_lines") == 1
    assert "sort_reading_order_columns" in source
    assert "order_by_pagexml_regions" in source
    assert "pagexml_region_oracle" in source
    assert "raw_predictions_included': False" in source
    assert "images_or_references_included': False" in source
    assert "if evidence_dir.exists():" in source
    assert "files.download(str(evidence_zip))" in source
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])


def test_frozen_layout_config_is_development_only():
    config = json.loads(Path(
        "experiments/2026-09-27/historical-layout-dev-v1/config.json"
    ).read_text(encoding="utf-8"))
    assert config["status"] == "frozen before implementation and inference"
    assert config["dataset"]["pages"] == 15
    assert len(config["dataset"]["collections"]) == 5
    assert config["boundaries"]["test_pages_used_for_tuning"] == 0
    assert config["boundaries"]["pagexml_oracle_is_promotable"] is False
    assert config["boundaries"]["historical_spelling_modernized"] is False
