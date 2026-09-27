import json
import math
from pathlib import Path
import subprocess

from training.build_historical_recognizer_v2_colab import (
    BASE_REVISION,
    CODE_REVISION,
    EHRI_REVISION,
    SYN_REVISION,
    build,
)
from training.recover_historical_recognizer_v2 import promotion_decision


def generated_notebook(tmp_path):
    target = tmp_path / "historical-v2.ipynb"
    build(target)
    return target, json.loads(target.read_text(encoding="utf-8"))


def test_committed_notebook_matches_generator(tmp_path):
    generated, _ = generated_notebook(tmp_path)
    committed = Path(__file__).resolve().parents[1] / "training/colab_historical_recognizer_v2.ipynb"
    assert generated.read_bytes() == committed.read_bytes()


def test_notebook_is_frozen_audited_and_does_not_publish(tmp_path):
    _, notebook = generated_notebook(tmp_path)
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["cells"][-1]["id"] == "download"
    assert [cell["id"] for cell in notebook["cells"]].index("inputs") < [
        cell["id"] for cell in notebook["cells"]
    ].index("tokenizer")
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for revision in (CODE_REVISION, BASE_REVISION, EHRI_REVISION, SYN_REVISION):
        assert revision in source
    assert "len(historical_train) == 258" in source
    assert "len(ehri_train) == 349" in source
    assert "len(synthetic_train) == 2000" in source
    assert "[:500]" in source
    assert "historical-replay-v2:" in source
    assert "effective_examples_per_epoch': 1365" in source
    assert "Exact train/evaluation image overlap" in source
    assert "tokenizer_audit['labels'] == 1402" in source
    assert "'--epochs', '4'" in source
    assert "'--lr', '2e-5'" in source
    assert "math.isclose(delta, limit, rel_tol=0.0, abs_tol=1e-12)" in source
    assert "str(corpus_root / 'train'),\n    str(corpus_root / 'train')," in source
    assert "upload_folder" not in source
    assert "push_to_hub" not in source
    assert "if promotion['all_gates_passed']" in source
    assert "files.download(str(evidence_zip))" in source
    assert "files.download(str(model_archive))" in source


def test_exact_boundary_regression_passes_promotion_gate(tmp_path):
    _, notebook = generated_notebook(tmp_path)
    candidate = "".join(
        next(cell for cell in notebook["cells"] if cell["id"] == "candidate")["source"]
    )
    baseline = {
        "historical-validation": {"cer": 0.34845049130763417},
        "real-lines-v1": {"cer": 0.053586380128384035},
        "ehri-test": {"cer": 0.2974418604651163},
    }
    observed = {
        "historical-validation": {"cer": 0.31418493323255225},
        "real-lines-v1": {"cer": 0.059726486184761375},
        "ehri-test": {"cer": 0.3174418604651163},
    }
    assert math.isclose(
        observed["ehri-test"]["cer"] - baseline["ehri-test"]["cer"],
        0.02,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    namespace = {
        "baseline_metrics": baseline,
        "evaluate_sets": lambda _: observed,
        "evaluation_sets": {name: None for name in baseline},
        "json": json,
        "model_dir": tmp_path / "model",
        "tokenizer_audit": {"roundtrip_mismatches": [], "over_128": 0},
        "work": tmp_path,
    }
    exec(candidate, namespace)
    assert namespace["promotion"]["ehri_regression_within_2pp"] is True
    assert namespace["promotion"]["all_gates_passed"] is True
    recovered = promotion_decision(baseline, observed, namespace["tokenizer_audit"])
    assert recovered["ehri_regression_within_2pp"] is True
    assert recovered["all_gates_passed"] is True


def test_code_revision_resolves_to_a_local_commit():
    root = Path(__file__).resolve().parents[1]
    resolved = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", f"{CODE_REVISION}^{{commit}}"],
        text=True,
    ).strip()
    assert len(CODE_REVISION) == 40
    assert resolved == CODE_REVISION


def test_install_cell_cleans_stale_transformers_files(tmp_path):
    _, notebook = generated_notebook(tmp_path)
    install = "".join(next(cell for cell in notebook["cells"] if cell["id"] == "install")["source"])
    assert "uninstall -q -y torchao transformers tokenizers huggingface_hub" in install
    assert "--no-cache-dir" in install
    assert "transformers==4.57.6" in install
    assert "tokenizers==0.22.2" in install


def test_all_plain_python_cells_compile(tmp_path):
    _, notebook = generated_notebook(tmp_path)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or cell["id"] == "install":
            continue
        compile("".join(cell["source"]), f"<{cell['id']}>", "exec")
