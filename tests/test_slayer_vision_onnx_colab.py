import io
import json
import tarfile
from pathlib import Path

import pytest

from training.build_slayer_vision_onnx_colab import build
from training.slayer_vision_onnx_smoke import (
    BASE_LM_REVISION,
    DATASET_REVISION,
    MODEL_REVISION,
    VISION_REVISION,
    decode_sequence,
    normalize_metric_text,
    safe_extract_tar,
    validate_generation_limit,
)


def test_notebook_is_reproducible_pinned_and_compilable(tmp_path):
    target = tmp_path / "colab_slayer_vision_onnx_smoke.ipynb"
    build(target)
    saved = Path(__file__).resolve().parents[1] / "training" / target.name
    assert target.read_bytes() == saved.read_bytes()
    assert target.stat().st_size < 100_000

    notebook = json.loads(target.read_text(encoding="utf-8"))
    assert notebook["metadata"]["accelerator"] == "GPU"
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == []
        source = "".join(cell["source"])
        if not source.startswith("%pip"):
            compile(source, f"<{cell['id']}>", "exec")

    run = "".join(next(cell for cell in notebook["cells"] if cell["id"] == "run")["source"])
    for revision in (MODEL_REVISION, BASE_LM_REVISION, VISION_REVISION, DATASET_REVISION):
        assert revision in run
    assert "tokens_decoded.json" not in run
    assert "tokenizer.decode(" in run
    assert "pages=2" in run


def test_install_cell_is_version_pinned(tmp_path):
    target = tmp_path / "notebook.ipynb"
    build(target)
    notebook = json.loads(target.read_text(encoding="utf-8"))
    install = "".join(next(cell for cell in notebook["cells"] if cell["id"] == "install")["source"])
    assert "onnxruntime-gpu==1.23.0" in install
    assert "tokenizers==0.22.2" in install
    assert "transformers" not in install
    assert "huggingface_hub==0.36.2" in install
    assert "jiwer==4.0.0" in install


def test_decode_uses_one_complete_sequence_call():
    class Tokenizer:
        calls = []

        def decode(self, ids, **kwargs):
            self.calls.append((ids, kwargs))
            return "W świetne błáwaty."

    tokenizer = Tokenizer()
    assert decode_sequence(tokenizer, [95, 96, 1234]) == "W świetne błáwaty."
    assert tokenizer.calls == [
        ([95, 96, 1234], {"skip_special_tokens": True})
    ]


def test_metric_normalization_preserves_historical_spelling():
    assert normalize_metric_text("  W  świetne błáwaty.\nſama  rzecz  ") == (
        "W świetne błáwaty.\nſama rzecz"
    )


def test_generation_limit_accounts_for_image_tokens():
    validate_generation_limit(316)
    with pytest.raises(ValueError, match="1..316"):
        validate_generation_limit(317)


def test_safe_extract_rejects_parent_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        payload = b"bad"
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="Unsafe archive member"):
        safe_extract_tar(archive_path, tmp_path / "output")
