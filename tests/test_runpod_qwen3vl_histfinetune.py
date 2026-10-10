import pytest

from training.runpod_qwen3vl_histfinetune import build_finetune_rows


def test_build_finetune_rows_excludes_test_split_and_overlap():
    pages = [
        {"id": "a", "split": "train", "image_sha256": "h1"},
        {"id": "b", "split": "validation", "image_sha256": "h2"},
        {"id": "c", "split": "test", "image_sha256": "h3"},
    ]
    train, dev = build_finetune_rows(pages, {"h9"})
    assert [r["id"] for r in train] == ["a"] and [r["id"] for r in dev] == ["b"]
    with pytest.raises(ValueError, match="test A"):
        build_finetune_rows(pages, {"h1"})


def test_build_finetune_rows_rejects_empty_or_duplicate():
    with pytest.raises(ValueError, match="Empty"):
        build_finetune_rows([{"id": "a", "split": "test", "image_sha256": "h"}], set())
    with pytest.raises(ValueError, match="Duplicate"):
        build_finetune_rows([{"id": "a", "split": "train", "image_sha256": "h1"},
                             {"id": "a", "split": "validation", "image_sha256": "h2"}], set())
