import hashlib
import json
import zipfile

from PIL import Image
import pytest

from training.audit_slayer_rfdetr_layout import (
    match_page,
    render_comparison,
    resolve_prediction_labels,
    safe_extract_zip,
    verify_checksums,
)


def test_safe_extract_and_checksum_verification(tmp_path):
    archive = tmp_path / "good.zip"
    content = b"payload"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("payload.json", content)
        output.writestr(
            "checksums.sha256",
            hashlib.sha256(content).hexdigest() + "  payload.json\n")
    root = safe_extract_zip(archive, tmp_path / "extracted")
    assert verify_checksums(root) == {
        "payload.json": hashlib.sha256(content).hexdigest()}

    (root / "payload.json").write_text("changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksums(root)


@pytest.mark.parametrize("name", ["../outside.txt", "C:evil"])
def test_safe_extract_rejects_unsafe_paths(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(name, "bad")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        safe_extract_zip(archive, tmp_path / "out")


def test_safe_extract_rejects_symlink_and_duplicates(tmp_path):
    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(symlink, "w") as output:
        output.writestr(info, "target")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        safe_extract_zip(symlink, tmp_path / "symlink-out")

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as output:
        output.writestr("same", "one")
        output.writestr("same", "two")
    with pytest.raises(ValueError, match="Duplicate ZIP members"):
        safe_extract_zip(duplicate, tmp_path / "duplicate-out")


def test_match_page_is_class_aware_and_score_filtered():
    truth = [
        {"label": "text_region", "bbox_xyxy": [0, 0, 100, 100]},
        {"label": "heading", "bbox_xyxy": [0, 120, 100, 150]},
    ]
    predictions = [
        {"label": "text_region", "bbox_xyxy": [2, 2, 98, 98], "score": 0.9},
        {"label": "figure", "bbox_xyxy": [0, 0, 100, 100], "score": 0.8},
        {"label": "heading", "bbox_xyxy": [0, 120, 100, 150], "score": 0.1},
    ]
    result = match_page(truth, predictions, threshold=0.25)
    assert result["totals"] == {"tp": 1, "fp": 1, "fn": 1}
    assert result["per_class"]["text_region"] == {"tp": 1, "fp": 0, "fn": 0}
    assert result["per_class"]["heading"] == {"tp": 0, "fp": 0, "fn": 1}
    assert result["per_class"]["figure"] == {"tp": 0, "fp": 1, "fn": 0}


def test_render_comparison_creates_private_review_preview(tmp_path):
    source = tmp_path / "page.jpg"
    target = tmp_path / "comparison.jpg"
    Image.new("RGB", (200, 300), "white").save(source)
    render_comparison(
        source,
        [{"label": "text_region", "bbox_xyxy": [10, 20, 150, 250]}],
        [{"label": "text_region", "bbox_xyxy": [12, 18, 148, 252], "score": 0.8}],
        target,
    )
    with Image.open(target) as rendered:
        assert rendered.width == 412
        assert rendered.height == 328


def test_prediction_labels_prefer_names_and_accept_known_id_schemes():
    names = ["text_region", "heading"]
    assert resolve_prediction_labels([99], ["heading"], names) == ["heading"]
    assert resolve_prediction_labels([0, 1], None, names) == names
    assert resolve_prediction_labels([1, 2], None, names) == names
    with pytest.raises(ValueError, match="Ambiguous predicted class ID scheme"):
        resolve_prediction_labels([1], None, names)
    with pytest.raises(ValueError, match="Invalid zero-based predicted class IDs"):
        resolve_prediction_labels([0, 99], None, names)

