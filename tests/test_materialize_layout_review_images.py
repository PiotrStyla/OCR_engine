import hashlib
import json

import pytest

from training.materialize_layout_review_images import materialize


def _inputs(tmp_path, *, file_name="images/p1.jpg", image_hash=None):
    source = tmp_path / "source.jpg"
    source.write_bytes(b"image-fixture")
    image_hash = image_hash or hashlib.sha256(source.read_bytes()).hexdigest()
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps([{
        "id": "p1", "file_name": file_name, "image_sha256": image_hash,
    }]), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"dataset": {
        "repo": "owner/data", "revision": "a" * 40, "selected_pages": 1,
    }}), encoding="utf-8")
    return source, selection, config


def test_materializes_verified_images_and_manifest(tmp_path):
    source, selection, config = _inputs(tmp_path)
    calls = []

    def downloader(repo, path, revision):
        calls.append((repo, path, revision))
        return source

    output = tmp_path / "images"
    report = materialize(selection, config, output, downloader)
    assert report["images"] == 1
    assert calls == [("owner/data", "pages/train/images/p1.jpg", "a" * 40)]
    assert (output / "p1.jpg").read_bytes() == b"image-fixture"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["images"][0]["page_id"] == "p1"
    assert manifest["selection_sha256"]
    with pytest.raises(FileExistsError):
        materialize(selection, config, output, downloader)


@pytest.mark.parametrize("file_name", ["../p1.jpg", "p1.jpg", "/images/p1.jpg"])
def test_rejects_unsafe_dataset_paths(tmp_path, file_name):
    source, selection, config = _inputs(tmp_path, file_name=file_name)
    output = tmp_path / "images"
    with pytest.raises(ValueError, match="Unsafe dataset image path"):
        materialize(selection, config, output, lambda *_: source)
    assert not output.exists()


def test_checksum_failure_removes_partial_output(tmp_path):
    source, selection, config = _inputs(tmp_path, image_hash="0" * 64)
    output = tmp_path / "images"
    with pytest.raises(ValueError, match="checksum mismatch"):
        materialize(selection, config, output, lambda *_: source)
    assert not output.exists()
