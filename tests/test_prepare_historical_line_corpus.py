import hashlib
import io
import json

from PIL import Image
import pytest

from training.prepare_historical_line_corpus import build, private_use_count


def image_bytes(color=255):
    stream = io.BytesIO()
    Image.new("L", (40, 20), color=color).save(stream, format="JPEG")
    return stream.getvalue()


def row(identifier, split, collection, text, data):
    return {
        "id": identifier,
        "split": split,
        "collection": collection,
        "page_id": identifier.split("__r")[0],
        "text": text,
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "license": "CC-BY-3.0",
        "file_name": f"images/{identifier}.jpg",
    }


def test_build_preserves_historical_spelling_and_quarantines_pua(tmp_path):
    train_image = image_bytes(250)
    validation_image = image_bytes(245)
    rows = {
        "train": [row("train__r000", "train", "train-c", "stárá\nſkładnia \ueada", train_image)],
        "validation": [row("val__r000", "validation", "val-c", "ɇzyk", validation_image)],
        "test": [row("test__r000", "test", "test-c", "test", image_bytes(240))],
    }
    payloads = {"train__r000.jpg": train_image, "val__r000.jpg": validation_image}

    def opener(url, timeout):
        return io.BytesIO(payloads[url.rsplit("/", 1)[-1]])

    def segmenter(image, follow_lines):
        assert follow_lines is False
        count = 2 if image.getpixel((0, 0))[0] == 250 else 1
        return {"boxes": [[0, i * 10, 40, (i + 1) * 10] for i in range(count)]}

    output = tmp_path / "corpus"
    report = build(rows, {"holdout-c"}, output, opener=opener, segmenter=segmenter)

    assert report["stats"]["train"]["accepted_lines"] == 1
    assert report["stats"]["train"]["quarantined_private_use_character_lines"] == 1
    assert (output / "train" / "train__r000__line000.txt").read_text(encoding="utf-8").strip() == "stárá"
    assert (output / "validation" / "val__r000__line000.txt").read_text(encoding="utf-8").strip() == "ɇzyk"
    quarantine = [json.loads(line) for line in (output / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()]
    assert quarantine[0]["reason"] == "private-use character"
    assert private_use_count("a\ueada") == 1


def test_build_rejects_collection_overlap(tmp_path):
    data = image_bytes()
    rows = {
        "train": [row("a__r000", "train", "same", "a", data)],
        "validation": [row("b__r000", "validation", "same", "b", data)],
        "test": [row("c__r000", "test", "test", "c", data)],
    }
    with pytest.raises(ValueError, match="Collection overlap"):
        build(rows, {"holdout"}, tmp_path / "corpus")


def test_build_rejects_source_checksum_mismatch(tmp_path):
    expected = image_bytes(255)
    actual = image_bytes(240)
    rows = {
        "train": [row("a__r000", "train", "train", "a", expected)],
        "validation": [row("b__r000", "validation", "validation", "b", actual)],
        "test": [row("c__r000", "test", "test", "c", image_bytes(230))],
    }

    def opener(url, timeout):
        return io.BytesIO(actual)

    with pytest.raises(ValueError, match="Source checksum mismatch"):
        build(rows, {"holdout"}, tmp_path / "corpus", opener=opener)
