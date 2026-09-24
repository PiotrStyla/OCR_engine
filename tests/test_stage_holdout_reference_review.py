import io
import json

from PIL import Image

from training.geometry_holdout_runner import digest
from training.stage_holdout_reference_review import stage


def png_bytes():
    stream = io.BytesIO()
    Image.new("L", (8, 4), color=255).save(stream, format="PNG")
    return stream.getvalue()


def test_stage_downloads_verified_image_and_builds_review(tmp_path):
    source = png_bytes()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "id": "sample",
                        "source_path": "sample.png",
                        "image_sha256": digest(source),
                        "text": "stárá ſkładnia \ueada",
                        "reference_private_use_count": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def opener(url, timeout):
        assert url.endswith("sample.png")
        assert timeout == 120
        return io.BytesIO(source)

    output = tmp_path / "review"
    report = stage(manifest, output, opener=opener)

    assert report["regions"] == 1
    assert report["private_use_characters"] == 1
    assert report["review"]["issues"] == 1
    assert "stárá ſkładnia" in (output / "manifest.jsonl").read_text(encoding="utf-8")
    assert (output / "review" / "index.html").is_file()


def test_stage_rejects_source_checksum_mismatch(tmp_path):
    source = png_bytes()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "id": "sample",
                        "source_path": "sample.png",
                        "image_sha256": "0" * 64,
                        "text": "tekst",
                        "reference_private_use_count": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def opener(url, timeout):
        return io.BytesIO(source)

    try:
        stage(manifest, tmp_path / "review", opener=opener)
    except ValueError as error:
        assert "checksum mismatch" in str(error).lower()
    else:
        raise AssertionError("Expected checksum mismatch")
