import copy

from PIL import Image
import pytest

from training.audit_printed_replay_v2_evidence import verify_page
from training.full_page_pilot import digest


@pytest.fixture
def page_fixture(tmp_path):
    directory = tmp_path / "native-pages/page-1"
    directory.mkdir(parents=True)
    native = Image.new("RGB", (200, 100), "white")
    native.putpixel((25, 21), (0, 0, 0))
    native.save(directory / "page.png")
    text = "Trzy bardzo proste slowa"
    (directory / "tesseract.tsv").write_text(
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t20\t20\t30\t5\t95\tTrzy\n"
        "5\t1\t1\t1\t1\t2\t55\t20\t30\t5\t94\tbardzo\n"
        "5\t1\t1\t1\t1\t3\t90\t20\t30\t5\t93\tproste\n"
        "5\t1\t1\t1\t1\t4\t125\t20\t30\t5\t92\tslowa\n", encoding="utf-8")
    source_dir = tmp_path / "source-package"
    source_dir.mkdir()
    (source_dir / "reference.txt").write_text(text, encoding="utf-8")
    source = {"id": "page-1", "split": "replay-candidate", "reference_file": "reference.txt",
              "work_family": "work-1", "reference_sha256": digest(source_dir / "reference.txt"),
              "source_revision": 1, "source_url": "https://pl.wikisource.org/w/index.php?oldid=1",
              "original_scan_sha1": "a" * 40, "scan_license": "public-domain",
              "transcription_license": "CC-BY-SA-4.0"}
    pair_dir = tmp_path / "pairs/replay-candidate"
    pair_dir.mkdir(parents=True)
    native.crop((18, 19, 157, 26)).save(pair_dir / "page-1-line-000.png")
    (pair_dir / "page-1-line-000.txt").write_text(text, encoding="utf-8")
    row = {"id": "page-1-line-000", "page_id": "page-1", "line_index": 0,
           "tesseract_key": [1, 1, 1, 1], "text": text, "bbox": [20, 20, 155, 25],
           "crop_bbox": [18, 19, 157, 26], "native_page_dimensions": [200, 100],
           "native_page_sha256": digest(directory / "page.png"), "source_span_normalized": [0, len(text)],
           "min_word_confidence": 92.0, "image": "pairs/replay-candidate/page-1-line-000.png",
           "text_file": "pairs/replay-candidate/page-1-line-000.txt",
           "image_sha256": digest(pair_dir / "page-1-line-000.png"),
           "text_sha256": digest(pair_dir / "page-1-line-000.txt"), "original_scan_sha256": "b" * 64,
           "label_status": "source-validated-text-automatic-exact-line-anchor",
           "crop_review_status": "not-human-reviewed", "eligible_for_training": False,
           "eligible_for_evaluation": False}
    row.update({key: source[key] for key in ("split", "work_family", "reference_sha256", "source_revision",
                                            "source_url", "original_scan_sha1", "scan_license", "transcription_license")})
    page = {"native_dimensions": [200, 100], "split": "replay-candidate", "lines": 1,
            "candidates": 1, "anchor_coverage_before_page_gate": 1.0}
    return tmp_path, source, page, [row], []


def test_native_tsv_and_pixels_verified(page_fixture):
    result = verify_page(*page_fixture)
    assert result["lines"] == result["candidates"] == 1


@pytest.mark.parametrize("field,value", [
    ("crop_bbox", [18, 18, 157, 26]), ("text", "Modernized transcription"),
    ("source_span_normalized", [1, 24]), ("tesseract_key", [1, 2, 1, 1]),
    ("eligible_for_training", True), ("split", "replay-probe"),
    ("native_page_sha256", "c" * 64), ("min_word_confidence", 99),
])
def test_metadata_tampering_rejected(page_fixture, field, value):
    modified = copy.deepcopy(page_fixture)
    modified[3][0][field] = value
    with pytest.raises(ValueError, match="metadata"):
        verify_page(*modified)


def test_rehashed_changed_crop_rejected(page_fixture):
    root, source, page, predictions, quarantine = page_fixture
    path = root / predictions[0]["image"]
    with Image.open(path) as crop:
        crop.putpixel((7, 2), (255, 0, 0))
        crop.save(path)
    predictions[0]["image_sha256"] = digest(path)
    with pytest.raises(ValueError, match="pixels"):
        verify_page(root, source, page, predictions, quarantine)


def test_missing_candidate_rejected(page_fixture):
    root, source, page, _, quarantine = page_fixture
    with pytest.raises(ValueError, match="coverage"):
        verify_page(root, source, page, [], quarantine)


def test_tsv_confidence_and_quarantine_recomputed(page_fixture):
    root, source, page, predictions, _ = page_fixture
    path = root / "native-pages/page-1/tesseract.tsv"
    path.write_text(path.read_text(encoding="utf-8").replace("\t95\t", "\t85\t"), encoding="utf-8")
    with pytest.raises(ValueError, match="counts"):
        verify_page(root, source, page, predictions, [])
    page.update(candidates=0, anchor_coverage_before_page_gate=0)
    rejected = {"page_id": "page-1", "line_index": 0, "key": [1, 1, 1, 1],
                "text": predictions[0]["text"], "bbox": predictions[0]["bbox"],
                "min_word_confidence": 85.0, "reason": "word-confidence-below-90"}
    assert verify_page(root, source, page, [], [rejected])["candidates"] == 0
    rejected["reason"] = "anchor-too-short"
    with pytest.raises(ValueError, match="Quarantine"):
        verify_page(root, source, page, [], [rejected])
