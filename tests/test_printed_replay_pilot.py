import ast
import hashlib
import json
from types import SimpleNamespace
import zipfile

import pytest

from training.mine_printed_replay import exact_anchors, normalize, parse_tsv
from training.prepare_printed_replay_sources import (
    WORKS, quality_level, reference_from_html, validate_works, verify_source_package)


def line(text, *, confidence=99, top=10):
    return {"text": text, "bbox": [10, top, 480, top + 10], "min_word_confidence": confidence}


def test_old_spelling_is_not_modernized():
    assert normalize("Po\u017f\u0142a\u0142  b\u0142\u00e1waty z\u030c") == "Po\u017f\u0142a\u0142 b\u0142\u00e1waty \u017e"
    reference = "Po\u017f\u0142a\u0142 w \u015bwietne b\u0142\u00e1waty na pocz\u0105tku."
    good, rejected, _ = exact_anchors([line(reference)], reference, 500, 700)
    assert len(good) == 1 and not rejected
    good, rejected, _ = exact_anchors([line(reference.replace("\u017f", "s").replace("\u00e1", "a"))], reference, 500, 700)
    assert not good and rejected[0]["reason"] == "no-unique-exact-source-anchor"


def test_no_fuzzy_or_hyphen_repair():
    good, rejected, _ = exact_anchors([line("Czy to krzyk w mo- rzu czy w niej?")],
        "Czy to krzyk w morzu czy w niej?", 500, 700)
    assert not good and len(rejected) == 1


def test_partial_word_and_ambiguous_anchor_rejected():
    for reference in ("Ala ma kotka oraz wielkie oko.", "Ala ma kotka i psa. Ala ma kotka i psa."):
        good, _, _ = exact_anchors([line("Ala ma kotka i psa.")], reference, 500, 700)
        assert not good


def test_partial_suffix_cannot_match():
    good, _, _ = exact_anchors([line("tam kotek na drodze.")], "Witam kotek na drodze.", 500, 700)
    assert not good


@pytest.mark.parametrize("confidence", [0, 89.9])
def test_confidence_gate(confidence):
    text = "To jest poprawna linia tekstu."
    good, rejected, _ = exact_anchors([line(text, confidence=confidence)], text, 500, 700)
    assert not good and rejected[0]["reason"] == "word-confidence-below-90"


def test_order_and_overlap_gate():
    a, b = "Pierwsza poprawna linia tekstu.", "Druga poprawna linia tekstu."
    for lines in ([line(b), line(a, top=30)], [line(a), line(b, top=15)]):
        good, rejected, _ = exact_anchors(lines, a + " " + b, 500, 700)
        assert not good and all(r["reason"] == "anchor-order-or-line-overlap" for r in rejected)


def test_low_coverage_page_not_admitted():
    text = "Jedyna poprawna linia tekstu."
    good, rejected, coverage = exact_anchors([line(text)], text + " inne" * 100, 500, 700)
    assert not good and coverage < 0.4
    assert rejected[0]["reason"] == "page-exact-anchor-coverage-below-40-percent"


def test_v2_preserves_exact_line_without_claiming_page_quality():
    text = "Jedyna poprawna linia tekstu."
    good, rejected, coverage = exact_anchors([line(text)], text + " inne" * 100,
                                            500, 700, page_min_coverage=0)
    assert len(good) == 1 and not rejected and coverage < 0.4
    assert good[0]["text"] == text


def test_v2_still_protects_confidence_and_order():
    text = "Jedyna poprawna linia tekstu."
    good, rejected, _ = exact_anchors([line(text, confidence=89)], text,
                                      500, 700, page_min_coverage=0)
    assert not good and rejected[0]["reason"] == "word-confidence-below-90"
    with pytest.raises(ValueError, match="threshold"):
        exact_anchors([], "", 500, 700, page_min_coverage=-1)


def test_tsv_words_grouped_by_native_line():
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    row = "5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t95.2\tPo\u017f\u0142a\u0142\n"
    result = parse_tsv(header + row, 100, 100)
    assert result[0]["text"] == "Po\u017f\u0142a\u0142" and result[0]["bbox"] == [10, 20, 40, 30]
    with pytest.raises(ValueError, match="geometry"):
        parse_tsv(header + row, 20, 20)
    quoted = parse_tsv(header + row.replace("Po\u017f\u0142a\u0142", '"Po\u017f\u0142a\u0142"'), 100, 100)
    assert quoted[0]["text"] == '"Po\u017f\u0142a\u0142"'


def test_reference_body_excludes_navigation():
    body = "Po\u017f\u0142a\u0142 b\u0142\u00e1waty. " * 12
    assert reference_from_html('<div>navigation</div><div class="pagetext"><p>' + body + "</p></div>") == body.strip()
    with pytest.raises(ValueError, match="exactly one"):
        reference_from_html("<p>no proofread body</p>")
    with pytest.raises(ValueError, match="Complex"):
        reference_from_html('<div class="pagetext"><table><tr><td>Text</td></tr></table></div>')


def test_quality_and_work_split_validation():
    assert quality_level('<noinclude><pagequality level="4" user="A" /></noinclude>Text') == 4
    validate_works(WORKS)
    with pytest.raises(ValueError):
        quality_level("Text without proofread status")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_works([WORKS[0], WORKS[0]])


def test_package_checksum_coverage(tmp_path):
    (tmp_path / "checksums.json").write_text("{}")
    (tmp_path / "unexpected.txt").write_text("payload")
    with pytest.raises(ValueError, match="coverage"):
        verify_source_package(tmp_path)


def test_v2_builder_pins_original_input_and_selects_new_protocol(tmp_path):
    import nbformat
    from training.build_printed_replay_colab import DATA_PATH, DATA_REVISION, build
    target = tmp_path / "v2.ipynb"
    build(target, "a" * 40, "v2")
    notebook = nbformat.read(target, 4)
    nbformat.validate(notebook)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    ast.parse(code)
    assert DATA_PATH in code and DATA_REVISION in code
    assert "'--protocol', 'v2'" in code
    assert "printed-replay-pilot-v2-" in code


def test_v2_packaging_contains_native_pages_and_no_promotion(tmp_path, monkeypatch):
    from PIL import Image
    from training import mine_printed_replay as module
    source = tmp_path / "source"
    source.mkdir()
    text = "Jedyna poprawna linia tekstu."
    (source / "reference.txt").write_text(text + " inne" * 100, encoding="utf-8")
    for name in ("checksums.json", "source-policy.json"):
        (source / name).write_text("{}")
    (source / "manifest.jsonl").write_text("")
    original = b"mock DjVu"
    rows = [{"id": "page-" + split, "split": split, "work_family": split, "scan_page": 1,
             "original_scan_url": "mock", "original_scan_sha1": hashlib.sha1(original).hexdigest(),
             "reference_file": "reference.txt", "reference_sha256": "mock", "source_revision": 1,
             "source_url": "mock", "scan_license": "public-domain", "transcription_license": "CC-BY-SA-4.0"}
            for split in ("replay-candidate", "replay-probe")]
    monkeypatch.setattr(module, "verify_source_package", lambda root: rows)
    monkeypatch.setattr(module, "download", lambda *args, **kwargs: original)
    monkeypatch.setattr(module.shutil, "which", lambda command: command)
    def process(command, **kwargs):
        if command[0] == "ddjvu" and "-format=ppm" in command:
            assert "-subsample=1" in command
            Image.new("RGB", (500, 700), "white").save(command[-1])
        if command[0] == "tesseract":
            assert "reference.txt" not in command
            header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            return SimpleNamespace(stdout=header + "5\t1\t1\t1\t1\t1\t10\t10\t470\t10\t99\t" + text + "\n", stderr="")
        return SimpleNamespace(stdout="", stderr="mock decoder version")
    monkeypatch.setattr(module.subprocess, "run", process)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "mock version")
    result = module.run(source, tmp_path / "output", protocol="v2")
    with zipfile.ZipFile(result) as archive:
        report = json.loads(archive.read("report.json"))
        assert report["schema"] == "slayer-printed-replay-mining-v2"
        assert report["candidates"] == 2 and not report["eligible_for_training"]
        for row in rows:
            assert "native-pages/" + row["id"] + "/page.png" in archive.namelist()
            assert "native-pages/" + row["id"] + "/tesseract.tsv" in archive.namelist()
        candidates = [json.loads(s) for s in archive.read("manifest.jsonl").decode().splitlines()]
        assert all(not r["eligible_for_training"] and not r["eligible_for_evaluation"] for r in candidates)
        assert all(r["line_index"] == 0 and r["text"] == text for r in candidates)
