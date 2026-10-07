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
