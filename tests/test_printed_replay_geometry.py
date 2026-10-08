import math

import pytest
from PIL import Image

from training import printed_replay_geometry as geometry
from training.mine_printed_replay import exact_anchors, parse_tsv


HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def tsv_page(words_by_line):
    rows = [HEADER]
    for line_number, words in enumerate(words_by_line, start=1):
        for word_number, (text, left, top, width, height, conf) in enumerate(words, start=1):
            rows.append("\t".join(str(value) for value in
                                  (5, 1, 1, 1, line_number, word_number, left, top, width, height, conf, text)))
    return "\n".join(rows) + "\n"


def word(cx, cy, text, *, width=120, height=24, conf=95):
    cx, cy = int(round(cx)), int(round(cy))
    return (text, cx - width // 2, cy - height // 2, width, height, conf)


def tilted_page():
    """Two 3-word lines descending by slope 0.05; native extents overlap by 8 px."""
    first = ("Ala ma kota oraz psa w ogrodzie", ["Ala ma", "kota oraz", "psa w ogrodzie"])
    second = ("Zosia czyta gruba ksiazke w domu", ["Zosia czyta", "gruba ksiazke", "w domu"])
    words = []
    for (text, parts), base in ((first, 300.0), (second, 340.0)):
        words.append([word(200, base + 0.05 * 200, parts[0]),
                      word(440, base + 0.05 * 440, parts[1]),
                      word(680, base + 0.05 * 680, parts[2])])
    return words, first[0] + " " + second[0]


def test_parse_words_matches_frozen_parse_tsv():
    words, _ = tilted_page()
    text = tsv_page(words)
    lines = geometry.parse_words(text, 1000, 1000)
    assert geometry.strip_words(lines) == parse_tsv(text, 1000, 1000)
    assert [len(line["words"]) for line in lines] == [3, 3]


def test_rotation_mapping_matches_pillow():
    native = Image.new("RGB", (200, 200), "white")
    source = (150, 40)
    native.putpixel(source, (255, 0, 0))
    rotated = native.rotate(2.5, center=(100, 100), resample=Image.Resampling.BICUBIC)
    mapped = geometry.rotate_point(*source, 100, 100, 2.5)
    found = [(x, y) for x in range(200) for y in range(200) if rotated.getpixel((x, y))[0] > 200]
    assert min(math.dist(point, mapped) for point in found) <= 1.0


def test_affine_matches_rotate_point():
    matrix = geometry.affine(2.5, 100, 100)
    for point in ((10, 10), (150, 40), (100, 100), (0, 200)):
        mapped = geometry.rotate_point(*point, 100, 100, 2.5)
        assert geometry.apply_affine(matrix, *point) == pytest.approx(mapped, abs=1e-9)


def test_native_extent_conflict_is_resolved_by_deskew():
    words, reference = tilted_page()
    lines = geometry.parse_words(tsv_page(words), 1000, 1000)
    native = Image.new("RGB", (1000, 1000), "white")
    baseline = geometry.run_frame(native, lines, reference, degrees=0.0)
    assert baseline["gated_by_overlap"] and baseline["matched"] == []
    assert [c["vertical_extent_overlap_px"] for c in baseline["conflicts"]] == [8]
    assert baseline["conflicts"][0]["source_order_conflict"] is False
    skew = geometry.page_skew_degrees(lines)
    assert skew == pytest.approx(math.degrees(math.atan(0.05)))
    deskewed = geometry.run_frame(native, lines, reference, degrees=skew)
    assert deskewed["conflicts"] == [] and deskewed["matched"]
    assert [row["line_index"] for row in deskewed["matched"]] == [0, 1]


def test_native_frame_keeps_frozen_gate_reasons():
    words, reference = tilted_page()
    words[0] = [(*words[0][0][:5], 80), *words[0][1:]]  # one low-confidence word
    lines = geometry.parse_words(tsv_page(words), 1000, 1000)
    native = Image.new("RGB", (1000, 1000), "white")
    result = geometry.run_frame(native, lines, reference, degrees=0.0)
    _, rejected, _ = exact_anchors(lines, reference, 1000, 1000)
    assert [r["reason"] for r in result["rejected"]] == [r["reason"] for r in rejected]
    assert result["rejected"][0]["reason"] == "word-confidence-below-90"


def test_neighbor_intrusion_reports_clipped_rects():
    crop = [100, 100, 300, 140]
    line = {"line_index": 0, "words": [{"bbox": [100, 100, 300, 140]}]}
    neighbor = {"line_index": 1, "words": [{"bbox": [260, 130, 400, 170]}]}
    inside = geometry.neighbor_intrusion(crop, line, [line, neighbor])
    assert inside == {1: {"px2": 40 * 10, "rects": [[260, 130, 300, 140]]}}
    assert geometry.neighbor_intrusion(crop, line, [line]) == {}


def test_page_skew_uses_median_and_clamps():
    def lines(slope):
        return [{"words": [{"bbox": [x, int(100 + slope * x), x + 40, int(120 + slope * x)]}
                           for x in (10, 210, 410)]}]
    assert geometry.page_skew_degrees(lines(0.05)) == pytest.approx(math.degrees(math.atan(0.05)))
    assert geometry.page_skew_degrees(lines(0.5)) == 0.0
    assert geometry.page_skew_degrees([{"words": [{"bbox": [0, 0, 10, 10]}]}]) == 0.0


def test_render_pair_roundtrip_and_pixels():
    from pathlib import Path
    import tempfile
    native = Image.new("RGB", (400, 400), "white")
    for x in range(400):
        native.putpixel((x, 200), (0, 0, 0))
    words, _ = tilted_page()
    lines = geometry.parse_words(tsv_page(words), 1000, 1000)
    framed = geometry.frame_lines(lines, 500, 500, 2.0)[0]
    with tempfile.TemporaryDirectory() as tmp:
        rotated = native.rotate(2.0, center=(200, 200), resample=Image.Resampling.BICUBIC)
        rendered = geometry.render_pair(rotated, {**lines[0], "text": "text"}, framed, 2.0, 400, 400,
                                        image_dir=Path(tmp), text_dir=Path(tmp), identifier="sample")
        with Image.open(rendered["image"]) as crop:
            assert crop.tobytes() == rotated.crop(rendered["crop_bbox"]).tobytes()
    assert rendered["roundtrip_error_px"] == pytest.approx(0.0, abs=1e-9)
    expected_corners = [rendered["crop_bbox"][:2], rendered["crop_bbox"][2:]]
    for actual, expected in zip((rendered["native_corners"][0], rendered["native_corners"][2]), expected_corners):
        assert geometry.apply_affine(rendered["affine_native_to_crop"], *actual) == pytest.approx(expected, abs=1e-3)
