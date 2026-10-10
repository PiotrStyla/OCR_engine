import math

import pytest
from PIL import Image

from training import printed_replay_geometry as v1
from training import printed_replay_geometry_v2 as v2


def word_box(x0, y0, x1, y1):
    return {"bbox": [x0, y0, x1, y1]}


def anchor(index, words, span):
    return {"line_index": index, "words": words, "source_span": span,
            "bbox": [min(w["bbox"][0] for w in words), min(w["bbox"][1] for w in words),
                     max(w["bbox"][2] for w in words), max(w["bbox"][3] for w in words)]}


def test_projection_matches_frozen_predicate_when_flat():
    a = anchor(0, [word_box(100, 100, 300, 124)], (0, 20))
    b = anchor(1, [word_box(100, 130, 300, 154)], (21, 40))
    conflict, detail = v2.pair_conflict(a, b, 0.0)
    assert not conflict and detail["slanted_overlap_px"] == 0
    overlapping = anchor(1, [word_box(100, 118, 300, 142)], (21, 40))
    conflict, detail = v2.pair_conflict(a, overlapping, 0.0)
    assert conflict and detail["slanted_overlap_px"] == pytest.approx(6.0)


def test_slanted_lines_resolve_aabb_union_conflict():
    # Two lines tilted by slope 0.05 whose axis-aligned unions overlap (8 px)
    # while their extents along the baseline normal are clearly separated.
    theta = math.degrees(math.atan(0.05))
    words_a = [word_box(140 + i * 240, 298 + int(0.05 * (140 + i * 240)),
                        260 + i * 240, 322 + int(0.05 * (140 + i * 240))) for i in range(3)]
    words_b = [word_box(140 + i * 240, 338 + int(0.05 * (140 + i * 240)),
                        260 + i * 240, 362 + int(0.05 * (140 + i * 240))) for i in range(3)]
    a = anchor(0, words_a, (0, 30))
    b = anchor(1, words_b, (31, 60))
    assert a["bbox"][3] > b["bbox"][1], "fixture must reproduce the native union conflict"
    conflict, _ = v2.pair_conflict(a, b, theta)
    assert not conflict


def test_flat_page_ignores_rounding_inflation():
    # Boxes separated by 1 px: AABB rotation plus outward rounding (V1) inflates
    # the extents into a conflict, the exact projection does not.
    lines = [[word_box(100, 100, 400, 122)], [word_box(100, 123, 400, 145)]]
    theta = 0.05
    framed = v1.frame_lines([{"words": w, "bbox": [100, 100, 400, 122]} for w in lines],
                            250, 123, theta)
    assert framed[0]["bbox"][3] > framed[1]["bbox"][1], "V1 framing must show the inflated conflict"
    a = anchor(0, lines[0], (0, 10))
    b = anchor(1, lines[1], (11, 20))
    conflict, _ = v2.pair_conflict(a, b, theta)
    assert not conflict


def test_true_overlap_and_order_conflicts_still_fire():
    a = anchor(0, [word_box(100, 100, 300, 124)], (0, 20))
    overlap = anchor(1, [word_box(100, 118, 300, 142)], (21, 40))
    conflict, _ = v2.pair_conflict(a, overlap, 2.0)
    assert conflict
    reversed_span = anchor(1, [word_box(100, 130, 300, 154)], (10, 40))
    conflict, detail = v2.pair_conflict(a, reversed_span, 0.0)
    assert conflict and detail["source_order_conflict"] is True


def test_shared_axis_prevents_frame_mismatch_noise():
    # Two lines with a 4 px gap along y, but line B's slope fit is off by ~4.2
    # degrees. Testing the pair in the lines' own frames couples x into the
    # separation and invents a sub-pixel-to-few-px overlap (0.46 px here — the
    # same scale as the 0.3-1.3 px false conflicts seen on real pages); the
    # shared axis measures the real gap.
    words_a = [word_box(x, 88, x + 60, 112) for x in (100, 450, 800)]
    words_b = [word_box(x, y - 12, x + 60, y + 12) for x, y in ((100, 128), (450, 154), (800, 180))]
    a = anchor(0, words_a, (0, 30))
    b = anchor(1, words_b, (31, 60))
    theta_b = v1.line_slope_degrees(words_b)
    assert abs(theta_b - 4.25) < 0.2, "fixture must fit a tilted second line"
    own_frames = max(v2.frame_overlap(a["words"], b["words"], 0.0),
                     v2.frame_overlap(a["words"], b["words"], theta_b))
    assert own_frames > 0.4, "mismatched frames must show the false overlap"
    conflict, detail = v2.pair_conflict(a, b, 0.0)
    assert not conflict and detail["slanted_overlap_px"] == 0


def test_flat_axis_snapping_keeps_frozen_touch_semantics():
    # Exactly-touching boxes are allowed by the frozen gate (bottom > top, not
    # >=). A noise-floor axis tilt would mix x in and invent an overlap; axes
    # below the estimator noise floor snap to flat, preserving that semantics.
    a = anchor(0, [word_box(100, 100, 800, 124)], (0, 30))
    b = anchor(1, [word_box(150, 124, 850, 148)], (31, 60))
    assert v2.frame_overlap(a["words"], b["words"], 0.0) == 0, "touching boxes: no conflict, no float dust"
    assert v2.frame_overlap(a["words"], b["words"], -0.019) > 0, "noise tilt must perturb touching boxes"
    conflict, _ = v2.pair_conflict(a, b, 0.0)
    assert not conflict
    # Snap rule: sub-noise axes become flat.
    assert abs(-0.019) < v2.FLAT_AXIS_EPS_DEGREES


def test_gate_keeps_frozen_all_or_nothing_semantics():
    words_a, words_b = [word_box(100, 100, 300, 124)], [word_box(100, 130, 300, 154)]
    clean = [anchor(0, words_a, (0, 20)), anchor(1, words_b, (21, 40))]
    admitted, details = v2.gate(clean, 0.0)
    assert admitted and details == []
    dirty = [anchor(0, words_a, (0, 20)),
             anchor(1, [word_box(100, 118, 300, 142)], (21, 40)),
             anchor(2, [word_box(100, 150, 300, 174)], (41, 60))]
    admitted, details = v2.gate(dirty, 0.0)
    assert not admitted and details[0]["line_indices"] == [0, 1]


def test_run_page_reuses_frozen_text_gates(tmp_path):
    from training.printed_replay_geometry import parse_words
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = [header]
    texts = ["Ala ma kota oraz psa w ogrodzie", "Zosia czyta gruba ksiazke w domu"]
    for line_no, text in enumerate(texts, start=1):
        for word_no, part in enumerate(text.split(), start=1):
            left = 100 + 40 * word_no
            rows.append("\t".join(str(v) for v in (5, 1, 1, 1, line_no, word_no, left, 300 + 40 * line_no,
                                                   30, 24, 95, part)))
    tsv = "\n".join(rows) + "\n"
    native = Image.new("RGB", (1000, 1000), "white")
    lines = parse_words(tsv, 1000, 1000)
    outcome = v2.run_page(native, lines, " ".join(texts))
    assert outcome["geometry_candidates"] == 2 and outcome["accepted_anchors"] == 2
    assert outcome["rejected_reasons"] == {} and outcome["conflicts"] == []
    assert [row["text"] for row in outcome["accepted"]] == texts
