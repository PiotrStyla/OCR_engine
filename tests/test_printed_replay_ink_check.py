import numpy as np
from PIL import Image

from training import printed_replay_ink_check as ink


def page(tmp_path, gray):
    path = tmp_path / "page.png"
    Image.fromarray(gray.astype(np.uint8), mode="L").save(path)
    return path


def blank(width=500, height=300, level=210):
    return np.full((height, width), level, dtype=np.uint8)


def draw_band(gray, x0, x1, y0, y1, level=40):
    gray[y0:y1, x0:x1] = level
    return gray


def test_ink_gap_separated_bands():
    gray = draw_band(blank(), 50, 450, 100, 120)
    gray = draw_band(gray, 50, 450, 130, 150)
    mask = ink.ink_mask(gray, [0, 0, 500, 300])
    result = ink.ink_gap(mask, [[50, 100, 450, 120]], [[50, 130, 450, 150]])
    assert result["verdict"] == "ink-separated"
    assert result["ink_min_gap_px"] == 10 and result["columns_in_contact"] == 0


def test_ink_gap_interleaved_bands():
    gray = draw_band(blank(), 50, 450, 100, 130)
    gray = draw_band(gray, 200, 250, 125, 155)
    mask = ink.ink_mask(gray, [0, 0, 500, 300])
    result = ink.ink_gap(mask, [[50, 100, 450, 130]], [[200, 125, 250, 155]])
    assert result["verdict"] == "ink-contact"
    assert result["ink_min_gap_px"] == -5 and result["columns_in_contact"] == 50


def test_crop_intrusion_counts_neighbor_ink():
    gray = draw_band(blank(), 100, 300, 80, 100)   # accepted line ink
    gray = draw_band(gray, 250, 350, 110, 120)     # neighbor ink entering the crop
    threshold = float((np.median(gray) + int(gray.min())) / 2.0)
    inside = ink.crop_intrusion(gray, [100, 75, 300, 115], [[250, 110, 350, 120]], threshold)
    assert inside == 50 * 5, "neighbor ink inside the crop rectangle is counted exactly"
    outside = ink.crop_intrusion(gray, [100, 75, 300, 105], [[250, 110, 350, 120]], threshold)
    assert outside == 0


def test_ink_mask_tracks_shaded_background():
    gray = blank(level=200)
    for x in range(500):
        gray[:, x] = 140 + (x // 5)          # shaded background 140..239
    draw_band(gray, 100, 400, 100, 120, level=60)   # ink darker than any background
    mask = ink.ink_mask(gray, [0, 0, 500, 300])
    rows = np.unique(np.nonzero(mask)[0])
    assert rows.min() >= 100 and rows.max() < 120, "threshold must isolate ink, not the shading"


def test_run_writes_reports(tmp_path):
    import json
    from training.full_page_pilot import read_rows
    gray = draw_band(blank(), 50, 450, 100, 120)
    gray = draw_band(gray, 50, 450, 130, 150)
    ink_root = tmp_path / "audit"
    directory = ink_root / "evidence" / "native-pages" / "page-1"
    directory.mkdir(parents=True)
    page(directory, gray)
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = [header]
    for line_no, top in ((1, 100), (2, 130)):
        for word_no, left in enumerate((50, 250), start=1):
            rows.append("\t".join(str(v) for v in (5, 1, 1, 1, line_no, word_no, left, top, 200, 20, 95,
                                                   f"s{line_no}{word_no}")))
    (directory / "tesseract.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    geometry_root = tmp_path / "geometry"
    (geometry_root / "evidence").mkdir(parents=True)
    (geometry_root / "evidence" / "report.json").write_text(json.dumps({
        "pages": [{"page_id": "page-1", "conflicts": [{"line_indices": [0, 1], "slanted_overlap_px": 2.0}]}]}),
        encoding="utf-8")
    (geometry_root / "evidence" / "pairs.jsonl").write_text(json.dumps({
        "id": "page-1-line-000", "page_id": "page-1", "intruding_neighbor_lines": [1],
        "crop_bbox": [40, 95, 460, 125], "neighbor_word_box_overlap_px2": 100}) + "\n", encoding="utf-8")
    out = tmp_path / "ink-out"
    report = ink.run(ink_root, geometry_root, out)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["pairs_checked"] == 1 and data["crops_checked"] == 1
    assert data["verdicts"]["ink-separated"] == 1
    assert data["cases"][0]["ink_min_gap_px"] == 10
    assert data["crop_cases"][0]["neighbor_ink_pixels_in_crop"] == 0, "neighbor ink sits below the crop"
    assert read_rows(out / "ink-pairs.jsonl")
