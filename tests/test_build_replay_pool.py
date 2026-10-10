import json

import numpy as np
from PIL import Image

from training import build_replay_pool as pool


def gray_with_band():
    gray = np.full((300, 500), 210, dtype=np.uint8)
    gray[100:120, 50:450] = 40      # line ink: rows 100..119
    return gray


def test_cut_crop_hugs_the_ink_band_not_the_box():
    gray = gray_with_band()
    word_boxes = [[50, 80, 450, 140]]                 # padded box taller than the ink
    crop = pool.cut_crop(gray, word_boxes, margin=1)
    assert crop == [50, 99, 450, 121], "cut follows the ink band with a 1 px margin"


def test_cut_crop_falls_back_to_boxes_without_ink():
    gray = gray_with_band()
    word_boxes = [[50, 200, 450, 240]]                # no ink inside
    crop = pool.cut_crop(gray, word_boxes, margin=1)
    assert crop == [50, 199, 450, 241]


def test_render_pool_crop_verifies_ink_cleanliness(tmp_path):
    gray = gray_with_band()
    frame = Image.fromarray(gray, mode="L").convert("RGB")
    clean = [50, 100, 450, 120]
    neighbor_clean = [[50, 140, 450, 180]]
    crop, inside = pool.render_pool_crop(frame, gray, clean, neighbor_clean)
    assert inside == 0 and crop.size == (400, 20)
    neighbor_dirty = [[50, 110, 450, 150]]            # ink rows 110..119 enter the crop
    _, inside = pool.render_pool_crop(frame, gray, clean, neighbor_dirty)
    assert inside == 10 * 400, "neighbor ink inside the crop is counted exactly"


def test_run_assembles_verified_pool(tmp_path):
    import json as _json
    audit_root = tmp_path / "audit"
    directory = audit_root / "evidence" / "native-pages" / "page-1"
    directory.mkdir(parents=True)
    gray = gray_with_band()
    Image.fromarray(gray, mode="L").save(directory / "page.png")
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = [header]
    for line_no, top in ((1, 100), (2, 140)):
        for word_no, left in enumerate((50, 250), start=1):
            words = {1: ("ala ma kota", "oraz pies domowy"), 2: ("zosia czyta", "gruba ksiazke")}
            rows.append("\t".join(str(v) for v in (5, 1, 1, 1, line_no, word_no, left, top, 200, 20, 95,
                                                   words[line_no][word_no - 1])))
    (directory / "tesseract.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    package = audit_root / "evidence" / "source-package"
    package.mkdir(parents=True)
    (package / "ref.txt").write_text("ala ma kota oraz pies domowy zosia czyta gruba ksiazke", encoding="utf-8")
    (audit_root / "evidence" / "source-manifest.jsonl").write_text(_json.dumps({
        "id": "page-1", "work_family": "w", "split": "replay-candidate",
        "reference_file": "ref.txt"}) + "\n", encoding="utf-8")
    (audit_root / "geometry-diagnostic.json").write_text(
        _json.dumps({"pages": [{"page_id": "page-1", "conflicts": []}]}), encoding="utf-8")
    geometry_root = tmp_path / "geometry"
    (geometry_root / "evidence").mkdir(parents=True)
    (geometry_root / "evidence" / "report.json").write_text(_json.dumps({"pages": []}), encoding="utf-8")
    out = tmp_path / "pool"
    pool.run(audit_root, geometry_root, out)
    manifest = (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest) == 2
    for line in manifest:
        row = _json.loads(line)
        assert row["ink_verified_clean"] and row["eligible_for_training"] is False
        assert row["cut_rule"] == "geometry-v2-accepted"
