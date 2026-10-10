import numpy as np
from PIL import Image

from training import printed_replay_crop_review as review


def gray_page():
    gray = np.full((300, 500), 210, dtype=np.uint8)
    gray[80:100, 50:450] = 40        # accepted line ink
    gray[110:130, 50:450] = 40       # neighbor line ink
    return gray


def test_trace_pixels_measure_depth_into_the_crop():
    gray = gray_page()
    crop = [40, 75, 460, 115]        # includes 5 rows of the neighbor ink (110..114)
    pixels = review.trace_pixels(gray, crop, [[50, 110, 450, 130]])
    assert len(pixels) == 5 * 400, "every neighbor ink pixel inside the crop is reported"
    assert max(depth for _, _, depth in pixels) == 4, "deepest trace sits 4 px inside the crop"
    assert all(x == min(x, 459) for x, _, _ in pixels)


def test_recommendation_bands():
    assert review.recommend([]) == ("no-trace", 0)
    assert review.recommend([(10, 10, 2)])[0] == "trim-2px"
    assert review.recommend([(10, 10, 8)])[0] == "trim-to-depth-or-accept"
    assert review.recommend([(10, 10, 9)])[0] == "review-closely"


def test_render_case_marks_traces(tmp_path):
    gray = gray_page()
    source = tmp_path / "crop.png"
    Image.fromarray(gray, mode="L").save(source)
    pixels = [(60, 112, 3), (61, 113, 2)]
    name = review.render_case(source, pixels, "case | 2 px", tmp_path / "sheet.png")
    assert name == "sheet.png" and (tmp_path / name).is_file()
    with Image.open(tmp_path / name) as sheet:
        assert sheet.mode == "RGB"
        assert sheet.getpixel((60, 112)) == (255, 0, 0), "trace pixel marked in red"
        assert sheet.getpixel((61, 113)) == (255, 0, 0)
        assert sheet.getpixel((200, 250)) == (210, 210, 210), "plain background untouched"


def test_trim_box_excludes_traces_per_side():
    crop = [40, 75, 460, 115]
    pixels = [(60, 112, 3), (61, 113, 2)]     # traces near the bottom edge
    trimmed, sides = review.trim_box(pixels, crop)
    assert sides == {"left": 0, "top": 0, "right": 0, "bottom": 3}
    assert trimmed == [40, 75, 460, 112]
    assert all(not (trimmed[0] <= x < trimmed[2] and trimmed[1] <= y < trimmed[3])
               for x, y, _ in pixels)
    edge = [(50, 80, 20), (300, 80, 25)]
    trimmed, sides = review.trim_box(edge, crop)
    assert sides["top"] == 6 and sides["bottom"] == 0
