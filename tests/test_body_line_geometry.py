from PIL import Image, ImageDraw
from training.body_line_geometry import detect_lines


def synthetic():
    im = Image.new('L', (600, 240), 255)
    draw = ImageDraw.Draw(im)
    for y, xs in [(45, range(30, 450, 35)), (135, range(280, 420, 35))]:
        for x in xs:
            draw.rectangle((x, y, x + 16, y + 35), fill=0)
    draw.rectangle((68, 29, 73, 35), fill=0)
    return im


def test_blank_and_short_line_and_accent():
    assert detect_lines(Image.new('L', (600, 240), 255))['boxes'] == []
    result = detect_lines(synthetic())
    assert len(result['boxes']) == 2
    assert result['boxes'][0][1] <= 29
    assert result['boxes'][0][3] > 80
    assert result['boxes'][1][0] > 250
    assert all(0 <= x <= 1 for x in result['foreign_ink_fraction'])


def test_source_pixels_unchanged_and_deterministic():
    image = synthetic()
    original = image.tobytes()
    assert detect_lines(image) == detect_lines(image)
    assert image.tobytes() == original


def test_edge_fragment_not_extra_line():
    image = synthetic()
    draw = ImageDraw.Draw(image)
    for x in range(30, 400, 35):
        draw.rectangle((x, 222, x + 15, 239), fill=0)
    result = detect_lines(image)
    assert len(result['boxes']) == 2
