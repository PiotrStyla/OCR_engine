from PIL import Image, ImageDraw
from training.body_line_geometry import detect_lines, crop_line_band


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


def test_experimental_terminal_colon_and_accent():
    image = synthetic()
    draw = ImageDraw.Draw(image)
    draw.rectangle((452, 52, 456, 57), fill=0)
    draw.rectangle((452, 69, 456, 74), fill=0)
    result = detect_lines(image, follow_lines=True)
    assert len(result['boxes']) == 2
    box = result['boxes'][0]
    assert box[2] > 456
    crop = crop_line_band(image, box, result['line_bands'][0])
    for x, y in [(454, 54), (454, 71), (70, 31)]:
        assert crop.getpixel((x - box[0], y - box[1])) == (0, 0, 0)


def test_slanted_lines_band_preserves_target_and_removes_neighbor():
    image = Image.new('L', (600, 240), 255)
    draw = ImageDraw.Draw(image)
    target = []
    for base in [40, 93]:
        for x in range(30, 500, 35):
            y = base + round(.12 * x)
            draw.rectangle((x, y, x + 16, y + 30), fill=0)
            if base == 40:
                target.append((x + 8, y + 15))
    result = detect_lines(image, follow_lines=True)
    assert len(result['boxes']) == 2
    box = result['boxes'][0]
    crop = crop_line_band(image, box, result['line_bands'][0])
    for x, y in target:
        assert crop.getpixel((x - box[0], y - box[1])) == (0, 0, 0)
    assert crop.getpixel((38 - box[0], 105 - box[1])) == (255, 255, 255)


def test_invalid_band_rejected():
    import pytest
    with pytest.raises(ValueError, match='Invalid line band'):
        crop_line_band(synthetic(), [0, 0, 10, 10], {'top': [0], 'bottom': [10]})
