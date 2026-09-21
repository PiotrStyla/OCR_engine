import json

import pytest
from PIL import Image

from training.prepare_ocr_images import prepare
from training.stage_impact_benchmark import digest


def make_manifest(tmp_path, large_second=False):
    path = tmp_path / 'disguised.jpg'
    first = Image.new('L', (10, 8), 127)
    second = Image.new('L', (20, 16) if large_second else (5, 4), 127)
    first.save(path, format='TIFF', save_all=True, append_images=[second])
    manifest = tmp_path / 'input.jsonl'
    manifest.write_text(json.dumps({'id': 'page', 'image': path.name,
                                   'sha256': digest(path), 'text': 'reference'}))
    return manifest


def test_tiff_disguised_as_jpeg_roundtrips_pixels(tmp_path):
    manifest = make_manifest(tmp_path)
    source_hash = digest(manifest)
    output = tmp_path / 'png'
    report = prepare(manifest, output)
    assert report['results'][0]['source_format'] == 'TIFF'
    assert report['results'][0]['source_frames'] == 2
    row = json.loads((output / 'manifest.jsonl').read_text())
    with Image.open(output / row['image']) as image:
        assert image.format == 'PNG'
        assert image.size == (10, 8)
        assert image.tobytes() == bytes([127]) * 80
    assert row['text'] == 'reference'
    assert digest(manifest) == source_hash
    with pytest.raises(FileExistsError):
        prepare(manifest, output)


def test_rejects_larger_later_frame(tmp_path):
    with pytest.raises(ValueError, match='largest raster'):
        prepare(make_manifest(tmp_path, large_second=True), tmp_path / 'png')


def test_rejects_modified_source(tmp_path):
    manifest = make_manifest(tmp_path)
    (tmp_path / 'disguised.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        prepare(manifest, tmp_path / 'png')
    assert not (tmp_path / 'png').exists()
