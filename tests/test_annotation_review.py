import hashlib
import json

import pytest
from PIL import Image

from training.build_annotation_review import build, issues


def fixture(tmp_path):
    image = tmp_path / 'page.png'
    Image.new('L', (12, 16), 255).save(image)
    manifest = tmp_path / 'manifest.jsonl'
    row = {'id': 'PAGE__1', 'image': image.name,
           'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
           'text': '</script><script>alert(1)</script>\ufffd\ue000'}
    manifest.write_text(json.dumps(row), encoding='utf-8')
    return manifest


def test_issues_use_browser_utf16_offsets():
    found = issues('\U0001f600a\ufffd\U000f0000')
    assert [(i['offset'], i['length'], i['kind']) for i in found] == [
        (3, 1, 'replacement'), (4, 2, 'private')]


def test_build_is_offline_and_script_safe(tmp_path):
    manifest = fixture(tmp_path)
    original = manifest.read_bytes()
    output = tmp_path / 'review'
    result = build(manifest, output)
    html = (output / 'index.html').read_text(encoding='utf-8')
    assert '</script><script>alert(1)</script>' not in html
    assert '\\u003c/script>' in html
    assert result['issues'] == 2 and result['flagged_pages'] == 1
    assert manifest.read_bytes() == original
    assert (output / 'images/0000.png').read_bytes() == (tmp_path / 'page.png').read_bytes()
    assert (output / 'app.js').exists()
    with pytest.raises(FileExistsError):
        build(manifest, output)


def test_modified_image_rejected_before_output(tmp_path):
    manifest = fixture(tmp_path)
    (tmp_path / 'page.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        build(manifest, tmp_path / 'review')
    assert not (tmp_path / 'review').exists()
