import io
import json
import pytest
from PIL import Image

from training import geometry_holdout_runner as runner


def manifest():
    rows = []
    for i in range(12):
        rows.append({'id': f'r{i}', 'split': 'train', 'collection': f'c{i}', 'page_id': f'p{i}',
                     'text': 'one\ntwo', 'region_type': 'paragraph', 'image_sha256': 'a' * 64,
                     'license': 'CC-BY-3.0', 'source_path': f'regions/train/images/r{i}.jpg',
                     'reference_status': 'upstream-unreviewed', 'reference_private_use_count': 0,
                     'eligible_for_benchmark': False})
    return {'scope': 'geometry holdout diagnostic; not benchmark or model holdout',
            'dataset': runner.DATASET, 'revision': runner.REVISION, 'regions': rows}


def test_manifest_validation_and_hash():
    data = json.dumps(manifest()).encode()
    assert len(runner.load_manifest(data, runner.digest(data))['regions']) == 12
    with pytest.raises(ValueError, match='checksum'):
        runner.load_manifest(data, '0' * 64)


def test_manifest_rejects_duplicate_collection_and_unsafe_path():
    value = manifest()
    value['regions'][1]['collection'] = value['regions'][0]['collection']
    data = json.dumps(value).encode()
    with pytest.raises(ValueError, match='one region'):
        runner.load_manifest(data, runner.digest(data))
    value = manifest()
    value['regions'][0]['source_path'] = '../secret'
    data = json.dumps(value).encode()
    with pytest.raises(ValueError, match='Unsafe'):
        runner.load_manifest(data, runner.digest(data))


def test_fetch_checks_source_hash():
    content = b'image'
    row = {'id': 'r', 'source_path': 'regions/train/images/r.jpg',
           'image_sha256': runner.digest(content)}
    class Response:
        def read(self):
            return content
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
    response = Response()
    assert runner.fetch_sources([row], lambda *args, **kwargs: response) == [content]
    row['image_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        runner.fetch_sources([row], lambda *args, **kwargs: response)


def test_segment_variants_keep_source_immutable(monkeypatch):
    image = Image.new('RGB', (20, 10), 'white')
    buf = io.BytesIO()
    image.save(buf, 'PNG')
    geometry = {'boxes': [[0, 0, 20, 10]], 'foreign_ink_fraction': [0.0],
                'line_bands': [{'top': [0] * 20, 'bottom': [10] * 20}]}
    monkeypatch.setattr(runner, 'detect_lines', lambda source, follow_lines=False: geometry)
    for variant in runner.VARIANTS:
        result, crops = runner.segment(buf.getvalue(), variant)
        assert result == geometry
        assert crops[0].size == (20, 10)


def test_scores_keep_all_regions_and_compare():
    rows = [{'id': 'a', 'text': 'abc', 'reference_private_use_count': 0},
            {'id': 'b', 'text': 'ábc', 'reference_private_use_count': 1}]
    rectangle = [{'id': 'a', 'text': 'axc', 'status': 'ok', 'detected_lines': 1},
                 {'id': 'b', 'text': 'abc', 'status': 'ok', 'detected_lines': 1}]
    bands = [{'id': 'a', 'text': 'abc', 'status': 'ok', 'detected_lines': 1},
             {'id': 'b', 'text': 'abc', 'status': 'ok', 'detected_lines': 1}]
    score = runner.score(rows, rectangle)
    assert score['all_regions']['regions'] == 2
    assert score['without_private_use_references']['regions'] == 1
    comparison = runner.compare_variants(rows, rectangle, bands)
    assert comparison['improved'] == 1
    assert comparison['net_character_edit_change'] == -1
