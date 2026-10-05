import json
import zipfile

from PIL import Image
import pytest

from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.prepare_recognizer_line_remediation import prepare


def fixture(tmp_path):
    source = tmp_path/'parent'
    source.mkdir()
    region = source/'region.jpg'
    image = Image.new('RGB', (12, 12), 'white')
    image.putpixel((5, 4), (20, 20, 20))
    image.save(region, 'JPEG')
    with Image.open(region) as decoded:
        decoded.crop((2, 1, 10, 8)).save(source/'crop.png')
    context = {'image': 'region.jpg', 'sha256': digest(region), 'width': 12, 'height': 12,
        'source_region_id': 'region', 'text': 'historical \u017f\u00e1\u0247'}
    base = {'image': 'crop.png', 'sha256': digest(source/'crop.png'), 'context': context,
        'source_region_id': 'region', 'source_split': 'train', 'source_reference_text': 'weak source',
        'text': context['text'], 'review_status': 'verified', 'geometry_decision': 'reject-crop',
        'review_event_id': 'event', 'annotation_note': 'cut', 'teacher_diagnostics': {'candidate_text': 'old raw'},
        'eligible_for_training': False, 'eligible_for_evaluation': False}
    rows = [{**base, 'id': 'reject', 'import_status': 'rejected-crop'},
            {**base, 'id': 'pending', 'import_status': 'pending', 'review_status': 'proposed'}]
    write_rows(source/'reviewed-lines.jsonl', rows)
    write_json(source/'import-report.json', {'review_export_sha256': 'review-sha'})
    write_json(source/'original-review.json', {'original': True})
    write_json(source/'source-report.json', {'original_manifest_sha256': 'original-sha', 'source_archive_sha256': 'archive-sha'})
    write_json(source/'pilot-config.json', {'frozen': True})
    write_json(source/'checksums.json', {p.name: digest(p) for p in source.iterdir() if p.is_file()})
    recipes = tmp_path/'recipes.json'
    write_json(recipes, {'schema': 'slayer-recognizer-recrop-proposals-v1', 'parent_review_sha256': 'review-sha',
        'actor_kind': 'AI-assistant', 'human_review_required': True,
        'proposals': [{'id': 'reject', 'crop_sha256': base['sha256'], 'context_sha256': context['sha256'],
                       'bbox': [0, 0, 12, 6]}]})
    return source, recipes, tmp_path/'followup', rows


def test_remediation_keeps_pending_text_and_requires_new_geometry_review(tmp_path):
    source, recipes, out, parent = fixture(tmp_path)
    original = (source/'reviewed-lines.jsonl').read_bytes()
    report = prepare(source, recipes, out)
    rows = read_rows(out/'input/manifest.jsonl')
    assert report['lines'] == 2 and report['prior_accepted_lines_repeated'] == 0
    assert report['training_examples_created'] == report['gold_labels_created'] == 0
    assert (source/'reviewed-lines.jsonl').read_bytes() == original
    assert all(r['eligible_for_training'] is r['eligible_for_evaluation'] is r['line_geometry_verified'] is False for r in rows)
    assert all(r['review_status'] == r['geometry_decision'] == 'unreviewed' for r in rows)
    assert rows[0]['sha256'] != parent[0]['sha256']
    assert rows[0]['parent_review_event_id'] == 'event'
    assert rows[1]['sha256'] == parent[1]['sha256'] and rows[1]['text'] == parent[1]['text']
    assert rows[1]['parent_weak_reference_text'] == 'weak source'
    with Image.open(source/'region.jpg') as original_image, Image.open(out/'input'/rows[0]['image']) as crop:
        assert original_image.crop((0, 0, 12, 6)).tobytes() == crop.tobytes()
    with zipfile.ZipFile(out.with_suffix('.zip')) as stream:
        assert all(not name.endswith(('.html', '.js', '.css', '.py')) for name in stream.namelist())
    assert (out/'review/index.html').exists()
    with pytest.raises(FileExistsError):
        prepare(source, recipes, out)


@pytest.mark.parametrize('field,value', [('crop_sha256', 'wrong'), ('context_sha256', 'wrong'),
    ('bbox', [-1, 0, 12, 6]), ('bbox', [0, 0, 13, 6]), ('bbox', [0, 0, 12, 0]), ('bbox', [0, 0, 12, 6.0])])
def test_bad_recrops_fail_before_writing(tmp_path, field, value):
    source, recipes, out, _ = fixture(tmp_path)
    plan = json.loads(recipes.read_text())
    plan['proposals'][0][field] = value
    write_json(recipes, plan)
    with pytest.raises(ValueError):
        prepare(source, recipes, out)
    assert not out.exists()


@pytest.mark.parametrize('kind', ['missing-proposal', 'parent-sha', 'approval', 'checksum'])
def test_history_and_no_automatic_approval_gates(tmp_path, kind):
    source, recipes, out, _ = fixture(tmp_path)
    plan = json.loads(recipes.read_text())
    if kind == 'missing-proposal':
        plan['proposals'] = []
    elif kind == 'parent-sha':
        plan['parent_review_sha256'] = 'wrong'
    elif kind == 'approval':
        plan['human_review_required'] = False
    else:
        (source/'crop.png').write_bytes(b'bad')
    write_json(recipes, plan)
    with pytest.raises(ValueError):
        prepare(source, recipes, out)
    assert not out.exists()
