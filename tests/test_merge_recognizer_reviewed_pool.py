from copy import deepcopy
import hashlib
import json
import zipfile

from PIL import Image
import pytest

from tests.test_import_recognizer_line_review import fixture
from training import import_recognizer_line_review as importer
from training import merge_recognizer_reviewed_pool as module
from training.full_page_pilot import digest, read_rows, write_json, write_rows


def seal(root):
    write_json(root/'checksums.json', {p.relative_to(root).as_posix(): digest(p)
        for p in root.rglob('*') if p.is_file() and p.name != 'checksums.json'})


def packages(tmp_path, monkeypatch):
    args, packet, row, context = fixture(tmp_path, monkeypatch)
    review = args['review_directory']
    packet['events'][0]['decision'] = 'proposed'
    Image.new('L', (20, 10), 20).save(review/'input/lines/accepted.png')
    accepted = {**row, 'id': 'accepted', 'image': 'lines/accepted.png',
                'sha256': digest(review/'input/lines/accepted.png')}
    write_rows(review/'input/manifest.jsonl', [row, accepted])
    write_json(review/'contexts.json', {'line': context, 'accepted': context})
    write_json(review/'diagnostics.json', {'line': {}, 'accepted': {}})
    write_json(review/'report.json', {'review_manifest_sha256': digest(review/'input/manifest.jsonl'),
        'original_manifest_sha256': 'frozen-original'})
    packet['manifest_sha256'] = digest(review/'input/manifest.jsonl')
    packet['events'].append({**packet['events'][0], 'id': 'accepted-event', 'page_id': 'accepted',
        'image_sha256': accepted['sha256'], 'decision': 'verified'})
    write_json(args['review_json'], packet)
    seal(review)
    importer.import_review(**args)
    base = args['output']
    ledger = read_rows(base/'reviewed-lines.jsonl')
    parent = next(r for r in ledger if r['id'] == 'line')
    follow = {**row, 'id': 'line__text-review-v1', 'parent_line_id': 'line',
        'parent_crop_sha256': row['sha256'], 'parent_review_event_id': parent['review_event_id'],
        'parent_review_export_sha256': digest(base/'original-review.json'),
        'parent_import_status': 'pending', 'parent_weak_reference_text': row['text'],
        'transformation': {'kind': 'unchanged-image-text-confirmation'}}
    write_rows(review/'input/manifest.jsonl', [follow])
    write_json(review/'contexts.json', {follow['id']: context})
    write_json(review/'diagnostics.json', {follow['id']: {}})
    write_json(review/'report.json', {'review_manifest_sha256': digest(review/'input/manifest.jsonl'),
        'original_manifest_sha256': 'frozen-original'})
    packet['manifest_sha256'] = digest(review/'input/manifest.jsonl')
    packet['events'] = [{**packet['events'][0], 'id': 'follow-event', 'page_id': follow['id'],
        'decision': 'verified', 'timestamp': '2026-10-05T12:00:00Z'}]
    write_json(args['review_json'], packet)
    seal(review)
    args['output'] = tmp_path/'followup'
    importer.import_review(**args)
    return dict(base_directory=base, followup_directory=args['output'], config=args['config'],
        metadata_directory=args['metadata_directory'], output=tmp_path/'pool')


def test_merge_replays_evidence_preserves_history_and_glyphs(tmp_path, monkeypatch):
    args = packages(tmp_path, monkeypatch)
    original = (args['base_directory']/'original-review.json').read_bytes()
    report = module.merge(**args)
    root = args['output']
    rows = read_rows(root/'manifest.jsonl')
    assert len(rows) == report['unique_root_lines'] == 2
    assert report['historical_ledger_rows'] == 3
    assert report['inactive_historical_rows'] == 1
    assert report['imports_replayed'] is True
    assert report['training_freeze_ready'] is report['model_training_performed'] is report['sota_claim'] is False
    assert all(r['gold'] is r['eligible_for_evaluation'] is False for r in rows)
    assert '\u017f' in rows[0]['text'] and '\u0247' in rows[0]['text']
    assert (root/'history/base/original-review.json').read_bytes() == original
    assert (args['base_directory']/'original-review.json').read_bytes() == original
    sums = module.verified_package(root)
    with zipfile.ZipFile(root.with_suffix('.zip')) as stream:
        assert set(stream.namelist()) == set(sums) | {'checksums.json'}
        assert all(hashlib.sha256(stream.read(n)).hexdigest() == sha for n, sha in sums.items())
    with pytest.raises(FileExistsError):
        module.merge(**args)


@pytest.mark.parametrize('field,value', [
    ('parent_line_id', 'unknown'), ('parent_line_id', 'accepted'),
    ('parent_crop_sha256', 'bad'), ('parent_review_event_id', 'bad'),
    ('parent_review_export_sha256', 'bad'), ('parent_weak_reference_text', 'modernized'),
    ('parent_import_status', 'verified'), ('source_image_sha256', 'bad')])
def test_unbound_followups_are_rejected(tmp_path, monkeypatch, field, value):
    args = packages(tmp_path, monkeypatch)
    base = read_rows(args['base_directory']/'training-candidates.jsonl')
    follow = read_rows(args['followup_directory']/'training-candidates.jsonl')
    follow[0][field] = value
    with pytest.raises(ValueError):
        module.validate_followups(base, follow, read_rows(args['base_directory']/'reviewed-lines.jsonl'),
            digest(args['base_directory']/'original-review.json'))


@pytest.mark.parametrize('kind', ['bytes', 'rows', 'missing-evidence', 'config', 'metadata'])
def test_tampering_fails_before_final_output(tmp_path, monkeypatch, kind):
    args = packages(tmp_path, monkeypatch)
    base = args['base_directory']
    if kind == 'bytes':
        (base/'original-review.json').write_bytes(b'{}')
    elif kind == 'rows':
        rows = read_rows(base/'training-candidates.jsonl')
        rows[0]['text'] = 'unreviewed replacement'
        write_rows(base/'training-candidates.jsonl', rows)
        seal(base)
    elif kind == 'missing-evidence':
        (base/'original-review.json').unlink()
    elif kind == 'config':
        write_json(args['config'], {})
    else:
        write_rows(args['metadata_directory']/'test/metadata.jsonl', [{}])
    with pytest.raises((ValueError, FileNotFoundError)):
        module.merge(**args)
    assert not args['output'].exists()


def test_duplicate_roots_and_images_fail(tmp_path, monkeypatch):
    args = packages(tmp_path, monkeypatch)
    base = read_rows(args['base_directory']/'training-candidates.jsonl')
    follow = read_rows(args['followup_directory']/'training-candidates.jsonl')
    ledger = read_rows(args['base_directory']/'reviewed-lines.jsonl')
    review_sha = digest(args['base_directory']/'original-review.json')
    with pytest.raises(ValueError):
        module.validate_followups(base, follow+[deepcopy(follow[0])], ledger, review_sha)
    follow[0]['sha256'] = base[0]['sha256']
    with pytest.raises(ValueError):
        module.validate_followups(base, follow, ledger, review_sha)
    assert module.root_id({'id': 'new', 'supersedes_crop_id': 'old'}) == 'old'


def test_recrop_requires_exact_pixels_and_bounded_rectangle(tmp_path):
    Image.new('RGB', (20, 10), 'white').save(tmp_path/'context.jpg')
    with Image.open(tmp_path/'context.jpg') as image:
        image.crop((0, 0, 10, 5)).save(tmp_path/'crop.png')
    row = {'image': 'crop.png', 'context': {'image': 'context.jpg'},
        'transformation': {'kind': 'rectangle-proposal-from-frozen-context', 'bbox_in_region': [0, 0, 10, 5]}}
    module.verify_recrop(tmp_path, row)
    row['transformation']['bbox_in_region'] = [-1, 0, 10, 5]
    with pytest.raises(ValueError):
        module.verify_recrop(tmp_path, row)
    row['transformation']['bbox_in_region'] = [0, 0, 10, 5]
    Image.new('RGB', (10, 5), 'black').save(tmp_path/'crop.png')
    with pytest.raises(ValueError):
        module.verify_recrop(tmp_path, row)
