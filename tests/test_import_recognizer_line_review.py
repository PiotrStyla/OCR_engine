import copy
import hashlib
import json
import zipfile

from PIL import Image
import pytest

from training import import_recognizer_line_review as module
from training.adjudicate_reviews import text_hash
from training.full_page_pilot import digest, read_rows, write_json, write_rows


def fixture(tmp_path, monkeypatch):
    review = tmp_path/'review'
    (review/'input/lines').mkdir(parents=True)
    (review/'input/regions').mkdir()
    Image.new('L', (20, 10), 180).save(review/'input/lines/line.png')
    Image.new('RGB', (40, 20), 'white').save(review/'input/regions/region.jpg')
    line_sha = digest(review/'input/lines/line.png')
    region_sha = digest(review/'input/regions/region.jpg')
    text = 'Na nie\u017fko\u0144czon\u0105 BOGU Chwa\u0142\u0247.'
    row = {'id': 'line', 'image': 'lines/line.png', 'sha256': line_sha, 'text': text,
        'width': 20, 'height': 10, 'source_region_id': 'region', 'page_id': 'page',
        'source_split': 'train', 'final_test': False, 'eligible_for_training': False,
        'eligible_for_evaluation': False, 'dataset': module.DATASET, 'revision': module.REVISION,
        'collection': 'allowed', 'source_image_sha256': region_sha, 'source_region_text': text}
    context = {'image': 'regions/region.jpg', 'sha256': region_sha, 'text': text,
        'source_region_id': 'region', 'width': 40, 'height': 20}
    write_rows(review/'input/manifest.jsonl', [row])
    write_json(review/'contexts.json', {'line': context})
    write_json(review/'diagnostics.json', {'line': {'candidate_text': 'raw teacher'}})
    write_json(review/'report.json', {'review_manifest_sha256': digest(review/'input/manifest.jsonl'),
                                    'original_manifest_sha256': 'frozen-original'})
    write_json(review/'checksums.json', {p.relative_to(review).as_posix(): digest(p)
                                       for p in review.rglob('*') if p.is_file()})
    config = tmp_path/'config.json'
    write_json(config, {'dataset': {'manifest_sha256': 'frozen-original', 'forbidden_collections': ['forbidden']}})
    metadata = tmp_path/'metadata'
    sums = {}
    for split in ('train', 'validation', 'test'):
        (metadata/split).mkdir(parents=True)
        source = {'id': 'region' if split == 'train' else split, 'split': split,
            'collection': 'allowed' if split == 'train' else split, 'page_id': 'page' if split == 'train' else split,
            'text': text, 'image_sha256': region_sha if split == 'train' else split}
        write_rows(metadata/split/'metadata.jsonl', [source])
        sums[split] = digest(metadata/split/'metadata.jsonl')
    monkeypatch.setattr(module, 'METADATA_SHA256', sums)
    observations = tmp_path/'observations.json'
    write_json(observations, {'schema': 'slayer-recognizer-data-pilot-agent-observations-v1', 'observations': []})
    event = {'id': 'event', 'page_id': 'line', 'timestamp': '2026-10-05T11:00:00Z',
        'reviewer': 'Reviewer', 'decision': 'verified', 'geometry_decision': 'complete-line',
        'note': '', 'before': text, 'after': text, 'original_text_sha256': text_hash(text),
        'image_sha256': line_sha, 'context_image_sha256': region_sha}
    packet = {'schema': 'slayer-recognizer-line-review-v1', 'manifest_sha256': digest(review/'input/manifest.jsonl'),
              'events': [event]}
    patch = tmp_path/'patch.json'
    write_json(patch, packet)
    return dict(review_directory=review, review_json=patch, config=config, metadata_directory=metadata,
                observations=observations, output=tmp_path/'imported'), packet, row, context


def test_import_preserves_bytes_glyphs_and_never_creates_gold(tmp_path, monkeypatch):
    args, packet, row, _ = fixture(tmp_path, monkeypatch)
    original = args['review_json'].read_bytes()
    report = module.import_review(**args)
    candidate = read_rows(args['output']/'training-candidates.jsonl')[0]
    assert candidate['text'] == row['text']
    assert candidate['text_sha256'] == text_hash(row['text'])
    assert candidate['eligible_for_training'] is True
    assert candidate['eligible_for_evaluation'] is candidate['gold'] is False
    assert report['training_candidates'] == 1 and report['gold_labels_created'] == 0
    assert (args['output']/'original-review.json').read_bytes() == original
    assert candidate['teacher_diagnostics']['candidate_text'] == 'raw teacher'
    with zipfile.ZipFile(args['output'].with_suffix('.zip')) as stream:
        sums = json.loads(stream.read('checksums.json'))
        assert set(sums) == set(stream.namelist()) - {'checksums.json'}
        assert all(hashlib.sha256(stream.read(n)).hexdigest() == sha for n, sha in sums.items())
    with pytest.raises(FileExistsError):
        module.import_review(**args)


@pytest.mark.parametrize('decision,geometry,note,status', [
    ('proposed', 'complete-line', '', 'pending'),
    ('verified', 'reject-crop', 'cut off', 'rejected-crop'),
    ('needs-review', 'unreviewed', '', 'pending')])
def test_text_and_geometry_are_separate_gates(tmp_path, monkeypatch, decision, geometry, note, status):
    args, packet, _, _ = fixture(tmp_path, monkeypatch)
    packet['events'][0].update(decision=decision, geometry_decision=geometry, note=note)
    write_json(args['review_json'], packet)
    report = module.import_review(**args)
    assert report['source_statuses'] == {status: 1}
    assert report['training_candidates'] == 0


@pytest.mark.parametrize('field,value', [('image_sha256', 'bad'), ('context_image_sha256', 'bad'),
    ('original_text_sha256', 'bad'), ('before', 'bad'), ('geometry_decision', 'invalid'),
    ('timestamp', '2026-10-05'), ('reviewer', ' '), ('decision', 'invalid')])
def test_tampered_events_fail_before_writing(tmp_path, monkeypatch, field, value):
    args, packet, _, _ = fixture(tmp_path, monkeypatch)
    packet['events'][0][field] = value
    write_json(args['review_json'], packet)
    with pytest.raises(ValueError):
        module.import_review(**args)
    assert not args['output'].exists()


@pytest.mark.parametrize('kind', ['duplicate', 'tie', 'backwards', 'bad-chain', 'schema', 'manifest', 'source-image'])
def test_bad_history_and_sources_fail_closed(tmp_path, monkeypatch, kind):
    args, packet, _, _ = fixture(tmp_path, monkeypatch)
    if kind in ('duplicate', 'tie', 'backwards', 'bad-chain'):
        e = copy.deepcopy(packet['events'][0])
        if kind != 'duplicate':
            e['id'] = 'second'
        if kind == 'backwards':
            e['timestamp'] = '2026-10-04T11:00:00Z'
        elif kind == 'bad-chain':
            e.update(timestamp='2026-10-05T12:00:00Z', before='not current')
        packet['events'].append(e)
    elif kind == 'source-image':
        (args['review_directory']/'input/lines/line.png').write_bytes(b'bad')
    else:
        packet['schema' if kind == 'schema' else 'manifest_sha256'] = 'bad'
    write_json(args['review_json'], packet)
    with pytest.raises(ValueError):
        module.import_review(**args)
    assert not args['output'].exists()


def test_counts_original_to_final_changes_and_does_not_read_notes_as_text(tmp_path, monkeypatch):
    args, packet, row, _ = fixture(tmp_path, monkeypatch)
    first = packet['events'][0]
    first.update(decision='proposed', after=row['text']+' x', note='not a transcription')
    second = {**first, 'id': 'second', 'timestamp': '2026-10-05T12:00:00Z',
              'decision': 'verified', 'before': first['after']}
    packet['events'].append(second)
    write_json(args['review_json'], packet)
    report = module.import_review(**args)
    assert report['changed_source_transcriptions'] == 1
    assert read_rows(args['output']/'training-candidates.jsonl')[0]['text'] == first['after']


@pytest.mark.parametrize('confirmed', [False, True])
def test_known_bad_crop_quarantined_and_context_replacement_has_new_identity(tmp_path, monkeypatch, confirmed):
    args, packet, row, context = fixture(tmp_path, monkeypatch)
    write_json(args['observations'], {'schema': 'slayer-recognizer-data-pilot-agent-observations-v1',
        'observations': [{'id': 'line', 'kind': 'crop-reference-alignment-failure', 'image_sha256': row['sha256']}]})
    if confirmed:
        c = {'id': 'line', 'scope': 'full-context-single-line', 'review_event_id': 'event',
            'crop_sha256': row['sha256'], 'context_sha256': context['sha256'], 'text_sha256': text_hash(row['text']),
            'source_kind': 'direct-user-message', 'question': 'Did you verify the full context?',
            'answer': '1', 'resolved_answer': 'Yes'}
        path = tmp_path/'confirmation.json'
        write_json(path, {'schema': 'slayer-line-context-scope-confirmation-v1', 'confirmations': [c]})
        args['context_confirmations'] = path
    report = module.import_review(**args)
    assert report['source_statuses'] == {'geometry-conflict': 1}
    old = read_rows(args['output']/'reviewed-lines.jsonl')[0]
    assert old['review_status'] == 'verified' and old['eligible_for_training'] is False
    assert report['training_candidates'] == int(confirmed)
    if confirmed:
        new = read_rows(args['output']/'training-candidates.jsonl')[0]
        assert new['id'] != old['id'] and new['sha256'] != old['sha256']
        assert new['supersedes_crop_sha256'] == old['sha256']
        with Image.open(args['output']/new['image']) as png, Image.open(args['output']/new['context']['image']) as jpg:
            assert png.size == jpg.size and png.tobytes() == jpg.tobytes()


@pytest.mark.parametrize('after', ['', 'bad\ufffd', 'private\ue000', 'two\nlines'])
def test_invalid_transcriptions_are_quarantined(tmp_path, monkeypatch, after):
    args, packet, _, _ = fixture(tmp_path, monkeypatch)
    packet['events'][0]['after'] = after
    write_json(args['review_json'], packet)
    report = module.import_review(**args)
    assert report['training_candidates'] == 0
    assert report['source_statuses'] == {'text-quality-quarantine': 1}


def test_frozen_metadata_is_checked(tmp_path, monkeypatch):
    args, _, _, _ = fixture(tmp_path, monkeypatch)
    (args['metadata_directory']/'test/metadata.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='Frozen region metadata'):
        module.import_review(**args)
    assert not args['output'].exists()


@pytest.mark.parametrize('kind', ['forbidden-collection', 'heldout-page', 'heldout-image'])
def test_firewall_rejects_leakage_before_writing(tmp_path, monkeypatch, kind):
    args, _, row, _ = fixture(tmp_path, monkeypatch)
    if kind == 'forbidden-collection':
        cfg = json.loads(args['config'].read_text())
        cfg['dataset']['forbidden_collections'].append(row['collection'])
        write_json(args['config'], cfg)
    else:
        path = args['metadata_directory']/'validation/metadata.jsonl'
        records = read_rows(path)
        records[0]['page_id' if kind == 'heldout-page' else 'image_sha256'] = (
            row['page_id'] if kind == 'heldout-page' else row['sha256'])
        write_rows(path, records)
        monkeypatch.setitem(module.METADATA_SHA256, 'validation', digest(path))
    with pytest.raises(ValueError, match='firewall'):
        module.import_review(**args)
    assert not args['output'].exists()


def test_multiple_reviewers_are_not_silently_collapsed(tmp_path, monkeypatch):
    args, packet, _, _ = fixture(tmp_path, monkeypatch)
    packet['events'].append({**packet['events'][0], 'id': 'second', 'reviewer': 'Other',
                             'timestamp': '2026-10-05T12:00:00Z'})
    write_json(args['review_json'], packet)
    with pytest.raises(ValueError, match='one reviewer'):
        module.import_review(**args)
    assert not args['output'].exists()


def test_unreviewed_lines_remain_in_ledger_but_not_training(tmp_path, monkeypatch):
    args, packet, _, _ = fixture(tmp_path, monkeypatch)
    packet['events'] = []
    write_json(args['review_json'], packet)
    report = module.import_review(**args)
    assert report['source_statuses'] == {'unreviewed': 1}
    assert report['training_candidates'] == 0
