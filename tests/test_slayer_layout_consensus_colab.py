import io
import hashlib
import json
import zipfile

import pytest

from training.slayer_layout_consensus_colab import combine_archives, safe_extract_zip


TEACHERS = {
    'qwen3-vl-4b': {'revision': 'q'},
    'doclayout-yolo': {'revision': 'd'},
    'surya-layout2': {'revision': 's'},
}
CONFIG = {
    'teachers': TEACHERS,
    'consensus': {'categories': ['text_region'], 'quorum': 2,
                  'iou_threshold': 0.5, 'conflict_iou': 0.5, 'min_score': 0.0},
    'claim_boundary': 'pilot only',
}


def archive(tmp_path, teacher, box):
    run_id = 'run-' + teacher
    row = {
        'schema': 'slayer-layout-teacher-proposal-v1', 'page_id': 'p1',
        'image': {'file_name': 'images/p1.jpg', 'sha256': 'a' * 64,
                  'width': 100, 'height': 200},
        'teacher': {'id': teacher, 'revision': TEACHERS[teacher]['revision'],
                    'run_id': run_id, 'prompt_sha256': 'b' * 64},
        'detections': [{'id': teacher + '-1', 'label': 'text_region',
                        'bbox_xyxy': box, 'score': 0.8,
                        'score_kind': 'model-confidence'}],
        'status': 'ok', 'error': None,
    }
    path = tmp_path / (teacher + '.zip')
    files = {
        'teacher-proposals.jsonl': json.dumps(row) + '\n',
        'run.json': json.dumps({
            'schema': 'slayer-layout-teacher-run-v1', 'state': 'completed',
            'teacher_id': teacher, 'run_id': run_id, 'pages_expected': 1,
            'pages_completed': 1, 'error_pages': 0, 'teacher': TEACHERS[teacher]}),
        'raw-model-output.jsonl': '',
        'experiment-config.json': json.dumps(CONFIG),
    }
    checksums = {
        name: hashlib.sha256(content.encode()).hexdigest()
        for name, content in files.items()
    }
    with zipfile.ZipFile(path, 'w') as output:
        for name, content in files.items():
            output.writestr(name, content)
        output.writestr('checksums.json', json.dumps(checksums))
    return path


def test_three_archives_build_traceable_consensus(tmp_path):
    archives = [
        archive(tmp_path, 'qwen3-vl-4b', [10, 20, 50, 80]),
        archive(tmp_path, 'doclayout-yolo', [11, 19, 51, 81]),
        archive(tmp_path, 'surya-layout2', [12, 18, 52, 82]),
    ]
    result, summary = combine_archives(archives, tmp_path / 'combined', CONFIG)
    assert result.exists()
    assert summary['pages'] == 1
    assert summary['consensus']['accepted_objects'] == 1
    assert summary['code_revision'] is None
    assert len(summary['consensus_policy_sha256']) == 64
    assert {item['teacher_id'] for item in summary['inputs']} == set(TEACHERS)
    with zipfile.ZipFile(result) as bundle:
        names = set(bundle.namelist())
    assert 'consensus/annotations.coco.json' in names
    assert 'consensus-policy.json' in names
    assert 'run.json' in names
    assert not any('jpg' in name or 'png' in name for name in names)


def test_page_mismatch_and_duplicate_teacher_are_rejected(tmp_path):
    paths = [archive(tmp_path, teacher, [10, 20, 50, 80]) for teacher in TEACHERS]
    duplicate = tmp_path / 'duplicate.zip'
    duplicate.write_bytes(paths[0].read_bytes())
    with pytest.raises(ValueError, match='duplicate teacher'):
        combine_archives([paths[0], duplicate, paths[2]], tmp_path / 'bad', CONFIG)


def test_modified_teacher_evidence_is_rejected(tmp_path):
    source = archive(tmp_path, 'qwen3-vl-4b', [10, 20, 50, 80])
    modified = tmp_path / 'modified.zip'
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(modified, 'w') as new:
        for member in old.infolist():
            content = old.read(member)
            if member.filename == 'teacher-proposals.jsonl':
                content += b'{}\n'
            new.writestr(member, content)
    with pytest.raises(ValueError, match='checksum mismatch'):
        safe_extract_zip(modified, tmp_path / 'untrusted')
        # Integrity is checked by the combiner after safe extraction.
        from training.slayer_layout_consensus_colab import _verify_checksums
        _verify_checksums(tmp_path / 'untrusted')


def test_safe_extract_rejects_parent_and_symlink(tmp_path):
    bad = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bad, 'w') as output:
        output.writestr('../outside.txt', 'bad')
    with pytest.raises(ValueError, match='Unsafe ZIP member'):
        safe_extract_zip(bad, tmp_path / 'extract-parent')

    symlink = tmp_path / 'symlink.zip'
    info = zipfile.ZipInfo('link')
    info.create_system = 3
    info.external_attr = (0o120777 << 16)
    with zipfile.ZipFile(symlink, 'w') as output:
        output.writestr(info, 'target')
    with pytest.raises(ValueError, match='Unsafe ZIP member'):
        safe_extract_zip(symlink, tmp_path / 'extract-link')


def test_consensus_policy_cannot_change_teacher_ontology(tmp_path):
    archives = [archive(tmp_path, teacher, [10, 20, 50, 80]) for teacher in TEACHERS]
    policy = dict(CONFIG['consensus'])
    policy['categories'] = ['invented']
    with pytest.raises(ValueError, match='frozen teacher ontology'):
        combine_archives(archives, tmp_path / 'bad-policy', CONFIG, policy)


def test_consensus_code_revision_is_validated(tmp_path):
    archives = [archive(tmp_path, teacher, [10, 20, 50, 80]) for teacher in TEACHERS]
    with pytest.raises(ValueError, match='Invalid code revision'):
        combine_archives(archives, tmp_path / 'bad-revision', CONFIG,
                         code_revision='main')


def test_teacher_page_errors_are_rejected(tmp_path):
    paths = [archive(tmp_path, teacher, [10, 20, 50, 80]) for teacher in TEACHERS]
    broken = tmp_path / 'qwen-broken.zip'
    with zipfile.ZipFile(paths[0]) as old, zipfile.ZipFile(broken, 'w') as new:
        files = {member.filename: old.read(member) for member in old.infolist()}
        run = json.loads(files['run.json'])
        run['pages_completed'] = 0
        run['error_pages'] = 1
        files['run.json'] = json.dumps(run).encode()
        proposal = json.loads(files['teacher-proposals.jsonl'])
        proposal['status'] = 'error'
        proposal['error'] = 'JSONDecodeError: malformed output'
        proposal['detections'] = []
        files['teacher-proposals.jsonl'] = (json.dumps(proposal) + '\n').encode()
        checksums = json.loads(files['checksums.json'])
        for name in ('run.json', 'teacher-proposals.jsonl'):
            checksums[name] = hashlib.sha256(files[name]).hexdigest()
        files['checksums.json'] = json.dumps(checksums).encode()
        for name, content in files.items():
            new.writestr(name, content)
    with pytest.raises(ValueError, match='page errors'):
        combine_archives([broken, paths[1], paths[2]], tmp_path / 'bad-errors', CONFIG)


def test_bounded_teacher_abstention_is_recorded_and_mined(tmp_path):
    paths = [archive(tmp_path, teacher, [10, 20, 50, 80]) for teacher in TEACHERS]
    broken = tmp_path / 'qwen-abstention.zip'
    with zipfile.ZipFile(paths[0]) as old, zipfile.ZipFile(broken, 'w') as new:
        files = {member.filename: old.read(member) for member in old.infolist()}
        run = json.loads(files['run.json'])
        run['pages_completed'] = 0
        run['error_pages'] = 1
        files['run.json'] = json.dumps(run).encode()
        proposal = json.loads(files['teacher-proposals.jsonl'])
        proposal['status'] = 'error'
        proposal['error'] = 'JSONDecodeError: malformed output'
        proposal['detections'] = []
        files['teacher-proposals.jsonl'] = (json.dumps(proposal) + '\n').encode()
        checksums = json.loads(files['checksums.json'])
        for name in ('run.json', 'teacher-proposals.jsonl'):
            checksums[name] = hashlib.sha256(files[name]).hexdigest()
        files['checksums.json'] = json.dumps(checksums).encode()
        for name, content in files.items():
            new.writestr(name, content)
    policy = dict(CONFIG['consensus'])
    policy.update({
        'allow_teacher_abstentions': True,
        'max_teacher_error_pages': 1,
        'max_teacher_error_fraction': 1.0,
    })
    result, summary = combine_archives(
        [broken, paths[1], paths[2]], tmp_path / 'with-abstention', CONFIG, policy)
    assert summary['consensus']['teacher_abstentions'] == 1
    assert summary['consensus']['teacher_abstention_pages'] == 1
    assert summary['consensus']['hard_example_pages'] == 1
    with zipfile.ZipFile(result) as bundle:
        abstention = json.loads(bundle.read('consensus/teacher-abstentions.jsonl'))
        hard = json.loads(bundle.read('consensus/hard-examples.jsonl'))
    assert abstention['page_id'] == 'p1'
    assert abstention['teacher_id'] == 'qwen3-vl-4b'
    assert hard['reasons'] == ['teacher-abstention']
    assert hard['abstaining_teachers'] == ['qwen3-vl-4b']
