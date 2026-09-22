import json

import pytest

from training.build_release import discover, build_release
from training.generate_documents import generate
from training.submission_tsv import unpack
from training.validate_submission import load_jsonl


def data_dir(tmp_path, name, seed, degradations):
    target = tmp_path / name
    generate(target, count=4, seed=seed, degradations=degradations)
    return target


@pytest.fixture()
def release(tmp_path):
    splits = {
        'train': discover(data_dir(tmp_path, 'train-src', 1, ('clean', 'scan'))),
        'testA': discover(data_dir(tmp_path, 'testA-src', 2, ('photo',))),
        'testB': discover(data_dir(tmp_path, 'testB-src', 3, ('compress',))),
    }
    report = build_release(splits, tmp_path / 'release', '0.1-test',
                           holdout_field='degradation', holdout_split='testB')
    return tmp_path / 'release', report


def test_release_layout_hides_the_right_ground_truth(release):
    output, report = release
    assert (output / 'hf' / 'train' / 'manifest-C.jsonl').exists()
    assert (output / 'hf' / 'train' / 'images').is_dir()
    assert not (output / 'hf' / 'testA' / 'manifest-A.jsonl').exists()
    assert (output / 'hf' / 'testA' / 'images').is_dir()
    assert (output / 'hf' / 'testA' / 'in-C.tsv').exists()
    assert (output / 'private' / 'testA' / 'manifest-C.jsonl').exists()
    assert not (output / 'hf' / 'testB').exists()
    assert (output / 'private' / 'testB' / 'manifest-B.jsonl').exists()
    assert (output / 'private' / 'testB' / 'images').is_dir()
    for name in ('DATASET_CARD.md', 'LICENSE-DATA.txt', 'LICENSE-CODE.txt', 'RELEASE.json'):
        assert (output / name).exists()
    assert report['roles'] == {'train': 'public-gt', 'testA': 'public-ids',
                               'testB': 'private'}
    assert report['integrity']['violations'] == 0
    assert not report['integrity_overridden']


def test_expected_tsv_round_trips_gold_payloads(release):
    output, _ = release
    package = output / 'amueval' / 'train' / 'C'
    restored = output / 'restored.jsonl'
    unpack(package / 'in.tsv', package / 'expected.tsv', restored, 'C')
    gold = {row['id']: row['fields'] for row in
            load_jsonl(output / 'hf' / 'train' / 'manifest-C.jsonl')}
    rows = {row['id']: row['fields'] for row in load_jsonl(restored)}
    assert rows == gold
    package_b = output / 'amueval' / 'train' / 'B'
    unpack(package_b / 'in.tsv', package_b / 'expected.tsv', restored, 'B')
    gold_html = {row['id']: row['html'] for row in
                 load_jsonl(output / 'hf' / 'train' / 'manifest-B.jsonl')}
    assert {row['id']: row['html'] for row in load_jsonl(restored)} == gold_html


def test_sample_submission_is_an_empty_valid_template(release):
    output, _ = release
    package = output / 'amueval' / 'testA' / 'A'
    restored = output / 'sample.jsonl'
    unpack(package / 'in.tsv', package / 'sample-out.tsv', restored, 'A')
    rows = load_jsonl(restored)
    assert all(row == {'id': row['id'], 'status': 'ok', 'text': ''} for row in rows)


def test_release_json_checksums_match_files(release):
    from training.stage_impact_benchmark import digest
    output, report = release
    for relative, sha in report['files'].items():
        if relative == 'RELEASE.json':
            continue
        assert digest(output / relative) == sha


def test_integrity_gate_refuses_leaky_release(tmp_path):
    splits = {
        'train': discover(data_dir(tmp_path, 'train', 5, ('clean',))),
        'testA': discover(data_dir(tmp_path, 'testA', 5, ('photo',))),  # the same seed: same text
    }
    with pytest.raises(ValueError, match='violations'):
        build_release(splits, tmp_path / 'refused', '0.0')
    report = build_release(splits, tmp_path / 'allowed', '0.0', allow_violations=True)
    assert report['integrity']['violations'] > 0
    assert report['integrity_overridden']


def test_existing_output_is_never_overwritten(tmp_path):
    splits = {'train': discover(data_dir(tmp_path, 'train', 9, ('clean',)))}
    build_release(splits, tmp_path / 'out', '0.1')
    with pytest.raises(FileExistsError):
        build_release(splits, tmp_path / 'out', '0.1')
