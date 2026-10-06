from copy import deepcopy
import json
from pathlib import Path

import pytest

from tests.test_recognizer_work_groups import pages, group, line
from training import gate_recognizer_review_import as module
from training.audit_recognizer_work_groups import validate_groups
from training.full_page_pilot import digest, read_rows, write_json, write_rows


def reviewed(status='single-review-candidate', collection='a'):
    return {**line(collection=collection), 'import_status': status,
            'review_status': 'verified' if status == 'single-review-candidate' else 'proposed',
            'geometry_decision': 'complete-line', 'gold': False,
            'eligible_for_training': status == 'single-review-candidate',
            'image': 'input/lines/line.png', 'context': {'image': 'input/regions/region.jpg'},
            'text': '\u017f \u0292\u0307 \u0292\u0301 \u01ef \u00e1 \u0247'}


def test_gate_retains_verified_text_but_blocks_training_and_does_not_modify_input():
    row = reviewed()
    before = deepcopy(row)
    result = module.gate_rows([row], pages(), validate_groups(pages(), [group()]))[0]
    assert row == before and result['text'] == row['text']
    assert result['annotation_verified'] and result['quarantine_for_future_training']
    assert result['eligible_for_training'] is result['eligible_for_evaluation'] is False
    assert result['image'] == 'history/import/input/lines/line.png'
    assert result['context']['image'] == 'history/import/input/regions/region.jpg'


@pytest.mark.parametrize('status', ['pending', 'rejected-crop', 'unreviewed', 'geometry-conflict', 'text-quality-quarantine'])
def test_nonaccepted_annotations_are_never_promoted(status):
    result = module.gate_rows([reviewed(status)], pages(), [])[0]
    assert not result['annotation_verified']
    assert not result['eligible_for_training']
    assert result['import_status'] == status


def test_unresolved_nonfamily_is_not_cleared_for_training():
    result = module.gate_rows([reviewed(collection='c')], pages(), [])[0]
    assert result['annotation_verified']
    assert not result['quarantine_for_future_training']
    assert result['training_gate_reason'] == 'bibliographic-identity-unresolved'
    assert not result['eligible_for_training']


@pytest.mark.parametrize('field,value', [('eligible_for_training', False), ('review_status', 'proposed'),
    ('geometry_decision', 'unreviewed'), ('geometry_decision', 'reject-crop'), ('gold', True)])
def test_false_candidate_status_fails(field, value):
    with pytest.raises(ValueError):
        module.gate_rows([{**reviewed(), field: value}], pages(), [])


def package_fixture(tmp_path, monkeypatch):
    imported = tmp_path/'imported'
    work = tmp_path/'work'
    imported.mkdir()
    work.mkdir()
    write_rows(imported/'original-manifest.jsonl', [line()])
    write_rows(imported/'reviewed-lines.jsonl', [reviewed()])
    (imported/'input/lines').mkdir(parents=True)
    (imported/'input/regions').mkdir()
    (imported/'input/lines/line.png').write_bytes(b'crop')
    (imported/'input/regions/region.jpg').write_bytes(b'context')
    (imported/'original-review.json').write_bytes(b'raw UTF-8 review')
    write_rows(work/'page-signatures.jsonl', pages())
    write_json(work/'policy.json', {'groups': [group()]})
    write_rows(work/'work-family-groups.jsonl', validate_groups(pages(), [group()]))
    write_json(work/'future-training-selection-policy.json', {'training_freeze_ready': False})
    write_json(work/'source-access.json', {})
    write_json(work/'audit-report.json', {'schema': 'slayer-recognizer-work-family-risk-audit-v1',
        'training_freeze_ready': False, 'bibliographic_work_identities_verified': 0,
        'dataset': 'dataset', 'revision': 'revision', 'policy_sha256': digest(work/'policy.json'),
        'input_sha256': {'expansion-manifest.jsonl': digest(imported/'original-manifest.jsonl'),
                         'page-signatures.jsonl': digest(work/'page-signatures.jsonl')}})
    for root in (work, imported):
        write_json(root/'checksums.json', {p.relative_to(root).as_posix(): digest(p)
            for p in root.rglob('*') if p.is_file()})
    def replay(*args):
        return {'events': 1, 'reviewed_lines': 1, 'review_export_sha256': 'export',
                'review_manifest_sha256': 'manifest'}
    monkeypatch.setattr(module, 'replay_package', replay)
    return imported, work, tmp_path/'config.json', tmp_path/'metadata', tmp_path/'output'


def test_package_preserves_raw_history_and_seals_asset_paths(tmp_path, monkeypatch):
    args = package_fixture(tmp_path, monkeypatch)
    report = module.gate(*args)
    out = args[-1]
    assert report['annotation_accepted'] == report['accepted_work_family_quarantined'] == 1
    assert report['eligible_training_examples'] == report['human_decisions_modified'] == 0
    assert (out/'history/import/original-review.json').read_bytes() == b'raw UTF-8 review'
    row = read_rows(out/'manifest.jsonl')[0]
    assert (out/row['image']).read_bytes() == b'crop'
    module.verified_package(out)
    with pytest.raises(FileExistsError):
        module.gate(*args)


def test_work_group_mutation_rejected_before_output(tmp_path, monkeypatch):
    args = package_fixture(tmp_path, monkeypatch)
    write_rows(args[1]/'work-family-groups.jsonl', [])
    write_json(args[1]/'checksums.json', {p.relative_to(args[1]).as_posix(): digest(p)
        for p in args[1].rglob('*') if p.is_file() and p.name != 'checksums.json'})
    with pytest.raises(ValueError, match='groups do not reproduce'):
        module.gate(*args)
    assert not args[-1].exists()


def test_failed_replay_does_not_create_output(tmp_path, monkeypatch):
    args = package_fixture(tmp_path, monkeypatch)
    def failed(*args):
        raise ValueError('Raw review does not reproduce')
    monkeypatch.setattr(module, 'replay_package', failed)
    with pytest.raises(ValueError, match='does not reproduce'):
        module.gate(*args)
    assert not args[-1].exists()
