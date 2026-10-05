from copy import deepcopy
import io
import json
from urllib.error import URLError

import pytest

from training import audit_recognizer_work_groups as module
from training.full_page_pilot import digest, read_rows, write_json, write_rows


def pages():
    return [{'id': c, 'collection': c, 'audit_split': split, 'image_sha256': c * 64,
             'source_page_url': 'https://example.org/' + c, 'dataset': 'dataset', 'revision': 'revision'}
            for c, split in [('a', 'train'), ('b', 'test'), ('c', 'train')]]


def group():
    return {'id': 'family', 'collections': ['a', 'b'], 'status': 'protective-work-family-group',
            'bibliographic_identity_verified': False, 'exact_scan_duplicate_claim': False,
            'visual_evidence_page_ids': ['a', 'b'], 'source_ids': ['catalog']}


def line(identifier='line', collection='a'):
    return {'id': identifier, 'page_id': collection, 'collection': collection,
            'dataset': 'dataset', 'revision': 'revision', 'source_split': 'train',
            'sha256': '0' * 64, 'eligible_for_training': True,
            'eligible_for_evaluation': False, 'final_test': False}


def seal(root):
    write_json(root/'checksums.json', {p.relative_to(root).as_posix(): digest(p)
               for p in root.rglob('*') if p.is_file() and p.name != 'checksums.json'})


def fixture(tmp_path):
    source, pool, expansion = [tmp_path/n for n in ('source', 'pool', 'expansion')]
    for root in (source, pool, expansion/'input'):
        root.mkdir(parents=True)
    write_rows(source/'page-signatures.jsonl', pages())
    write_json(source/'audit-report.json', {'pages': 3, 'dataset': 'dataset', 'revision': 'revision'})
    write_rows(pool/'manifest.jsonl', [line()])
    write_rows(expansion/'input/manifest.jsonl', [line('new-line', 'c')])
    for root in (source, pool, expansion):
        seal(root)
    files = {'page-signatures.jsonl': source/'page-signatures.jsonl',
             'source-audit-report.json': source/'audit-report.json',
             'reviewed-manifest.jsonl': pool/'manifest.jsonl',
             'expansion-manifest.jsonl': expansion/'input/manifest.jsonl'}
    policy = {'schema': 'slayer-recognizer-work-family-policy-v1', 'dataset': 'dataset',
              'revision': 'revision', 'pages': 3, 'input_sha256': {n: digest(p) for n, p in files.items()},
              'sources': [{'id': 'catalog', 'url': 'https://example.org/catalog', 'scope': 'title only'}],
              'groups': [group()]}
    write_json(tmp_path/'policy.json', policy)
    write_json(tmp_path/'sources.json', {'schema': 'slayer-bibliographic-source-access-v1',
        'sources': [{'source_id': 'catalog', 'url': 'https://example.org/catalog', 'scope': 'title only',
                     'identity_verified': False, 'status': 'unavailable', 'error_type': 'URLError'}]})
    return (source, tmp_path/'policy.json', tmp_path/'sources.json', pool, expansion, tmp_path/'output')


def test_disjoint_page_ids_do_not_clear_work_risk():
    result = module.validate_groups(pages(), [group()])
    assert result[0]['cross_split_risk'] is True
    assert result[0]['training_collections_to_quarantine'] == ['a']
    assert result[0]['bibliographic_identity_verified'] is False


@pytest.mark.parametrize('change', [
    {'collections': ['a', 'a']}, {'collections': ['a', 'missing']},
    {'visual_evidence_page_ids': ['a']}, {'visual_evidence_page_ids': ['a', 'c']},
    {'bibliographic_identity_verified': True}, {'exact_scan_duplicate_claim': True},
    {'status': 'human-verified'},
])
def test_invalid_group_or_false_certification_rejected(change):
    with pytest.raises(ValueError):
        module.validate_groups(pages(), [{**group(), **change}])


def test_overlapping_groups_and_duplicate_pages_rejected():
    with pytest.raises(ValueError):
        module.validate_groups(pages(), [group(), {**group(), 'id': 'another'}])
    with pytest.raises(ValueError):
        module.validate_groups(pages() + pages()[:1], [group()])


def test_same_split_family_does_not_generate_cross_split_quarantine():
    rows = pages()
    rows[1]['audit_split'] = 'train'
    result = module.validate_groups(rows, [group()])
    assert not result[0]['cross_split_risk']
    assert result[0]['training_collections_to_quarantine'] == []


def test_overlay_preserves_input_without_clearing_remaining_lines():
    rows = [line(), line('unresolved', 'c')]
    before = deepcopy(rows)
    groups = module.validate_groups(pages(), [group()])
    result = module.line_dispositions(rows, pages(), groups, batch='pool')
    assert rows == before
    assert [r['quarantine_for_future_training'] for r in result] == [True, False]
    assert all(r['eligible_for_training'] is False for r in result)
    assert result[1]['reason'] == 'bibliographic-identity-unresolved'


@pytest.mark.parametrize('change', [{'collection': 'c'}, {'dataset': 'other'},
    {'revision': 'other'}, {'source_split': 'test'}, {'eligible_for_evaluation': True},
    {'final_test': True}, {'page_id': 'missing'}, {'page_id': 'b', 'collection': 'b'}])
def test_line_source_and_heldout_boundary(change):
    with pytest.raises(ValueError):
        module.line_dispositions([{**line(), **change}], pages(), [], batch='pool')


def test_full_audit_is_deterministic_and_never_rewrites_labels(tmp_path):
    args = fixture(tmp_path)
    originals = [digest(args[3]/'manifest.jsonl'), digest(args[4]/'input/manifest.jsonl')]
    report = module.audit(*args)
    assert report['training_collections_to_quarantine'] == ['a']
    assert report['batch_impact'] == {'reviewed-pool': {'lines': 1, 'quarantined': 1},
                                     'expansion-review': {'lines': 1, 'quarantined': 0}}
    assert report['training_freeze_ready'] is False
    assert report['human_decisions_modified'] == 0
    assert originals == [digest(args[3]/'manifest.jsonl'), digest(args[4]/'input/manifest.jsonl')]
    assert module.audit(*args[:-1], tmp_path/'reproduced') == report
    assert digest(args[-1]/'line-dispositions.jsonl') == digest(tmp_path/'reproduced/line-dispositions.jsonl')
    with pytest.raises(FileExistsError):
        module.audit(*args)


def test_frozen_input_corruption_fails_before_output(tmp_path):
    args = fixture(tmp_path)
    write_rows(args[3]/'manifest.jsonl', [line('altered')])
    seal(args[3])
    with pytest.raises(ValueError, match='Frozen input'):
        module.audit(*args)
    assert not args[-1].exists()


def test_receipt_cannot_certify_identity(tmp_path):
    args = fixture(tmp_path)
    receipts = json.loads(args[2].read_text())
    receipts['sources'][0]['identity_verified'] = True
    write_json(args[2], receipts)
    with pytest.raises(ValueError, match='receipt identity'):
        module.audit(*args)


def test_duplicate_roots_between_batches_rejected(tmp_path):
    args = fixture(tmp_path)
    write_rows(args[4]/'input/manifest.jsonl', [{**line('new-line', 'c'), 'root_line_id': 'line'}])
    seal(args[4])
    policy = json.loads(args[1].read_text())
    policy['input_sha256']['expansion-manifest.jsonl'] = digest(args[4]/'input/manifest.jsonl')
    write_json(args[1], policy)
    with pytest.raises(ValueError, match='Duplicate root'):
        module.audit(*args)


def test_access_failure_is_recorded_without_disabling_tls(tmp_path):
    def failed(*args, **kwargs):
        assert kwargs['timeout'] == 15
        raise URLError('certificate expired')
    result = module.collect_sources({'sources': [{'id': 's', 'url': 'https://example.org', 'scope': 'catalog'}]},
                                    tmp_path/'receipt.json', opener=failed)
    assert result[0]['status'] == 'unavailable'
    assert result[0]['identity_verified'] is False
    assert 'sha256' not in result[0]


def test_access_success_records_digest_not_copied_content(tmp_path):
    class Response(io.BytesIO):
        status = 200
        def geturl(self):
            return 'https://example.org'
    result = module.collect_sources({'sources': [{'id': 's', 'url': 'https://example.org', 'scope': 'catalog'}]},
                                    tmp_path/'receipt.json', opener=lambda *a, **k: Response(b'catalog'))
    assert result[0]['bytes'] == 7 and len(result[0]['sha256']) == 64
    assert result[0]['identity_verified'] is False
    assert 'content' not in result[0]
