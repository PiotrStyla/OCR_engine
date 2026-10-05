import io

import pytest

from training.audit_recognizer_source_splits import compare_pages, diplomatic_shingles
from training import audit_recognizer_source_splits as module
from training.full_page_pilot import digest, write_json, write_rows
from PIL import Image


def page(identifier, split, **kwargs):
    return {'id': identifier, 'audit_split': split, 'image_sha256': identifier,
        'document_id': identifier, 'edition_id': identifier, 'dhash': '0000000000000000',
        'width': 100, 'height': 200, 'text': '', **kwargs}


def test_diplomatic_signatures_preserve_historical_spelling():
    a = diplomatic_shingles('a b c d e f \u017f\u00e1\u0247')
    b = diplomatic_shingles('a b c d e f sa\u0247')
    assert a != b
    assert next(iter(a))[-1] == '\u017f\u00e1\u0247'


def test_same_split_not_compared_and_headers_not_text_leakage():
    flags, count = compare_pages([page('a', 'train'), page('b', 'train'),
                                  page('c', 'test', text='Repeated short heading')])
    assert count == 2
    assert len(flags) == 2
    assert all(f['reasons'] == ['perceptual-image-proposal'] for f in flags)
    assert all(f['proven_duplicate'] is False for f in flags)


def test_long_partial_text_overlap_is_review_proposal_not_proof():
    words = ' '.join(f'w{i}' for i in range(100))
    flags, count = compare_pages([page('a', 'train', text=words),
        page('b', 'test', text=' '.join(words.split()[10:40]), dhash='ffffffffffffffff')])
    assert count == 1 and flags[0]['reasons'] == ['long-text-overlap-proposal']
    assert flags[0]['text_containment'] == 1 and flags[0]['proven_duplicate'] is False


def test_exact_identity_conflicts_separate_from_heuristic_similarity():
    flags, _ = compare_pages([page('a', 'train'),
        page('b', 'test', image_sha256='a', document_id='a', edition_id='a')])
    assert set(flags[0]['reasons']) == {'exact-image', 'shared-source-document-id',
        'shared-digital-edition-id', 'perceptual-image-proposal'}


def test_aspect_ratio_guard_and_hamming_threshold():
    assert compare_pages([page('a', 'train'), page('b', 'test', width=150)])[0] == []
    assert compare_pages([page('a', 'train'), page('b', 'test', dhash='ffffffffffffffff')])[0] == []


@pytest.mark.parametrize('image,text', [(-1, .95), (65, .95), (4, 0), (4, 1.1)])
def test_invalid_thresholds_rejected(image, text):
    with pytest.raises(ValueError):
        compare_pages([], near_image_distance=image, near_text_similarity=text)


def fixture(tmp_path, monkeypatch):
    source = tmp_path/'metadata'
    specs = {}
    stream = io.BytesIO()
    Image.new('RGB', (20, 30), 'white').save(stream, 'JPEG')
    payload = stream.getvalue()
    import hashlib
    for i, split in enumerate(('train', 'validation', 'test')):
        (source/'pages'/split).mkdir(parents=True)
        path = source/'pages'/split/'metadata.jsonl'
        write_rows(path, [{'id': split, 'split': split, 'collection': split, 'width': 20, 'height': 30,
            'text': 'short title', 'edition_id': i, 'document_id': i, 'file_name': f'images/{split}.jpg',
            'source_xml_path': split+'.xml', 'image_sha256': hashlib.sha256(payload).hexdigest()}])
        specs[split] = (digest(path), 1)
    monkeypatch.setattr(module, 'PAGE_METADATA', specs)
    holdout = tmp_path/'holdout.json'
    write_json(holdout, {'dataset': module.DATASET, 'revision': module.REVISION,
                        'regions': [{'collection': 'geometry'}]})
    config = tmp_path/'config.json'
    write_json(config, {'dataset': {'forbidden_collections': ['geometry']}})
    return source, holdout, config, tmp_path/'audit', payload


def test_audit_verifies_all_bytes_but_does_not_claim_work_disjointness(tmp_path, monkeypatch):
    source, holdout, config, out, payload = fixture(tmp_path, monkeypatch)
    report = module.audit(source, holdout, config, out, opener=lambda *a, **k: io.BytesIO(payload))
    assert report['page_image_hashes_verified'] == 3
    assert report['cross_split_comparisons'] == report['overlap_proposals'] == 3
    assert report['work_identities_verified'] == 0 and report['training_freeze_ready'] is False
    assert report['external_final_test_corpora_audited'] is False
    with pytest.raises(FileExistsError):
        module.audit(source, holdout, config, out)


def test_bad_metadata_fails_before_download_or_output(tmp_path, monkeypatch):
    source, holdout, config, out, _ = fixture(tmp_path, monkeypatch)
    (source/'pages/test/metadata.jsonl').write_text('{}')
    with pytest.raises(ValueError, match='metadata mismatch'):
        module.audit(source, holdout, config, out)
    assert not out.exists()


def test_bad_scan_cannot_receive_verified_signature(tmp_path, monkeypatch):
    source, holdout, config, out, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='checksum mismatch'):
        module.audit(source, holdout, config, out, opener=lambda *a, **k: io.BytesIO(b'wrong'))
    assert not (out/'page-signatures.jsonl').exists()


@pytest.mark.parametrize('corrupt', [False, True])
def test_cached_images_are_rechecked_without_network(tmp_path, monkeypatch, corrupt):
    source, holdout, config, out, payload = fixture(tmp_path, monkeypatch)
    cache = tmp_path/'cache'
    cache.mkdir()
    for split in module.PAGE_METADATA:
        (cache/(split+'.jpg')).write_bytes(payload)
    def no_network(*a, **k):
        raise AssertionError('Cache must avoid network')
    if corrupt:
        (cache/'train.jpg').write_bytes(b'wrong')
        with pytest.raises(ValueError, match='Cached image checksum mismatch'):
            module.audit(source, holdout, config, out, opener=no_network, image_cache=cache)
    else:
        assert module.audit(source, holdout, config, out, opener=no_network, image_cache=cache)['page_image_hashes_verified'] == 3
