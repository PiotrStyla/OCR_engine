import hashlib
import io
import json
import struct
import zipfile

import pytest
from training.audit_reviewed_recognizer_result import paired_metrics, verify_zip, weights_digest


def package(extra=False, corrupt=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('data.json', b'bad' if corrupt else b'{}')
        archive.writestr('checksums.json', json.dumps({'data.json': hashlib.sha256(b'{}').hexdigest()}))
        if extra:
            archive.writestr('extra.json', b'{}')
    return zipfile.ZipFile(io.BytesIO(buffer.getvalue()))


def test_exact_checksum_coverage():
    with package() as archive:
        sums, weights = verify_zip(archive)
    assert set(sums) == {'data.json'}
    assert weights is None


@pytest.mark.parametrize('kind', ['extra', 'corrupt'])
def test_archive_changes_rejected(kind):
    with package(extra=kind == 'extra', corrupt=kind == 'corrupt') as archive:
        with pytest.raises(ValueError):
            verify_zip(archive)


def test_weights_are_finite_full_and_shape_checked():
    raw = json.dumps({'decoder.weight': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}).encode()
    raw += b' ' * (-len(raw) % 8)
    payload = struct.pack('<Q', len(raw)) + raw + struct.pack('<ff', 1., 2.)
    sha, stats = weights_digest(io.BytesIO(payload), len(payload))
    assert sha == hashlib.sha256(payload).hexdigest()
    assert stats['parameters'] == 2 and stats['all_weights_finite']
    broken = payload[:-4] + struct.pack('<f', float('nan'))
    with pytest.raises(ValueError, match='Non-finite'):
        weights_digest(io.BytesIO(broken), len(broken))


def test_historical_glyphs_are_not_modernized():
    baseline = [{'domain': 'history', 'id': 'a', 'reference': '\u017f\u00e1', 'prediction': 'sa'}]
    candidate = [{**baseline[0], 'prediction': '\u017f\u00e1'}]
    metrics, combined, lines = paired_metrics(baseline, candidate)
    assert metrics['history']['baseline']['cer'] == 1.
    assert combined['candidate'] == 0.
    assert lines[0]['reference'] == '\u017f\u00e1'


def test_generated_replacement_character_is_reported_not_removed():
    baseline = [{'domain': 'print', 'id': 'a', 'reference': 'abc', 'prediction': 'abc'}]
    candidate = [{**baseline[0], 'prediction': 'abc\ufffd'}]
    metrics, _, lines = paired_metrics(baseline, candidate)
    assert metrics['print']['candidate']['replacement_characters'] == 1
    assert lines[0]['candidate'].endswith('\ufffd')


@pytest.mark.parametrize('mutation', ['duplicate', 'id', 'reference'])
def test_paired_predictions_cannot_change_reference_or_identity(mutation):
    baseline = [{'domain': 'history', 'id': 'a', 'reference': 'abc', 'prediction': 'abc'}]
    candidate = [dict(baseline[0])]
    if mutation == 'duplicate':
        candidate *= 2
    elif mutation == 'id':
        candidate[0]['id'] = 'b'
    else:
        candidate[0]['reference'] = 'changed'
    with pytest.raises(ValueError):
        paired_metrics(baseline, candidate)
