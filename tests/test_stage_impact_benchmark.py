import hashlib
import io
import json
import tarfile

import pytest

from training.stage_impact_benchmark import digest, stage


def fixture(tmp_path, *, corrupt=False, link=False):
    data = b'image fixture'
    archive = tmp_path / 'test.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        member = tarfile.TarInfo('impact-print-v2/../impact-corpus/pages/images/BOOK__1.jpg')
        member.size = len(data)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = '/outside'
        bundle.addfile(member, None if link else io.BytesIO(data))
    manifest = tmp_path / 'frozen.jsonl'
    row = {'id': 'BOOK__1', 'image': '..\\old\\BOOK__1.jpg', 'text': 'text',
           'sha256': hashlib.sha256(b'wrong' if corrupt else data).hexdigest()}
    manifest.write_text(json.dumps(row), encoding='utf-8')
    train = tmp_path / 'train.jsonl'
    train.write_text('', encoding='utf-8')
    return archive, manifest, train, tmp_path / 'staged'


def test_portable_staging_preserves_frozen_evidence(tmp_path):
    args = fixture(tmp_path)
    original = args[1].read_bytes()
    report = stage(*args, expected_sha256=digest(args[0]))
    row = json.loads((args[3] / 'manifest.jsonl').read_text())
    assert row['image'] == 'images/BOOK__1.jpg'
    assert digest(args[3] / row['image']) == row['sha256']
    assert args[1].read_bytes() == original
    assert report['pages_verified'] == 1
    with pytest.raises(FileExistsError):
        stage(*args, expected_sha256=digest(args[0]))


@pytest.mark.parametrize('option', ['corrupt', 'link'])
def test_rejects_invalid_images_before_writing(tmp_path, option):
    args = fixture(tmp_path, **{option: True})
    with pytest.raises(ValueError):
        stage(*args, expected_sha256=digest(args[0]))
    assert not args[3].exists()


def test_rejects_wrong_archive_hash(tmp_path):
    args = fixture(tmp_path)
    with pytest.raises(ValueError, match='Archive checksum'):
        stage(*args, expected_sha256='wrong')


@pytest.mark.parametrize('overlap', ['collection', 'hash'])
def test_rejects_train_test_overlap(tmp_path, overlap):
    args = fixture(tmp_path)
    row = json.loads(args[1].read_text())
    args[2].write_text(json.dumps({
        'id': 'BOOK__2' if overlap == 'collection' else 'OTHER__1',
        'sha256': row['sha256'] if overlap == 'hash' else 'different',
    }))
    with pytest.raises(ValueError, match='overlap'):
        stage(*args, expected_sha256=digest(args[0]))
    assert not args[3].exists()
