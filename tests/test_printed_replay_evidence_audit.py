import hashlib
import io
import json
import zipfile

import pytest

from training.audit_printed_replay_evidence import verify_archive


def archive(payload=b"{}", *, checksum=True, name="report.json", extra=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        info = zipfile.ZipInfo("placeholder")
        info.filename = name
        zipped.writestr(info, payload)
        zipped.writestr("checksums.json", json.dumps({name: hashlib.sha256(payload if checksum else b"bad").hexdigest()}))
        if extra:
            zipped.writestr("uncovered.txt", "extra")
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


def test_complete_archive_validates():
    with archive() as zipped:
        assert set(verify_archive(zipped)) == {"report.json"}


@pytest.mark.parametrize("name", ["../bad", "/absolute", "a\\b", "C:/bad"])
def test_unsafe_zip_members_rejected(name):
    with archive(name=name) as zipped, pytest.raises(ValueError, match="Unsafe"):
        verify_archive(zipped)


def test_checksum_and_coverage_rejected():
    with archive(checksum=False) as zipped, pytest.raises(ValueError, match="payload"):
        verify_archive(zipped)
    with archive(extra=True) as zipped, pytest.raises(ValueError, match="coverage"):
        verify_archive(zipped)
