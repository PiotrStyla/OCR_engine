import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

import pytest

from training.probe_vision import load_cases, main, run_cases


def test_remote_import_works_without_site_packages():
    code = ('import sys; import training.probe_vision; '
            'assert not any(x in sys.modules for x in ("cv2", "torch", "numpy", "openai"))')
    subprocess.run([sys.executable, '-S', '-c', code],
                   cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)


def test_public_engine_exports_remain_available():
    from ocr import OcrEngine, recognize
    from ocr.pipeline import OcrEngine as ActualEngine, recognize as actual_recognize
    assert OcrEngine is ActualEngine and recognize is actual_recognize


def fixture(tmp_path):
    # Header is sufficient for preflight; no image decoding or remote request here.
    image = tmp_path/'page.png'
    image.write_bytes(b'\x89PNG\r\n\x1a\nfixture')
    row = {'id': 'page-1', 'image': image.name,
           'sha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'text': 'SECRET REFERENCE'}
    manifest = tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps(row)+'\n', encoding='utf-8')
    return manifest, image


def test_dry_run_has_no_reference_or_writes(tmp_path, capsys, monkeypatch):
    manifest, _ = fixture(tmp_path)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    output = tmp_path/'output'
    main(['--manifest',str(manifest),'--model','test-model','--output',str(output)])
    plan = json.loads(capsys.readouterr().out)
    assert plan['mode'] == 'dry-run' and plan['max_requests'] == 1
    assert 'SECRET REFERENCE' not in json.dumps(plan)
    assert not output.exists()


def test_checksum_and_duplicate_rejected(tmp_path):
    manifest, image = fixture(tmp_path)
    original = manifest.read_text()
    manifest.write_text(original+original)
    with pytest.raises(ValueError, match='unique'):
        load_cases(manifest,1)
    manifest.write_text(original)
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        load_cases(manifest,1)


def test_error_is_sanitized_and_stops_requests(tmp_path):
    manifest, _ = fixture(tmp_path)
    cases = load_cases(manifest,1)
    calls = []
    def parse(path):
        calls.append(path)
        raise RuntimeError('credential-should-never-appear')
    run_cases(cases*2,SimpleNamespace(parse=parse),tmp_path)
    saved = (tmp_path/'predictions.jsonl').read_text()
    assert len(calls) == 1
    assert 'credential-should-never-appear' not in saved
    assert json.loads(saved)['status'] == 'error'


def test_only_image_path_sent_and_empty_output_retained(tmp_path):
    manifest, _ = fixture(tmp_path)
    cases = load_cases(manifest,1)
    def parse(path):
        assert path == cases[0]['path']
        return SimpleNamespace(to_dict=lambda: {'text':'', 'source_sha256':cases[0]['sha256']})
    run_cases(cases,SimpleNamespace(parse=parse),tmp_path)
    row = json.loads((tmp_path/'predictions.jsonl').read_text())
    assert row['status'] == 'ok' and row['empty_output']
    assert 'SECRET REFERENCE' not in json.dumps(row)
