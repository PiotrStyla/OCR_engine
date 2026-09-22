import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from training.benchmark_vision_impact import load_cases, main, run_cases


def fixture(tmp_path, pages=1):
    manifest_rows = []
    for i in range(pages):
        image = tmp_path / f'page{i}.png'
        image.write_bytes(b'\x89PNG\r\n\x1a\nfixture' + str(i).encode())
        manifest_rows.append({'id': f'page-{i}', 'image': image.name,
                              'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                              'text': 'SECRET REFERENCE'})
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in manifest_rows), encoding='utf-8')
    return manifest


def test_dry_run_has_no_reference_or_writes(tmp_path, capsys, monkeypatch):
    manifest = fixture(tmp_path, pages=3)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    output = tmp_path / 'output'
    main(['--manifest', str(manifest), '--model', 'test-model', '--output', str(output)])
    plan = json.loads(capsys.readouterr().out)
    assert plan['mode'] == 'dry-run' and plan['max_requests'] == 3
    assert plan['stops_on_error'] is False
    assert 'SECRET REFERENCE' not in json.dumps(plan)
    assert not output.exists()


def test_limit_zero_takes_all_pages(tmp_path):
    manifest = fixture(tmp_path, pages=4)
    assert len(load_cases(manifest, 0)) == 4
    assert len(load_cases(manifest, 2)) == 2


def test_checksum_and_duplicate_rejected(tmp_path):
    manifest = fixture(tmp_path)
    original = manifest.read_text()
    manifest.write_text(original + original)
    with pytest.raises(ValueError, match='unique'):
        load_cases(manifest, 0)
    manifest.write_text(original)
    (tmp_path / 'page0.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        load_cases(manifest, 0)


def test_run_attempts_all_pages_despite_errors(tmp_path):
    """Unlike the bounded pilot, one failure must not stop the remaining pages."""
    manifest = fixture(tmp_path, pages=3)
    cases = load_cases(manifest, 0)
    calls = []

    def parse(path):
        calls.append(path)
        raise RuntimeError('credential-should-never-appear')

    run_cases(cases, SimpleNamespace(parse=parse), tmp_path)
    saved = [json.loads(l) for l in (tmp_path / 'predictions.jsonl').read_text().splitlines()]
    assert len(calls) == 3
    assert all(r['status'] == 'error' for r in saved)
    assert 'credential-should-never-appear' not in json.dumps(saved)


def test_ok_output_recorded(tmp_path):
    manifest = fixture(tmp_path, pages=2)
    cases = load_cases(manifest, 0)

    def parse(path):
        return SimpleNamespace(to_dict=lambda: {'text': 'hello', 'source_sha256':
                               next(c['sha256'] for c in cases if c['path'] == path)})

    run_cases(cases, SimpleNamespace(parse=parse), tmp_path)
    rows = [json.loads(l) for l in (tmp_path / 'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and all(r['status'] == 'ok' for r in rows)
    assert 'SECRET REFERENCE' not in json.dumps(rows)
