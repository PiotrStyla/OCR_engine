import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from training.run_kraken_benchmark import load_cases, run_cases
from training.stage_impact_benchmark import digest
from training.validate_submission import validate, load_jsonl


def fixture(tmp_path):
    image = tmp_path / 'page.png'; image.write_bytes(b'fixture')
    manifest = tmp_path / 'manifest.jsonl'
    rows = [{'id': str(index), 'image': image.name, 'sha256': digest(image), 'text': 'SECRET GT'}
            for index in range(2)]
    manifest.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    return manifest


def predictions(tmp_path, rows):
    path = tmp_path / 'predictions.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    return path


def test_runner_captures_each_failure_without_reference_leakage(tmp_path):
    manifest = fixture(tmp_path)
    cases = load_cases(manifest)
    assert all(set(case) == {'id', 'path', 'sha256'} for case in cases)
    calls = []
    def predict(path):
        assert isinstance(path, Path)
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError('private exception detail')
        return 'recognized'
    output = tmp_path / 'run'
    report = run_cases(cases, predict, output, {'engine': 'mock'})
    assert report['pages_completed'] == 2 and report['error_pages'] == 1
    assert report['state'] == 'completed'
    assert 'private exception detail' not in (output / 'predictions.jsonl').read_text()
    assert validate(manifest, output / 'predictions.jsonl')['complete']
    with pytest.raises(FileExistsError):
        run_cases(cases, predict, output, {})


def test_interrupted_run_retains_progress_and_missing_pages(tmp_path):
    manifest = fixture(tmp_path)
    def predict(path):
        raise KeyboardInterrupt()
    output = tmp_path / 'run'
    with pytest.raises(KeyboardInterrupt):
        run_cases(load_cases(manifest), predict, output, {})
    assert json.loads((output / 'run.json').read_text())['state'] == 'interrupted'
    assert not validate(manifest, output / 'predictions.jsonl', allow_missing=True)['complete']


def test_changed_image_is_not_sent_to_predictor(tmp_path):
    cases = load_cases(fixture(tmp_path))
    cases[0]['path'].write_bytes(b'changed')
    def predict(path):
        pytest.fail('Changed image passed to predictor')
    report = run_cases(cases, predict, tmp_path / 'run', {})
    assert report['error_pages'] == 2


@pytest.mark.parametrize('rows,error', [
    ([{'id': '0', 'status': 'ok', 'text': ''}], 'Missing'),
    ([{'id': 'wrong', 'status': 'ok', 'text': ''}], 'unknown'),
    ([{'id': '0', 'status': 'ok', 'text': ''}] * 2, 'duplicate'),
    ([{'id': '0', 'status': 'ok', 'text': 7}], 'status/text'),
    ([{'id': '0', 'status': 'error', 'text': 'partial'}], 'empty text'),
    ([{'id': '0', 'status': 'ok', 'text': '', 'elapsed_seconds': -1}], 'elapsed'),
    ([{'id': '0', 'status': 'ok', 'text': '', 'elapsed_seconds': True}], 'elapsed'),
    ([{'id': '0', 'status': 'ok', 'text': '', 'source_sha256': 'wrong'}], 'hash'),
])
def test_invalid_submissions(tmp_path, rows, error):
    with pytest.raises(ValueError, match=error):
        validate(fixture(tmp_path), predictions(tmp_path, rows))


@pytest.mark.parametrize('content', ['{"id":"0","id":"1"}', '{"value":NaN}', '[]'])
def test_strict_json_records(tmp_path, content):
    path = tmp_path / 'bad.jsonl'; path.write_text(content)
    with pytest.raises(ValueError):
        load_jsonl(path)


def test_empty_successful_output_is_valid(tmp_path):
    manifest = fixture(tmp_path)
    path = predictions(tmp_path, [{'id': str(i), 'status': 'ok', 'text': ''} for i in range(2)])
    assert validate(manifest, path)['error_pages'] == 0


def test_preflight_needs_no_site_packages_or_models(tmp_path):
    manifest = fixture(tmp_path)
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-S', '-m', 'training.run_kraken_benchmark',
        '--manifest', str(manifest), '--output', str(tmp_path / 'preflight')],
        cwd=repo, env={**os.environ, 'PYTHONPATH': str(repo)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['state'] == 'preflight-only'
