import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from training.run_kraken_benchmark import load_cases, run_cases
from training.stage_impact_benchmark import digest
from training.validate_submission import ZERO_SHOT_PROMPTS, load_jsonl, validate, validate_meta


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


def fixture_bc(tmp_path, subtask):
    image = tmp_path / 'page.png'; image.write_bytes(b'fixture')
    manifest = tmp_path / 'manifest.jsonl'
    if subtask == 'B':
        row = {'id': '0', 'image': image.name, 'sha256': digest(image),
               'html': '<table><tr><td>x</td></tr></table>'}
    else:
        row = {'id': '0', 'image': image.name, 'sha256': digest(image),
               'doc_type': 'faktura', 'fields': {'invoice_number': 'FV/1'}}
    manifest.write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')
    return manifest


@pytest.mark.parametrize('subtask,row,error', [
    ('B', {'id': '0', 'status': 'ok', 'html': 7}, 'status/html'),
    ('B', {'id': '0', 'status': 'error', 'html': 'partial'}, 'empty html'),
    ('C', {'id': '0', 'status': 'ok', 'fields': []}, 'status/fields'),
    ('C', {'id': '0', 'status': 'error', 'fields': {'invoice_number': 'x'}}, 'empty fields'),
    ('C', {'id': '0', 'status': 'ok', 'fields': {'unknown': 'x'}}, 'outside the document schema'),
    ('C', {'id': '0', 'status': 'ok', 'fields': {'gross_total': True}}, 'field value'),
])
def test_invalid_submissions_per_subtask(tmp_path, subtask, row, error):
    with pytest.raises(ValueError, match=error):
        validate(fixture_bc(tmp_path, subtask), predictions(tmp_path, [row]), subtask=subtask)


@pytest.mark.parametrize('subtask,row', [
    ('B', {'id': '0', 'status': 'ok', 'html': '<table></table>'}),
    ('B', {'id': '0', 'status': 'error', 'html': ''}),
    ('C', {'id': '0', 'status': 'ok', 'fields': {'invoice_number': 'FV/2', 'gross_total': 1234.56}}),
    ('C', {'id': '0', 'status': 'ok', 'fields': {}}),
])
def test_valid_submissions_per_subtask(tmp_path, subtask, row):
    report = validate(fixture_bc(tmp_path, subtask), predictions(tmp_path, [row]), subtask=subtask)
    assert report['subtask'] == subtask
    assert report['eligible_for_scoring']


def test_manifest_payload_must_match_subtask(tmp_path):
    manifest = fixture_bc(tmp_path, 'C')
    row = {'id': '0', 'status': 'ok', 'fields': {'invoice_number': 'FV/1'}}
    with pytest.raises(ValueError, match='outside schema'):
        bad = tmp_path / 'bad.jsonl'
        data = json.loads(Path(manifest).read_text(encoding='utf-8'))
        data['doc_type'] = 'niedokument'
        bad.write_text(json.dumps(data), encoding='utf-8')
        validate(bad, predictions(tmp_path, [row]), subtask='C')


def meta(**overrides):
    value = {'schema': 'polocrbench-submission-meta-v1', 'team': 'Zespol OCR', 'track': 'open',
             'subtasks': ['A'], 'models': [{'name': 'model-x', 'open_weight': False}],
             'finetuned': False, 'training_data': []}
    value.update(overrides)
    return value


def zero_shot_meta(**overrides):
    return meta(track='zero-shot',
                prompt_version=next(iter(ZERO_SHOT_PROMPTS)),
                prompt_sha256=ZERO_SHOT_PROMPTS[next(iter(ZERO_SHOT_PROMPTS))],
                **overrides)


def test_track_declarations():
    assert validate_meta(meta())['track'] == 'open'
    constrained = meta(track='constrained', training_data=['organizer:polocrbench-train-v1'],
                       models=[{'name': 'surya', 'open_weight': True}])
    assert validate_meta(constrained)['open_weight']
    assert validate_meta(zero_shot_meta(models=[{'name': 'gemini', 'open_weight': False}]))['track'] == 'zero-shot'
    with pytest.raises(ValueError, match='Meta track'):
        validate_meta(meta(track='razem'))
    with pytest.raises(ValueError, match='organizer:'):
        validate_meta(meta(track='constrained', models=[{'name': 'm', 'open_weight': True}],
                           training_data=['prywatne-dane']))
    with pytest.raises(ValueError, match='open-weight'):
        validate_meta(meta(track='constrained', training_data=['organizer:polocrbench-train-v1']))
    with pytest.raises(ValueError, match='prompt version'):
        validate_meta(meta(track='zero-shot'))
    with pytest.raises(ValueError, match='forbids'):
        validate_meta(zero_shot_meta(finetuned=True))
    with pytest.raises(ValueError, match='subtasks'):
        validate_meta(meta(subtasks=['A', 'A']))
    with pytest.raises(ValueError, match='models'):
        validate_meta(meta(models=[{'name': 'm'}]))


def test_zero_shot_prompt_registry_matches_prompt_file():
    repo = Path(__file__).resolve().parents[1]
    prompt = repo / 'benchmarks' / 'polocrbench' / 'prompts' / 'zero_shot_prompt_v1.md'
    assert digest(prompt) == ZERO_SHOT_PROMPTS['polocrbench-zero-shot-prompt-v1']


def test_validate_reports_declared_track_metadata(tmp_path):
    manifest = fixture_bc(tmp_path, 'B')
    report = validate(manifest, predictions(tmp_path, [{'id': '0', 'status': 'ok', 'html': ''}]),
                      subtask='B', meta=meta(subtasks=['B']))
    assert report['meta']['subtasks'] == ['B']
    assert 'self-declared' in report['scope']
