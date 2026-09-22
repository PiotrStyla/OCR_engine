import hashlib
import json
from types import SimpleNamespace

import pytest

from training.kie_eval import SCHEMAS
from training.run_vision_baseline import (
    PROMPT_FILE,
    extract_payload,
    load_cases,
    load_templates,
    render_prompt,
    resolve_api_key,
    run_cases,
    send_page,
)
from training.validate_submission import validate


def test_frozen_prompt_templates_are_loaded_and_pinned():
    templates = load_templates()
    assert set(templates) == {'A', 'B', 'C'}
    assert 'Markdown' in templates['A'] and 'HTML' in templates['B']
    assert '{doc_type}' in templates['C'] and '{fields}' in templates['C']


def test_unpinned_prompt_file_is_refused(tmp_path):
    rogue = tmp_path / 'prompt.md'
    rogue.write_text(PROMPT_FILE.read_text(encoding='utf-8') + 'dopisek', encoding='utf-8')
    with pytest.raises(ValueError, match='registry'):
        load_templates(rogue)


def test_kie_prompt_fills_schema_placeholders():
    template = load_templates()['C']
    row = {'doc_type': 'faktura'}
    prompt = render_prompt('C', template, row)
    assert '{doc_type}' not in prompt and '{fields}' not in prompt
    assert 'Typ dokumentu:\nfaktura' in prompt
    for name, kind in SCHEMAS['faktura'].items():
        assert f'{name}: {kind}' in prompt


@pytest.mark.parametrize('content,expected', [
    ('```text\nAla ma kota\n```', 'Ala ma kota'),
    ('Ala\nma kota', 'Ala\nma kota'),
])
def test_extract_text(content, expected):
    assert extract_payload('A', content) == expected


def test_extract_html_keeps_first_table():
    content = 'Odpowiedź:\n```html\n<table><tr><td>a</td></tr></table>\n```\ntekst po'
    assert extract_payload('B', content) == '<table><tr><td>a</td></tr></table>'
    assert extract_payload('B', 'brak tabeli') == 'brak tabeli'


def test_extract_html_selects_slot_on_multi_table_pages():
    content = '<table><tr><td>pierwsza</td></tr></table> x ' \
              '<table><tr><td>druga</td></tr></table>'
    assert extract_payload('B', content, slot=1) == '<table><tr><td>druga</td></tr></table>'
    assert extract_payload('B', content, slot=5) == '<table><tr><td>pierwsza</td></tr></table>'


def test_extract_fields_parses_first_json_object():
    assert extract_payload('C', '```json\n{"invoice_number": "FV/1"}\n```') == \
        {'invoice_number': 'FV/1'}
    assert extract_payload('C', 'wynik: {"a": 1} koniec') == {'a': 1}
    with pytest.raises(ValueError, match='JSON object'):
        extract_payload('C', 'brak jsona')


class FakeUsage:
    def model_dump(self):
        return {'prompt_tokens': 100, 'completion_tokens': 40, 'total_tokens': 140}


class FakeClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        content = self.contents.pop(0)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason='stop',
                                     message=SimpleNamespace(content=content))],
            usage=FakeUsage(), model='fake-model')


def test_send_page_composes_frozen_prompt_and_usage():
    from ocr.page_parser import image_message
    client = FakeClient(['{"gross_total": "1 234,56 zł"}'])
    payload, elapsed, usage = send_page(
        client, 'fake-model', 'C', _image(), load_templates()['C'],
        {'doc_type': 'faktura'}, 2048)
    assert payload == {'gross_total': '1 234,56 zł'}
    assert elapsed >= 0 and usage['total_tokens'] == 140
    request = client.requests[0]
    assert request['temperature'] == 0 and request['max_tokens'] == 2048
    content = request['messages'][0]['content']
    assert content[0]['type'] == 'text'
    assert content[0]['text'] == render_prompt('C', load_templates()['C'], {'doc_type': 'faktura'})
    assert content == image_message(_image(), content[0]['text'])


def _image(tmp=None):
    from pathlib import Path
    import PIL.Image
    path = Path(__file__).resolve().parent / '_vision_fixture.png'
    if not path.exists():
        PIL.Image.new('RGB', (20, 20), 'white').save(path)
    return path


def fixture(tmp_path, subtask):
    image = tmp_path / 'page.png'
    image.write_bytes(b'fixture')
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / 'manifest.jsonl'
    if subtask == 'C':
        row = {'id': 'd1', 'image': image.name, 'sha256': sha, 'doc_type': 'faktura',
               'fields': {'invoice_number': 'FV/1'}}
    elif subtask == 'B':
        row = {'id': 't1', 'image': image.name, 'sha256': sha,
               'html': '<table><tr><td>x</td></tr></table>'}
    else:
        row = {'id': 'p1', 'image': image.name, 'sha256': sha, 'text': 'Ala'}
    manifest.write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')
    return manifest, image


def test_run_cases_records_costs_and_keeps_going(tmp_path):
    manifest, image = fixture(tmp_path, 'C')
    cases = [{'id': 'd1', 'path': image, 'row': {'doc_type': 'faktura'}},
             {'id': 'd2', 'path': image, 'row': {'doc_type': 'faktura'}}]
    sent = [({'invoice_number': 'FV/2'}, 1.5, {'prompt_tokens': 10, 'completion_tokens': 5,
                                              'total_tokens': 15}),
            (RuntimeError('boom'), None, None)]
    output = tmp_path / 'run'

    def send(case):
        payload, elapsed, usage = sent.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload, elapsed, usage

    run = run_cases('C', cases, send, output, {'model': 'fake'})
    assert run['state'] == 'completed'
    assert run['pages_completed'] == 1 and run['error_pages'] == 1
    assert run['cost'] == {'pages_timed': 1, 'mean_elapsed_seconds': 1.5,
                           'max_elapsed_seconds': 1.5, 'total_elapsed_seconds': 1.5,
                           'tokens': {'completion_tokens': 5, 'prompt_tokens': 10,
                                      'total_tokens': 15}}
    rows = [json.loads(line) for line in
            (output / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows[0]['status'] == 'ok' and rows[0]['fields'] == {'invoice_number': 'FV/2'}
    assert rows[0]['elapsed_seconds'] == 1.5
    assert rows[1] == {'id': 'd2', 'status': 'error', 'fields': {},
                      'error_type': 'RuntimeError'}
    with pytest.raises(FileExistsError):
        run_cases('C', cases, send, output, {})


def test_error_rows_validate_and_score_as_zero(tmp_path):
    manifest, image = fixture(tmp_path, 'B')
    output = tmp_path / 'run'

    def send(case):
        raise ValueError('brak odpowiedzi')

    run_cases('B', [{'id': 't1', 'path': image, 'row': {}}], send, output, {'model': 'fake'})
    predictions = output / 'predictions.jsonl'
    assert validate(manifest, predictions, subtask='B')['error_pages'] == 1
    from training.table_eval import evaluate as table_evaluate
    result = table_evaluate(manifest, predictions)
    assert result['teds_mean'] == 0.0 and result['errors_or_missing'] == 1


def test_transient_errors_are_retried(tmp_path):
    manifest, image = fixture(tmp_path, 'A')
    attempts = []

    class RateLimit(Exception):
        status_code = 429

    def send(case):
        attempts.append(1)
        if len(attempts) < 3:
            raise RateLimit('throttled')
        return 'Ala', 0.5, None

    run = run_cases('A', [{'id': 'p1', 'path': image, 'row': {}}], send,
                    tmp_path / 'run', {'model': 'fake'}, base_sleep=0.0)
    assert run['pages_completed'] == 1 and len(attempts) == 3


def test_api_key_resolution_matches_endpoint():
    env = {'OPENAI_API_KEY': 'o', 'OPENROUTER_API_KEY': 'r'}
    assert resolve_api_key('https://openrouter.ai/api/v1', env) == ('r', 'OPENROUTER_API_KEY')
    assert resolve_api_key('https://api.openai.com/v1', env) == ('o', 'OPENAI_API_KEY')
    assert resolve_api_key('https://generativelanguage.googleapis.com/v1beta/openai/',
                           {'GEMINI_API_KEY': 'g'}) == ('g', 'GEMINI_API_KEY')
    assert resolve_api_key('https://fabryka.ai/v1',
                           {'FABRYKA_API_KEY': 'f', 'OPENAI_API_KEY': 'o'}) == \
        ('f', 'FABRYKA_API_KEY')
    assert resolve_api_key('https://inny.host/v1', {'OPENAI_API_KEY': 'o'}) == ('o', 'OPENAI_API_KEY')
    with pytest.raises(ValueError, match='No API key'):
        resolve_api_key('https://openrouter.ai/api/v1', {})


def test_load_cases_checks_hashes(tmp_path):
    manifest, image = fixture(tmp_path, 'A')
    assert load_cases(manifest)[0]['id'] == 'p1'
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        load_cases(manifest)
