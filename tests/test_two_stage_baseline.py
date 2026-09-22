import hashlib
import json
from types import SimpleNamespace

import pytest

from training.run_two_stage_baseline import (
    _TEXT_TEMPLATES,
    build_cases,
    compose_message,
    load_text_sources,
    page_text_for,
    run_cases,
    send_text,
)
from training.validate_submission import validate


def test_text_sources_resolve_pages_and_table_slots(tmp_path):
    source = tmp_path / 'pred-a.jsonl'
    source.write_text('\n'.join(json.dumps(row) for row in [
        {'id': 'strona', 'status': 'ok', 'text': 'Ala ma kota'},
        {'id': 'padla', 'status': 'error', 'text': ''},
    ]), encoding='utf-8')
    texts = load_text_sources([source])
    assert texts == {'strona': 'Ala ma kota', 'padla': None}
    assert page_text_for({'id': 'strona'}, texts) == 'Ala ma kota'
    assert page_text_for({'id': 'strona__t0'}, texts) == 'Ala ma kota'
    assert page_text_for({'id': 'strona__t12'}, texts) == 'Ala ma kota'
    assert page_text_for({'id': 'obca__t0'}, texts) is None
    assert page_text_for({'id': 'padla'}, texts) is None


def test_compose_message_carries_transcript_and_filled_schema():
    message = compose_message('C', _TEXT_TEMPLATES['C'], {'doc_type': 'faktura'},
                              'Faktura FV/1, suma 1234,56 zł')
    assert 'Transkrypcja strony:\nFaktura FV/1, suma 1234,56 zł' in message
    assert 'Typ dokumentu: faktura' in message and 'invoice_number: text' in message
    assert '{doc_type}' not in message and '{fields}' not in message
    assert '{' not in compose_message('B', _TEXT_TEMPLATES['B'], {}, 'tekst')


class FakeUsage:
    def model_dump(self):
        return {'prompt_tokens': 50, 'completion_tokens': 10, 'total_tokens': 60}


class FakeClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason='stop',
                                     message=SimpleNamespace(content=self.contents.pop(0)))],
            usage=FakeUsage(), model='bielik-11b-v3')


def test_send_text_uses_string_content_and_slot_extraction():
    client = FakeClient(['<table><tr><td>a</td></tr></table> '
                         '<table><tr><td>b</td></tr></table>'])
    payload, elapsed, usage = send_text(client, 'bielik-11b-v3', 'B', _TEXT_TEMPLATES['B'],
                                        {'id': 's__t1', 'table_index': 1},
                                        'transkrypcja', 2048)
    assert payload == '<table><tr><td>b</td></tr></table>'
    assert elapsed >= 0 and usage['total_tokens'] == 60
    request = client.requests[0]
    assert request['temperature'] == 0
    content = request['messages'][0]['content']
    assert isinstance(content, str)  # Fabryka accepts string content only
    assert content.endswith('Transkrypcja strony:\ntranskrypcja')


def fixture(tmp_path, rows):
    image = tmp_path / 'page.png'
    image.write_bytes(b'fixture')
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps({'id': id_, 'image': image.name,
                                              'sha256': sha, **extra})
                                  for id_, extra in rows), encoding='utf-8')
    return manifest, image


def test_two_stage_run_scores_and_isolates_upstream_failures(tmp_path):
    manifest, image = fixture(tmp_path, [
        ('d1', {'doc_type': 'faktura', 'fields': {'invoice_number': 'FV/9'}}),
        ('d2', {'doc_type': 'faktura', 'fields': {'invoice_number': 'FV/2'}}),
    ])
    texts = {'d1': 'Faktura FV/9 ujmuje pole invoice_number: fv/9', 'd2': None}
    cases = build_cases(manifest, texts)
    assert [case['text'] for case in cases] == ['Faktura FV/9 ujmuje pole invoice_number: fv/9', None]
    client = FakeClient(['{"invoice_number": "fv/9"}'])

    def send(case):
        if case['text'] is None:
            raise RuntimeError('UpstreamOCR')
        return send_text(client, 'bielik-11b-v3', 'C', _TEXT_TEMPLATES['C'],
                         case['row'], case['text'], 2048)

    run = run_cases('C', cases, send, tmp_path / 'run', {'model': 'bielik-11b-v3'})
    assert run['pages_completed'] == 1 and run['error_pages'] == 1
    rows = [json.loads(line) for line in
            (tmp_path / 'run' / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows[0]['fields'] == {'invoice_number': 'fv/9'}  # casefold match with 'FV/9'
    assert rows[1]['error_type'] == 'RuntimeError'
    predictions = tmp_path / 'run' / 'predictions.jsonl'
    assert validate(manifest, predictions, subtask='C')['error_pages'] == 1
    from training.kie_eval import evaluate as kie_evaluate
    result = kie_evaluate(manifest, predictions)
    # one TP on d1, one FN on the failed page: F1 = 2/(2+0+1)
    assert result['f1_micro'] == pytest.approx(2 / 3)
