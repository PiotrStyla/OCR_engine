"""Two-stage baseline: OCR text -> language model -> subtask B/C payloads.

Track ``open`` (any models, any pipeline). Stage 1 is any recognizer whose
per-page text already exists as subtask A predictions (Tesseract.js CPU
baseline, Surya, Qwen-VL, API — ``training.run_vision_baseline``); this runner
is stage 2: it sends the transcript to a text model (default: Fabryka AI,
``bielik-11b-v3``, OpenAI-compatible chat with STRING content only) and
extracts the subtask payload mechanically.

Prompts are the text variant ``polocrbench-two-stage-prompt-v1`` (the frozen
``zero_shot_prompt_v1`` is for direct image input in the zero-shot/API track).
For C the ``{doc_type}``/``{fields}`` placeholders come from the frozen
schemas, exactly like the vision runner.

Page text lookup order for a manifest row: exact ``id``, then ``id`` with a
trailing ``__t<index>`` table slot stripped (the multi-table-per-page
convention). A page whose OCR failed upstream becomes an ``error`` row
(``error_type: UpstreamOCR``) — the chain cannot extract what was never read.

Cost reporting per page is that of the second stage (see ``run.json``);
stage-1 costs live in its own run. Dry run by default (standard library only);
``--execute`` needs the endpoint key (``FABRYKA_API_KEY`` for fabryka.ai).

Usage:
  python -m training.run_two_stage_baseline --subtask C --manifest manifest-C.jsonl \
      --text-run runs/tesseract-A/predictions.jsonl --output runs/bielik-C --execute
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from training.run_vision_baseline import (
    PAYLOAD_KEYS,
    create_client,
    extract_payload,
    load_cases,
    render_prompt,
    run_cases,
)
from training.stage_impact_benchmark import digest

PROTOCOL_VERSION = 'polocrbench-two-stage-v1'
PROMPT_VERSION = 'polocrbench-two-stage-prompt-v1'
_SUBTASKS = ('B', 'C')
_SLOT_SUFFIX = re.compile(r'__t\d+$')
_TEXT_TEMPLATES = {
    'B': (
        'Poniżej znajduje się transkrypcja strony dokumentu, może zawierać błędy OCR. '
        'Na jej podstawie wypisz wszystkie tabele ze strony po kolei, każdą jako osobną '
        'tabelę HTML zachowującą strukturę wierszy i kolumn, użycie th/td oraz atrybuty '
        'colspan i rowspan, a w komórkach ich dokładną treść. Nie dodawaj klas ani '
        'stylów. Jeśli tabel brak, zwróć pusty ciąg. Odpowiedz wyłącznie kodem HTML.'
    ),
    'C': (
        'Poniżej znajduje się transkrypcja dokumentu, może zawierać błędy OCR. '
        'Wyodrębnij z niej pola zgodnie ze schematem. Typ dokumentu: {doc_type}. '
        'Pola: {fields}. Zwróć wyłącznie obiekt JSON z kluczami ze schematu i odczytanymi '
        'wartościami; pól brakowych nie umieszczaj. Nie zmyślaj wartości. Nie dodawaj '
        'komentarzy ani tekstu spoza JSON-a.'
    ),
}


def load_text_sources(paths):
    """id -> transcript from subtask A predictions or manifests (JSONL)."""
    from training.validate_submission import load_jsonl
    texts = {}
    for path in paths:
        for row in load_jsonl(Path(path)):
            value = row.get('text')
            if isinstance(value, str):
                texts[row['id']] = value if row.get('status', 'ok') == 'ok' else None
    return texts


def page_text_for(row, texts):
    """Transcript for a manifest row or None when its page failed upstream."""
    if row['id'] in texts:
        return texts[row['id']]
    page_id = _SLOT_SUFFIX.sub('', row['id'])
    return texts.get(page_id)


def compose_message(subtask, template, row, text):
    prompt = render_prompt(subtask, template, row) if subtask == 'C' else template
    return f'{prompt}\n\n---\nTranskrypcja strony:\n{text}'


def send_text(client, model, subtask, template, row, text, max_tokens):
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=0,
        messages=[{'role': 'user', 'content': compose_message(subtask, template, row, text)}])
    elapsed = time.perf_counter() - start
    choice = response.choices[0]
    if choice.finish_reason != 'stop':
        raise ValueError(f'Incomplete response: finish_reason={choice.finish_reason}')
    if choice.message.content is None:
        raise ValueError('Endpoint returned no content')
    usage = response.usage.model_dump() if response.usage is not None else None
    slot = int(row.get('table_index', 0) or 0)
    return extract_payload(subtask, choice.message.content, slot=slot), elapsed, usage


def build_cases(manifest, texts):
    """Shared load_cases rows plus their page transcripts (None = upstream error)."""
    cases = load_cases(manifest)
    for case in cases:
        case['text'] = page_text_for(case['row'], texts)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subtask', choices=_SUBTASKS, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--text-run', action='append', required=True,
                        help='subtask A predictions/manifest JSONL (repeatable)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='bielik-11b-v3')
    parser.add_argument('--base-url', default='https://fabryka.ai/v1')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--sleep', type=float, default=1.0)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    texts = load_text_sources(args.text_run)
    cases = build_cases(args.manifest, texts)
    plan = {'protocol_version': PROTOCOL_VERSION, 'prompt_version': PROMPT_VERSION,
            'manifest_sha256': digest(args.manifest),
            'text_runs': [digest(path) for path in args.text_run],
            'subtask': args.subtask, 'pages': len(cases),
            'pages_with_text': sum(case['text'] is not None for case in cases),
            'model': args.model, 'base_url': args.base_url}
    if not args.execute:
        print(json.dumps({**plan, 'state': 'preflight-only'}, indent=2))
        return
    from training.run_vision_baseline import resolve_api_key
    api_key, key_env = resolve_api_key(args.base_url)
    client = create_client(args.base_url, api_key)
    template = _TEXT_TEMPLATES[args.subtask]

    def send(case):
        if case['text'] is None:
            raise RuntimeError('UpstreamOCR')
        return send_text(client, args.model, args.subtask, template,
                         case['row'], case['text'], args.max_tokens)

    run = run_cases(args.subtask, cases, send, args.output,
                    {**plan, 'key_env': key_env}, base_sleep=args.sleep)
    print(json.dumps({key: run[key] for key in
                      ('state', 'pages_completed', 'error_pages', 'cost')
                      if key in run}, indent=2))


if __name__ == '__main__':
    main()
