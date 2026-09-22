"""Zero-shot/API baseline for PolOCRBench subtasks A/B/C with the frozen prompt.

Sends the page image with the organizer prompt
(``benchmarks/polocrbench/prompts/zero_shot_prompt_v1.md``, per-subtask
templates extracted verbatim; the file hash must match the pinned registry) to
any OpenAI-compatible chat endpoint (Gemini's OpenAI shim, OpenRouter, OpenAI).
Per-page latency and token usage are recorded so ``run.json`` carries the
cost-per-page report.

Manifests/predictions follow the subtask formats (A ``text``, B ``html``,
C ``fields``). For C the template's ``{doc_type}`` and ``{fields}`` placeholders
come from the frozen schema (``training.kie_eval.SCHEMAS``). Responses are
unwrapped mechanically only (markdown fences dropped, B keeps the first
``<table>...</table>``, C parses the first JSON object) — never content-fixed,
per the zero-shot track rules.

Dry run by default (standard library only): validates the manifest and prints
the plan. ``--execute`` performs network calls and needs an endpoint key in
``GEMINI_API_KEY``, ``OPENAI_API_KEY`` or ``OPENROUTER_API_KEY``.

Usage:
  python -m training.run_vision_baseline --subtask C --manifest manifest-C.jsonl \
      --output runs/api-c --model openai/gpt-4o-mini \
      --base-url https://openrouter.ai/api/v1 --execute
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from training.stage_impact_benchmark import digest
from training.validate_submission import load_jsonl, unique_ids

PROTOCOL_VERSION = 'polocrbench-vision-baseline-v1'
PROMPT_VERSION = 'polocrbench-zero-shot-prompt-v1'
PROMPT_FILE = (Path(__file__).resolve().parent.parent / 'benchmarks' / 'polocrbench'
               / 'prompts' / 'zero_shot_prompt_v1.md')
PAYLOAD_KEYS = {'A': 'text', 'B': 'html', 'C': 'fields'}
_FENCE = re.compile(r'^```[a-zA-Z]*\n(.*?)\n?```$', re.S)
_KEY_ENVS = ('GEMINI_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY')
_URL_KEYS = (('generativelanguage.googleapis.com', 'GEMINI_API_KEY'),
             ('openrouter.ai', 'OPENROUTER_API_KEY'),
             ('openai.com', 'OPENAI_API_KEY'))


def load_templates(path=PROMPT_FILE):
    """Frozen prompt templates per subtask; refuses unpinned prompt files."""
    from training.validate_submission import ZERO_SHOT_PROMPTS
    data = Path(path).read_bytes()
    if ZERO_SHOT_PROMPTS.get(PROMPT_VERSION) != hashlib.sha256(data).hexdigest():
        raise ValueError('Prompt file does not match the pinned zero-shot registry')
    text = data.decode('utf-8')
    templates = {}
    for subtask in 'ABC':
        section = text.split(f'## {subtask} ', 1)[1].split('\n## ', 1)[0]
        match = re.search(r'```text\n(.*?)```', section, re.S)
        if not match:
            raise ValueError(f'No prompt template for subtask {subtask}')
        templates[subtask] = match.group(1).strip()
    return templates


def render_prompt(subtask, template, row):
    if subtask != 'C':
        return template
    from training.kie_eval import SCHEMAS
    schema = SCHEMAS[row['doc_type']]
    fields = ', '.join(f'{name}: {kind}' for name, kind in schema.items())
    prompt = template.replace('{doc_type}', row['doc_type']).replace('{fields}', fields)
    if '{doc_type}' in prompt or '{fields}' in prompt:
        raise ValueError('KIE prompt placeholders left unfilled')
    return prompt


def extract_tables(content):
    """All <table>...</table> blocks of a response, in document order."""
    lowered = content.lower()
    tables, offset = [], 0
    while True:
        start = lowered.find('<table', offset)
        if start < 0:
            return tables
        end = lowered.find('</table>', start)
        if end < 0:
            return tables
        tables.append(content[start:end + len('</table>')])
        offset = end + len('</table>')


def extract_payload(subtask, content, slot=0):
    """Mechanical unwrapping only: fences, tables by slot, first JSON object.

    For B the response keeps every table it returns; a manifest slot takes the
    table at that position, or the first one when fewer were returned.
    """
    text = (content or '').strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    if subtask == 'C':
        start, end = text.find('{'), text.rfind('}')
        if start < 0 or end <= start:
            raise ValueError('No JSON object in KIE response')
        fields = json.loads(text[start:end + 1])
        if not isinstance(fields, dict):
            raise ValueError('KIE response must be a JSON object')
        return fields
    if subtask == 'B':
        tables = extract_tables(text)
        if not tables:
            return text
        return tables[slot] if slot < len(tables) else tables[0]
    return text


def resolve_api_key(base_url, env=None):
    """Key matching the endpoint host first, then any available one."""
    env = os.environ if env is None else env
    preferred = [name for host, name in _URL_KEYS if host in base_url]
    ordered = preferred + [name for name in _KEY_ENVS if name not in preferred]
    for name in ordered:
        if env.get(name):
            return env[name], name
    raise ValueError(f'No API key in environment ({", ".join(_KEY_ENVS)})')


def create_client(base_url, api_key):
    from openai import OpenAI  # optional dependency: ocr-engine[correct]
    return OpenAI(base_url=base_url, api_key=api_key)


def send_page(client, model, subtask, image, template, row, max_tokens):
    from ocr.page_parser import image_message  # lazy: ocr package pulls torch
    messages = [{'role': 'user', 'content': image_message(image, render_prompt(subtask, template, row))}]
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=max_tokens)
    elapsed = time.perf_counter() - start
    choice = response.choices[0]
    if choice.finish_reason != 'stop':
        raise ValueError(f'Incomplete response: finish_reason={choice.finish_reason}')
    if choice.message.content is None:
        raise ValueError('Endpoint returned no content')
    usage = response.usage.model_dump() if response.usage is not None else None
    slot = int(row.get('table_index', 0) or 0)
    return extract_payload(subtask, choice.message.content, slot=slot), elapsed, usage


def load_cases(manifest, limit=None):
    manifest = Path(manifest)
    rows = load_jsonl(manifest)
    if not rows:
        raise ValueError('Empty manifest')
    unique_ids(rows, 'Manifest')
    cases = []
    for row in rows[:limit]:
        path = (manifest.parent / row['image']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Image checksum mismatch: {row["id"]}')
        cases.append({'id': row['id'], 'path': path, 'row': row})
    return cases


def _with_backoff(send, retries=5, base_sleep=4.0):
    for attempt in range(retries + 1):
        try:
            return send()
        except Exception as error:  # noqa: BLE001 - retried only when transient
            transient = getattr(error, 'status_code', None) in (429, 503)
            if not transient or attempt == retries:
                raise
            time.sleep(base_sleep * (2 ** attempt))


def run_cases(subtask, cases, send, output, metadata, retries=5, base_sleep=4.0):
    """Attempt every case; failures become error rows and the run continues."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload_key = PAYLOAD_KEYS[subtask]
    run = {**metadata, 'started_utc': datetime.now(timezone.utc).isoformat(),
           'pages_expected': len(cases), 'pages_completed': 0, 'error_pages': 0,
           'state': 'running'}

    def save_state():
        temporary = output / 'run.json.tmp'
        temporary.write_text(json.dumps(run, indent=2), encoding='utf-8', newline='\n')
        temporary.replace(output / 'run.json')

    save_state()
    elapsed_total, elapsed_max, timed, usage_total = 0.0, 0.0, 0, {}
    try:
        with (output / 'predictions.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for case in cases:
                row = {'id': case['id']}
                try:
                    payload, elapsed, usage = _with_backoff(
                        lambda: send(case), retries=retries, base_sleep=base_sleep)
                    row.update({'status': 'ok', payload_key: payload,
                                'elapsed_seconds': round(elapsed, 3)})
                    if usage:
                        row['usage'] = usage
                        for key, value in usage.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                usage_total[key] = usage_total.get(key, 0) + value
                    timed += 1
                    elapsed_total += elapsed
                    elapsed_max = max(elapsed_max, elapsed)
                    run['pages_completed'] += 1
                except Exception as error:  # noqa: BLE001 - each failure isolated
                    empty = {} if subtask == 'C' else ''
                    row.update({'status': 'error', payload_key: empty,
                                'error_type': type(error).__name__})
                    run['error_pages'] += 1
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
        run['state'] = 'completed'
    except BaseException:
        run['state'] = 'interrupted'
        raise
    finally:
        run['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if timed:
            run['cost'] = {'pages_timed': timed,
                           'mean_elapsed_seconds': round(elapsed_total / timed, 3),
                           'max_elapsed_seconds': round(elapsed_max, 3),
                           'total_elapsed_seconds': round(elapsed_total, 3),
                           'tokens': {key: int(value) for key, value in sorted(usage_total.items())}}
        save_state()
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subtask', choices=sorted(PAYLOAD_KEYS), required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--base-url', default='https://openrouter.ai/api/v1')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sleep', type=float, default=4.0)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cases = load_cases(args.manifest, args.limit)
    templates = load_templates()
    plan = {'protocol_version': PROTOCOL_VERSION, 'prompt_version': PROMPT_VERSION,
            'prompt_sha256': digest(PROMPT_FILE), 'manifest_sha256': digest(args.manifest),
            'subtask': args.subtask, 'pages': len(cases),
            'model': args.model, 'base_url': args.base_url}
    if not args.execute:
        print(json.dumps({**plan, 'state': 'preflight-only'}, indent=2))
        return
    api_key, key_env = resolve_api_key(args.base_url)
    client = create_client(args.base_url, api_key)
    template = templates[args.subtask]

    def send(case):
        return send_page(client, args.model, args.subtask, case['path'], template,
                         case['row'], args.max_tokens)

    run = run_cases(args.subtask, cases, send, args.output,
                    {**plan, 'key_env': key_env}, base_sleep=args.sleep)
    print(json.dumps({key: run[key] for key in
                      ('state', 'pages_completed', 'error_pages', 'cost')
                      if key in run}, indent=2))


if __name__ == '__main__':
    main()
