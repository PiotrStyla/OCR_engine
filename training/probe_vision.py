"""Bounded Gemini vision pilot. Default: local preflight, no network requests.

The API receives images and a fixed prompt, never reference transcriptions.
Use a free-tier project or confirm its billing before --execute.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from ocr.page_parser import PROMPT, RemotePageParser, image_message

BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'


def load_cases(manifest, limit):
    manifest = Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [r['id'] for r in rows]
    if not rows or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Manifest requires unique nonempty IDs')
    cases = []
    for row in rows[:limit]:
        path = (manifest.parent / row['image']).resolve()
        data = path.read_bytes()
        if len(data) > 10_000_000:
            raise ValueError('Image exceeds 10 MB limit')
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Image checksum mismatch: {row["id"]}')
        image_message(path)  # validate format before constructing any client
        cases.append({'id': row['id'], 'path': path, 'sha256': row['sha256']})
    return cases


def run_cases(cases, parser, output):
    """Stop after the first failure; unattempted pages remain missing for scoring."""
    for case in cases:
        started = time.perf_counter()
        row = {'id': case['id'], 'source_sha256': case['sha256']}
        try:
            # Recheck immediately before sending; never send ground truth.
            if hashlib.sha256(case['path'].read_bytes()).hexdigest() != case['sha256']:
                raise ValueError('Image changed after preflight')
            result = parser.parse(case['path']).to_dict()
            if result['source_sha256'] != case['sha256']:
                raise ValueError('Image changed during request')
            row.update(result)
            row['status'] = 'ok'
            row['empty_output'] = not result['text'].strip()
            # Empty output is valid on a genuinely blank page; scorer determines errors.
        except Exception as error:
            # Exception messages may contain response bodies, URLs or credentials.
            row.update(status='error', text='', error_type=type(error).__name__,
                       elapsed_seconds=time.perf_counter()-started)
        with (output/'predictions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(f'{case["id"]}: {row["status"]}')
        if row['status'] == 'error':
            break


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--limit', type=int, default=1)
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args(argv)
    if not 1 <= args.limit <= 5 or not 1 <= args.max_tokens <= 16384 or not args.model.strip():
        ap.error('Require limit 1..5, max-tokens 1..16384 and explicit model')
    cases = load_cases(args.manifest, args.limit)
    plan = {'mode': 'execute' if args.execute else 'dry-run', 'provider': 'gemini',
            'base_url': BASE_URL, 'model': args.model, 'pages': len(cases),
            'ids': [c['id'] for c in cases], 'max_requests': len(cases),
            'max_output_tokens_per_request': args.max_tokens, 'max_retries': 0,
            'timeout_seconds': 60, 'prompt': PROMPT,
            'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
            'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            'time_utc': datetime.now(timezone.utc).isoformat(),
            'references_sent': False, 'billing_verified': False}
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        ap.error('Set GEMINI_API_KEY in the process environment; do not put it in files or arguments')
    # Lazy import: dry-run needs only the standard library.
    from openai import OpenAI
    import httpx
    args.output.mkdir(parents=True, exist_ok=False)
    with OpenAI(api_key=key, base_url=BASE_URL, max_retries=0, timeout=60,
                http_client=httpx.Client(follow_redirects=False, timeout=60)) as client:
        (args.output/'run.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        run_cases(cases, RemotePageParser(client, args.model, args.max_tokens), args.output)


if __name__ == '__main__':
    main()
