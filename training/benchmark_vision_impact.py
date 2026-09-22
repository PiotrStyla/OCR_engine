"""Full-page vision benchmark on the frozen IMPACT-Polish v2 test (36 pages).

Unlike training.probe_vision (a bounded 1-5 page pilot that stops on first
error), this runs the complete frozen benchmark: it attempts every page,
records failures as empty hypotheses, and scores with the shared page scorer.

Reuses the same request path (ocr.page_parser.RemotePageParser + fixed PROMPT,
references never sent) and the same scorer (training.benchmark_pages.evaluate).

Two steps, so the network run is isolated from scoring:
  1. --execute writes predictions.jsonl (needs GEMINI_API_KEY in the env).
  2. scoring runs automatically against the staged manifest afterwards.

Staged manifest is produced by training.stage_impact_benchmark (id, image,
sha256, text with local image paths). Dry-run needs only the standard library.
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
TESSERACT_BASELINE = {'cer_micro': 0.3311, 'wer_micro': 0.8163}


def render_png(source, dest):
    """Re-encode any PIL-readable page (the frozen set is TIFF) to PNG for the vision API."""
    from PIL import Image
    with Image.open(source) as im:
        im.convert('RGB').save(dest, format='PNG')
    return dest


def load_cases(manifest, limit, render_dir=None):
    manifest = Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [r['id'] for r in rows]
    if not rows or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Manifest requires unique nonempty IDs')
    cases = []
    for row in (rows[:limit] if limit else rows):
        path = (manifest.parent / row['image']).resolve()
        data = path.read_bytes()
        if len(data) > 10_000_000:
            raise ValueError(f'Image exceeds 10 MB limit: {row["id"]}')
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Image checksum mismatch: {row["id"]}')
        # The frozen images are TIFF; the vision endpoint accepts PNG/JPEG only.
        # Send a PNG render but keep integrity anchored to the original sha256.
        # Rendering (and format validation) happens only on the execute path so
        # the dry-run plan stays stdlib-only.
        send = path
        if render_dir is not None:
            send = render_png(path, Path(render_dir) / f"{row['id']}.png")
            image_message(send)  # validate PNG/JPEG before constructing any client
        cases.append({'id': row['id'], 'path': path, 'sha256': row['sha256'], 'send': send})
    return cases


def _parse_with_backoff(parser, send, retries, base_sleep):
    """Retry only transient rate/availability errors (429/503); free-tier limits RPM."""
    transient = ('RateLimitError', 'InternalServerError', 'APITimeoutError', 'APIConnectionError')
    for attempt in range(retries + 1):
        try:
            return parser.parse(send)
        except Exception as error:
            if type(error).__name__ not in transient or attempt == retries:
                raise
            time.sleep(base_sleep * (2 ** attempt))


def run_cases(cases, parser, output, sleep=4.0, retries=5):
    """Attempt every page; a failure is recorded as an empty hypothesis and the run continues.

    A base delay plus exponential backoff keeps the run under free-tier RPM limits.
    """
    predictions = output / 'predictions.jsonl'
    for i, case in enumerate(cases):
        started = time.perf_counter()
        row = {'id': case['id'], 'source_sha256': case['sha256']}
        try:
            # Integrity is anchored to the original page; the sent PNG is a render of it.
            if hashlib.sha256(case['path'].read_bytes()).hexdigest() != case['sha256']:
                raise ValueError('Image changed after preflight')
            send = case.get('send', case['path'])
            result = _parse_with_backoff(parser, send, retries, sleep).to_dict()
            if hashlib.sha256(case['path'].read_bytes()).hexdigest() != case['sha256']:
                raise ValueError('Image changed during request')
            row.update(result)
            # Keep integrity anchored to the original page, not the sent PNG render.
            row['source_sha256'] = case['sha256']
            row['status'] = 'ok'
            row['empty_output'] = not result['text'].strip()
        except Exception as error:
            # Never persist exception text: it may carry response bodies or the key.
            row.update(status='error', text='', error_type=type(error).__name__,
                       elapsed_seconds=time.perf_counter() - started)
        with predictions.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'{case["id"]}: {row["status"]}')
        if sleep and i < len(cases) - 1:
            time.sleep(sleep)  # base spacing between requests
    return predictions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True, type=Path,
                    help='Staged manifest (id, image, sha256, text) from stage_impact_benchmark')
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--limit', type=int, default=0, help='0 = all pages (default)')
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--sleep', type=float, default=4.0,
                    help='Base seconds between requests (free-tier RPM throttle)')
    ap.add_argument('--retries', type=int, default=5,
                    help='Retries with exponential backoff on 429/503')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args(argv)
    if args.limit < 0 or not 1 <= args.max_tokens <= 16384 or not args.model.strip():
        ap.error('Require limit >= 0, max-tokens 1..16384 and explicit model')
    cases = load_cases(args.manifest, args.limit)  # dry-run: no render, stdlib only
    plan = {'mode': 'execute' if args.execute else 'dry-run', 'provider': 'gemini',
            'base_url': BASE_URL, 'model': args.model, 'pages': len(cases),
            'ids': [c['id'] for c in cases], 'max_requests': len(cases),
            'max_output_tokens_per_request': args.max_tokens, 'max_retries': 0,
            'timeout_seconds': 60, 'prompt': PROMPT,
            'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
            'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            'time_utc': datetime.now(timezone.utc).isoformat(),
            'references_sent': False, 'billing_verified': False,
            'stops_on_error': False, 'base_sleep_seconds': args.sleep,
            'retries_on_429_503': args.retries, 'tesseract_baseline': TESSERACT_BASELINE}
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        ap.error('Set GEMINI_API_KEY in the process environment; do not put it in files or arguments')
    from openai import OpenAI
    import httpx
    args.output.mkdir(parents=True, exist_ok=False)
    render_dir = args.output / 'render'
    render_dir.mkdir()
    # Re-load with rendering: converts each TIFF to PNG and validates the sent format.
    cases = load_cases(args.manifest, args.limit, render_dir=render_dir)
    (args.output / 'run.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    with OpenAI(api_key=key, base_url=BASE_URL, max_retries=0, timeout=60,
                http_client=httpx.Client(follow_redirects=False, timeout=60)) as client:
        predictions = run_cases(cases, RemotePageParser(client, args.model, args.max_tokens),
                                args.output, sleep=args.sleep, retries=args.retries)
    # Score against the same staged manifest with the shared page scorer.
    from training.benchmark_pages import evaluate
    summary = evaluate(args.manifest, predictions)
    summary['system'] = f'gemini vision ({args.model})'
    summary['split'] = 'impact-print-v2-test'
    summary['tesseract_baseline'] = TESSERACT_BASELINE
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n=== Gemini {args.model} on {summary['pages']} pages ===")
    print(f"  CER {summary['cer_micro']*100:.2f}% | WER {summary['wer_micro']*100:.2f}%"
          f" | errors/missing {summary['errors_or_missing']}")
    print(f"  Tesseract baseline: CER {TESSERACT_BASELINE['cer_micro']*100:.2f}%"
          f" | WER {TESSERACT_BASELINE['wer_micro']*100:.2f}%")


if __name__ == '__main__':
    main()
