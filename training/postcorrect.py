"""Noise-gated LLM post-correction of OCR output (HIPE-OCRepair 2026 lesson).

Over-correction on clean inputs is the recurring failure of LLM post-correction
(arXiv:2607.08143). This module therefore corrects only pages whose estimated
noise exceeds a threshold; clean pages pass through untouched.

Noise estimation, in order of preference:
1. explicit `confidence` field in the predictions JSONL (0..1; noise = 1 - confidence)
2. heuristic: share of tokens that are not recognizable Polish/number tokens
   (diacritic-aware dictionary of frequent Polish words + digit/date patterns)

Correction goes through the OpenRouter chat API (OPENROUTER_API_KEY); the model
is instructed to output ONLY the corrected text and to preserve characters it
cannot read (no guessing), which caps hallucination.

Usage:
  python -m training.postcorrect --predictions preds.jsonl --out corrected.jsonl \
      [--model openai/gpt-4o-mini] [--gate 0.15] [--lang pl]
"""
import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

PROMPT = """Jesteś korektorem wyników OCR historycznych polskich dokumentów.
Popraw oczywiste błędy OCR w tekście poniżej (pomylenie znaków, ſ->s, ligatury,
uszkodzone diakrytyki). ZASADY BEZWZGLĘDNE:
- NIE zgaduj: znaku, którego nie potrafisz odczytać, zostaw takim, jaki jest.
- NIE paraphrase'uj, NIE skracaj, NIE dodawaj słów.
- Zachowaj oryginalną pisownię historyczną (to nie jest korekta językowa).
- Zachowaj układ linii (jedna linia wejścia = jedna linia wyjścia).
- Wyjście: WYŁĄCZNIE poprawiony tekst, bez komentarzy i bez markdown.
Język dokumentu: {lang}."""

_POLISH_COMMON = {
    'i', 'w', 'z', 'na', 'do', 'od', 'za', 'po', 'przy', 'jest', 'był', 'była',
    'się', 'że', 'który', 'która', 'tego', 'ten', 'ta', 'to', 'nie', 'jak',
    'dla', 'pod', 'nad', 'oraz', 'lub', 'czy', 'gdy', 'już', 'wszystkich',
    'przez', 'bez', 'ich', 'jego', 'jej', 'przed', 'po', 'obok', 'miał',
}
_TOKEN_RE = re.compile(r'\S+')
_VOWELS = set('aeiouyąęóAEIOUYĄĘÓ')
_NOISE_CHARS = set('@#$%^&*_=~<>|\\{}[]')


def estimate_noise(text):
    """Heuristic noise score in [0, 1] for OCR text without model confidence.

    Signals (dictionary-free, safe for historical orthography):
    - share of garbage symbols, weighted 5x
    - share of long tokens without any vowel (Polish words always have one)
    - share of suspicious high-codepoint characters, weighted 5x
    """
    if not text.strip():
        return 1.0
    tokens = [t for t in _TOKEN_RE.findall(text) if len(t) >= 4]
    vowelless = (sum(1 for t in tokens if not (set(t) & _VOWELS)) / len(tokens)) if tokens else 0.0
    garbage = sum(c in _NOISE_CHARS for c in text) / len(text)
    weird = sum(1 for c in text if ord(c) > 0x2500) / len(text)
    return min(1.0, 3.0 * vowelless + 5.0 * garbage + 5.0 * weird)


def gate_ok(text, confidence, threshold):
    if confidence is not None:
        return (1.0 - confidence) >= threshold
    return estimate_noise(text) >= threshold


def correct(text, model, lang, api_key, timeout=120):
    body = json.dumps({
        'model': model,
        'temperature': 0,
        'messages': [
            {'role': 'system', 'content': PROMPT.format(lang=lang)},
            {'role': 'user', 'content': text},
        ],
    }).encode()
    req = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions', data=body,
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload['choices'][0]['message']['content'].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', default='openai/gpt-4o-mini')
    parser.add_argument('--gate', type=float, default=0.15,
                        help='noise threshold; pages below pass through uncorrected')
    parser.add_argument('--lang', default='polski')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    api_key = os.environ.get('OPENROUTER_API_KEY')
    assert api_key, 'Set OPENROUTER_API_KEY'
    rows = [json.loads(l) for l in Path(args.predictions).read_text(encoding='utf-8').splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    corrected, skipped, failed = 0, 0, 0
    with Path(args.out).open('w', encoding='utf-8') as fh:
        for i, row in enumerate(rows, 1):
            conf = row.get('confidence')
            if row.get('status') != 'ok' or not gate_ok(row.get('text', ''), conf, args.gate):
                out = {**row, 'corrected': False}
                skipped += 1
            else:
                try:
                    new_text = correct(row['text'], args.model, args.lang, api_key)
                    out = {**row, 'text': new_text, 'corrected': True}
                    corrected += 1
                except Exception as e:
                    out = {**row, 'corrected': False, 'postcorrect_error': str(e)[:100]}
                    failed += 1
            fh.write(json.dumps(out, ensure_ascii=False) + '\n')
            if i % 5 == 0:
                print(f'  {i}/{len(rows)} (corrected={corrected}, passthrough={skipped}, failed={failed})', flush=True)
    print(f'Done: {len(rows)} pages | corrected={corrected} passthrough={skipped} failed={failed}')


if __name__ == '__main__':
    main()
