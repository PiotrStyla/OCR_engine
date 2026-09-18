"""PolOCRBench subtask A metric: full-page transcription (Markdown).

Deterministic, no LLM judges. Extends the frozen IMPACT-v2 protocol
(training/benchmark_pages.py, kept unchanged for reproducibility) with:

- quote normalization (typographic -> straight) before all comparisons,
- structural edit distance over the Markdown skeleton (headings, list items,
  paragraphs) using per-element text similarity, normalized to [0, 1].

Normalization (in order): Unicode NFC, typographic quotes -> straight,
whitespace collapse. Case and diacritics preserved.

Predictions JSONL: {id, text, status: ok|error, elapsed_seconds?}.
Usage: python -m training.transcription_eval --manifest M.jsonl --predictions P.jsonl --output R.json
"""
import argparse
import difflib
import hashlib
import json
import re
import unicodedata
from pathlib import Path

_QUOTE_MAP = str.maketrans({
    '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'",
    '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"',
    '\u00ab': '"', '\u00bb': '"', '\u2039': "'", '\u203a': "'",
})
_WS_RE = re.compile(r'\s+')
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$')
_LIST_RE = re.compile(r'^\s*(?:[-*+]\s+|\d+[.)]\s+)(.*)$')


def normalize(text):
    text = unicodedata.normalize('NFC', text).translate(_QUOTE_MAP)
    return _WS_RE.sub(' ', text).strip()


def _skeleton(markdown):
    """[(kind, normalized_text)] - h1..h6, li, p in document order."""
    skeleton = []
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            text = normalize(m.group(2).strip().lstrip('#').strip())
            if text:
                skeleton.append((f'h{level}', text))
            continue
        m = _LIST_RE.match(line)
        if m:
            text = normalize(m.group(1))
            if text:
                skeleton.append(('li', text))
            continue
        text = normalize(line.strip().strip('`'))
        if text:
            skeleton.append(('p', text))
    return skeleton


def _levenshtein_sim(ref, hyp):
    """Structural similarity in [0, 1]: Levenshtein over elements; substitution
    cost = 1 - text similarity (SequenceMatcher) when kinds match, else 1."""
    n, m = len(ref), len(hyp)
    if n == 0 and m == 0:
        return 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            k_ref, t_ref = ref[i - 1]
            k_hyp, t_hyp = hyp[j - 1]
            if k_ref == k_hyp:
                sub = 1.0 - difflib.SequenceMatcher(None, t_ref, t_hyp).ratio()
            else:
                sub = 1.0
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub)
        prev = cur
    dist = prev[m]
    return max(0.0, 1.0 - dist / max(n, m))


def structure_report(ref_md, hyp_md):
    ref, hyp = _skeleton(ref_md), _skeleton(hyp_md)
    sim = _levenshtein_sim(ref, hyp)
    kinds = {k for k, _ in ref} | {k for k, _ in hyp}
    detail = {}
    for k in sorted(kinds):
        r = [t for kk, t in ref if kk == k]
        h = [t for kk, t in hyp if kk == k]
        detail[k] = {'ref': len(r), 'hyp': len(h),
                     'sim': _levenshtein_sim([(k, t) for t in r], [(k, t) for t in h])}
    return {'structure_similarity': round(sim, 6),
            'elements': {'ref': len(ref), 'hyp': len(hyp)},
            'per_kind': detail}


def evaluate(manifest, predictions, with_structure=True):
    from jiwer import cer, wer
    manifest = Path(manifest)
    records = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Manifest must contain unique nonempty records')
    rows = [json.loads(line) for line in Path(predictions).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate predictions')
    supplied = {r['id']: r for r in rows}
    if set(supplied) - {r['id'] for r in records}:
        raise ValueError('Predictions contain unknown IDs')
    references, hypotheses, output = [], [], []
    for row in records:
        data = (manifest.parent / row['image']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Image checksum mismatch: {row["id"]}')
        reference = normalize(row['text'])
        prediction = supplied.get(row['id'])
        if prediction and prediction.get('status') not in ('ok', 'error'):
            raise ValueError('Prediction status must be ok or error')
        status = prediction['status'] if prediction else 'missing'
        hypothesis = normalize(prediction['text']) if status == 'ok' else ''
        references.append(reference)
        hypotheses.append(hypothesis)
        entry = {'id': row['id'], 'status': status, 'cer': cer(reference, hypothesis),
                 'wer': wer(reference, hypothesis), 'exact_match': reference == hypothesis}
        if with_structure and status == 'ok':
            entry['structure'] = structure_report(row['text'], prediction['text'])
        output.append(entry)
    result = {'pages': len(records), 'errors_or_missing': sum(e['status'] != 'ok' for e in output),
              'cer_micro': cer(references, hypotheses), 'wer_micro': wer(references, hypotheses),
              'normalization': 'NFC + typographic quotes -> straight + whitespace; case/diacritics preserved',
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'predictions_sha256': hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
              'results': output}
    if with_structure:
        sims = [e['structure']['structure_similarity'] for e in output if 'structure' in e]
        result['structure_similarity'] = sum(sims) / len(sims) if sims else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-structure', action='store_true')
    args = parser.parse_args()
    result = evaluate(args.manifest, args.predictions, with_structure=not args.no_structure)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {k: result[k] for k in ('pages', 'errors_or_missing', 'cer_micro', 'wer_micro') if k in result}
    if 'structure_similarity' in result:
        summary['structure_similarity'] = result['structure_similarity']
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
