"""Strict development-only selection; the unchanged baseline is always available."""
import math
import unicodedata

DOMAINS = ('historical-development', 'ordinary-development')


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def summarize(rows):
    from jiwer import cer, wer

    scores = {}
    identities = [(row['domain'], row['id']) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate development prediction identity')
    if set(row['domain'] for row in rows) != set(DOMAINS):
        raise ValueError('Both protected development domains are required')
    for domain in (*DOMAINS, 'combined'):
        subset = [row for row in rows if domain == 'combined' or row['domain'] == domain]
        refs = [normalize(row['reference']) for row in subset]
        hyps = [normalize(row['prediction']) for row in subset]
        if not refs or not all(refs):
            raise ValueError('Missing or empty development references')
        scores[domain] = dict(cer=cer(refs, hyps), wer=wer(refs, hyps), lines=len(refs),
                             replacement_characters=sum(text.count('\ufffd') for text in hyps))
    return scores


def eligibility(candidate, baseline):
    reasons = []
    for domain in (*DOMAINS, 'combined'):
        for metric in ('cer', 'wer'):
            if not math.isfinite(candidate[domain][metric]) or candidate[domain][metric] < 0:
                raise ValueError('Invalid candidate metric')
        if candidate[domain]['lines'] != baseline[domain]['lines']:
            raise ValueError('Development coverage drift')
        if candidate[domain]['replacement_characters'] > baseline[domain]['replacement_characters']:
            reasons.append(domain + ':additional-replacement-characters')
    if candidate[DOMAINS[0]]['cer'] >= baseline[DOMAINS[0]]['cer'] - 1e-12:
        reasons.append('historical:CER-not-improved')
    for metric in ('cer', 'wer'):
        if candidate[DOMAINS[1]][metric] > baseline[DOMAINS[1]][metric] + 1e-12:
            reasons.append('ordinary:' + metric.upper() + '-regression')
    if candidate['combined']['cer'] >= baseline['combined']['cer'] - 1e-12:
        reasons.append('combined:CER-not-improved')
    return dict(eligible=not reasons, reasons=reasons)


def select(baseline, candidates):
    judged = [dict(row, **eligibility(row['metrics'], baseline)) for row in candidates]
    eligible = [row for row in judged if row['eligible']]
    winner = min(eligible, key=lambda row: (row['metrics']['combined']['cer'], row['id'])) if eligible else None
    return dict(selected=winner['id'] if winner else 'unchanged-baseline', candidates=judged,
                baseline_retained=winner is None, selection_uses_development_data=True,
                independent_benchmark=False, production_promoted=False, sota_claim=False)


def checkpoint_metrics(tokenizer, references, baseline, output):
    """Use raw references in dataset order, rather than tokenizer-normalized ground truth."""
    import numpy as np
    from training.full_page_pilot import write_rows

    iteration = 0

    def compute(prediction):
        nonlocal iteration
        tokens = prediction.predictions
        if isinstance(tokens, tuple):
            tokens = tokens[0]
        tokens = np.where(tokens == -100, tokenizer.pad_token_id, tokens)
        labels = np.where(prediction.label_ids == -100, tokenizer.pad_token_id, prediction.label_ids)
        hyps = tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decoded = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if len(hyps) != len(references) or len(decoded) != len(references):
            raise ValueError('Checkpoint prediction coverage drift')
        if any(normalize(row['reference']) != normalize(text) for row, text in zip(references, decoded)):
            raise ValueError('Checkpoint label order or tokenizer round-trip drift')
        rows = [dict(row, prediction=hyp) for row, hyp in zip(references, hyps)]
        scores = summarize(rows)
        gate = eligibility(scores, baseline)
        iteration += 1
        write_rows(output / f'checkpoint-evaluation-{iteration:03d}.jsonl', rows)
        result = {'cer': scores['combined']['cer'], 'wer': scores['combined']['wer'],
                  'eligible': int(gate['eligible']),
                  'selection_score': scores['combined']['cer'] if gate['eligible'] else 1e6 + scores['combined']['cer']}
        for domain in DOMAINS:
            for metric in ('cer', 'wer', 'replacement_characters'):
                result[domain + '_' + metric] = scores[domain][metric]
        return result

    return compute
