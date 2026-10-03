"""Paired, post-hoc resolution diagnostic; never load reference text in a worker."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

from training.benchmark_pages import evaluate
from training.full_page_comparison import read_rows, repetition_flags, write_rows
from training.full_page_pilot import digest, write_json


ENGINE = 'qwen3-vl-4b'
PAGE_IDS = ('Slawna_wiktoria_FT__437089', 'Wiesc_FT__436884', 'Choragiew_FT__436799')
ARMS = {'one-mp': 1048576, 'four-mp': 4194304}


def arm_configs(config):
    if tuple(row['id'] for row in config['selection']) != PAGE_IDS:
        raise ValueError('Use the frozen three-page diagnostic selection')
    if config['arms'] != ARMS or set(config['models']) != {ENGINE}:
        raise ValueError('Use exactly the frozen resolution arms and model')
    if config['models'][ENGINE]['max_pixels'] != ARMS['one-mp']:
        raise ValueError('Base model must retain the v5 one-megapixel budget')
    output = {}
    for label, pixels in ARMS.items():
        candidate = copy.deepcopy(config)
        candidate['models'][ENGINE]['max_pixels'] = pixels
        output[label] = candidate
    return output


def prepare_subset(config, dataset):
    arm_configs(config)
    dataset = Path(dataset)
    manifest = dataset/'manifest.jsonl'
    if digest(manifest) != config['dataset']['manifest_sha256']:
        raise ValueError('Full source manifest differs from the frozen v2 bundle')
    rows = {row['id']: row for row in read_rows(manifest)}
    if not set(PAGE_IDS) <= set(rows):
        raise ValueError('Selected pages missing from source bundle')
    selected = [rows[page_id] for page_id in PAGE_IDS]
    if any(row['split'] != 'validation' or row.get('eligible_for_training')
           for row in selected):
        raise ValueError('Diagnostic must use non-training validation pages')
    inputs = [{key: row[key] for key in ('id', 'image', 'sha256', 'width', 'height')}
              for row in selected]
    for row in inputs:
        row['source_regions'] = []
        if digest(dataset/row['image']) != row['sha256']:
            raise ValueError('Selected image checksum mismatch')
    write_rows(dataset/'resolution-manifest.jsonl', selected)
    write_rows(dataset/'resolution-inputs.jsonl', inputs)
    report = {'source_manifest_sha256': digest(manifest),
              'subset_manifest_sha256': digest(dataset/'resolution-manifest.jsonl'),
              'inference_inputs_sha256': digest(dataset/'resolution-inputs.jsonl'),
              'selection': config['selection'], 'pages': len(selected),
              'post_hoc_selection': True, 'reference_text_sent_to_model': False,
              'reference_geometry_sent_to_model': False, 'gold_pages': 0}
    write_json(dataset/'resolution-selection.json', report)
    return report


def compare(config, dataset, work):
    configs = arm_configs(config)
    dataset, work = Path(dataset), Path(work)
    manifest = dataset/'resolution-manifest.jsonl'
    references = read_rows(manifest)
    if tuple(row['id'] for row in references) != PAGE_IDS:
        raise ValueError('Subset page identity or order changed')
    source = {row['id']: row for row in read_rows(dataset/'manifest.jsonl')}
    if (digest(dataset/'manifest.jsonl') != config['dataset']['manifest_sha256']
            or references != [source[key] for key in PAGE_IDS]):
        raise ValueError('Subset references differ from the frozen source')
    scores = work/'scores'
    scores.mkdir(parents=True, exist_ok=True)
    reports, table = {}, []
    for label in ARMS:
        prediction_path = work/'arms'/label/'predictions'/f'{ENGINE}-full-page.jsonl'
        identity_path = prediction_path.parent/'identity.json'
        identity_verified = False
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding='utf-8'))
            if (identity['engine'] != ENGINE or identity['spec'] != configs[label]['models'][ENGINE]
                    or identity['input_sha256'] != digest(dataset/'resolution-inputs.jsonl')):
                raise ValueError('Worker identity differs from the frozen resolution arm')
            identity_verified = True
        if not prediction_path.exists():
            prediction_path = scores/f'{label}-missing.jsonl'
            write_rows(prediction_path, [])
        predictions = read_rows(prediction_path)
        if any(row.get('format', 'plain') != 'plain' for row in predictions):
            raise ValueError('Resolution experiment requires raw plain-text predictions')
        report = evaluate(manifest, prediction_path)
        supplied = {row['id']: row for row in predictions}
        report.update(cer_macro=sum(row['cer'] for row in report['results'])/len(references),
                      worker_identity_verified=identity_verified,
                      eos_pages=sum(row.get('finish_reason') == 'eos' for row in predictions),
                      token_limit_pages=sum(bool(row.get('token_limit_reached')) for row in predictions))
        for reference, result in zip(references, report['results']):
            prediction = supplied.get(reference['id'], {})
            text = prediction.get('text', '')
            trace = prediction.get('input_geometry', {})
            table.append({'arm': label, **result,
                'finish_reason': prediction.get('finish_reason', 'missing'),
                'token_limit_reached': prediction.get('token_limit_reached'),
                'generated_tokens': prediction.get('generated_tokens'),
                'elapsed_seconds': prediction.get('elapsed_seconds'),
                'peak_allocated_bytes': prediction.get('peak_allocated_bytes'),
                'input_geometry': trace,
                'reference_long_s_count': reference['text'].count('\u017f'),
                'raw_long_s_count': text.count('\u017f'),
                'reference_a_acute_count': reference['text'].count('\u00e1'),
                'raw_a_acute_count': text.count('\u00e1'),
                'reference_poslal_count': reference['text'].count('Po\u017f\u0142a\u0142'),
                'raw_poslal_count': text.count('Po\u017f\u0142a\u0142'),
                **repetition_flags(text)})
        reports[label] = report
    paired = []
    for page_id in PAGE_IDS:
        rows = {row['arm']: row for row in table if row['id'] == page_id}
        pixels = {arm: row['input_geometry'].get('processed_pixels') for arm, row in rows.items()}
        both_ok = all(row['status'] == 'ok' for row in rows.values())
        increased = (pixels['four-mp'] > pixels['one-mp']
                     if all(isinstance(value, int) for value in pixels.values()) else None)
        paired.append({'id': page_id, 'both_ok': both_ok, 'processed_pixels': pixels,
                       'processed_pixels_increased': increased,
                       'cer_delta_four_minus_one': rows['four-mp']['cer']-rows['one-mp']['cer']})
    summary = {'schema': 'slayer-full-page-resolution-v6-result', 'pages': len(references),
        'reports': reports, 'post_hoc_selection': True, 'all_pages_included': True,
        'paired': paired,
        'both_arms_rerun_requested': True, 'only_model_spec_change': 'max_pixels',
        'successful_page_inferences': sum(row['status'] == 'ok' for row in table),
        'reference_status': 'single-review-draft-not-gold', 'gold_pages': 0,
        'scope_policy': config['scope_policy'], 'sota_claim': False, 'model_promotion': False,
        'character_counts_are_recall': False,
        'claim_boundary': 'Three outcome-selected development pages, not a held-out benchmark. '
                          'No failed or capped outputs removed. Actual input grids must be checked. '
                          'Timing is sequential same-session diagnostics, not a performance benchmark.'}
    write_json(scores/'metrics.json', summary)
    write_rows(scores/'per-page.jsonl', table)
    with (scores/'per-page.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows([{**row, 'input_geometry': json.dumps(row['input_geometry'])}
                         for row in table])
    return summary
