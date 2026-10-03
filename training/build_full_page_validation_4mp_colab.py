"""Build the single no-upload full-page validation v7 Colab."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from training.build_full_page_comparison_colab import ROOT, CONFIG_PATH as V5_CONFIG, build as build_base
from training.build_full_page_pilot_colab import markdown
from training.build_historical_recognizer_colab import code_cell


CONFIG_PATH = ROOT/'experiments/2026-10-03/full-page-validation-4mp-v7/config.json'


def freeze_config():
    config = copy.deepcopy(json.loads(V5_CONFIG.read_text(encoding='utf-8')))
    config.update(schema='slayer-full-page-validation-4mp-v7',
        work_name='slayer-full-page-validation-4mp-v7',
        evidence_archive_name='full-page-validation-4mp-v7-evidence.zip')
    config['models']['qwen3-vl-4b']['max_pixels'] = 4194304
    config['baseline']['role_in_v7'] = 'Input-bundle integrity only; Ovis is not scored or rerun'
    config['retained_baseline'] = {
        'repo': 'PiotrSty/slayer-ocr-experiment-evidence',
        'revision': '5f30a4d0e32f3e4606c6a37690e3e21b59022d7b',
        'path': 'experiments/2026-10-03/full-page-comparison-v5-result/full-page-comparison-v5-evidence.zip',
        'archive_sha256': 'bf6ddc8cb25c6041f034b2fff05a933d5f9197ba1025a9f2c71c6c63e25e62e7',
        'raw_predictions_sha256': '3aec77bd1e63ae736a8d6110a40fbb40bafa6cb7bc6547659797e95eff6795a1',
        'inference_code_revision': '0d74a667f94901516dd9d3f75f9e478ca63502be',
        'rerun': False}
    config['comparison_profile'] = {
        'schema': 'slayer-full-page-validation-4mp-v7-result',
        'labels': ['qwen-v5-1mp-retained', 'qwen-v7-4mp'],
        'kind': 'Same model spec except max_pixels; separate-runtime retained baseline comparison',
        'claim_boundary': 'All 15 provisional v2 validation references. Separate sessions, not a controlled '
                          'same-session causal trial or speed benchmark. No errors or caps excluded. '
                          'Content scope and reading order are unadjudicated; no gold or SOTA claim.'}
    config['scope_policy'].update(
        baseline_prompt_matched=True,
        comparison_kind=config['comparison_profile']['kind'],
        metrics='All 15 validation pages; provisional v2 references; NFC and whitespace only',
        runtime_measurements_comparable=False, same_session_comparison=False,
        actual_input_grid_verification_required=True, character_counts_are_recall=False)
    config['prior_4mp_evidence'] = {
        'repo': 'PiotrSty/slayer-ocr-experiment-evidence',
        'revision': 'bf2bf8372ea99c8fb3175fe12a8689f57f8e3fd9',
        'path': 'experiments/2026-10-03/full-page-resolution-v6-result/full-page-resolution-v6-evidence.zip',
        'archive_sha256': 'aeed9e2dcc5091d0f6114e306c955119349faeca53d9cb00278247ca4704c065',
        'pages': 3, 'same_T4_fit_for_all_15_guaranteed': False,
        'profile_reused_unchanged': True, 'historical_spelling_fixed': False}
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return config


def build(target, code_revision):
    build_base(target, code_revision, CONFIG_PATH)
    notebook = json.loads(Path(target).read_text(encoding='utf-8'))
    replacements = {
        'scope': markdown('scope', '# OCR 4 MP: pełny split validation, 15 stron\n\n'
            'Wybierz GPU i **Uruchom wszystko**. Nie wgrywaj plików. '
            'Dane i zapisany odczyt Qwen 1 MP z v5 pobiorą się automatycznie z HF. '
            'Teraz uruchamiany jest wyłącznie Qwen 4 MP.\n\n'
            'Konfiguracja modelu jest identyczna jak wariant 4 MP z v6. '
            'Porównanie z v5 pochodzi z różnych sesji: nie jest kontrolowanym testem szybkości. '
            'Referencje są prowizoryczne, nie gold. Zachowujemy ſ, á, błędy i pętle. '
            '4 MP zadziałało na trzech stronach T4, ale nie gwarantuje to wykonania całych 15.\n'),
        'inputs': code_cell('inputs', """from training import full_page_validation_4mp as validation
dataset = WORK/'dataset'
staged = comparison.stage_bundle(CONFIG, dataset)
assert staged['pages'] == 15 and staged['gold_pages'] == 0
baseline = validation.stage_baseline(CONFIG, dataset, WORK)
write_json(WORK/'validation-code-provenance.json', {
    'code_revision': CODE_REVISION,
    'validation_sha256': digest(repo/'training/full_page_validation_4mp.py')})
print(json.dumps(baseline, indent=2))
print('Only image paths/hashes/dimensions enter the worker. No reference text or boxes.')
"""),
        'evaluation': code_cell('evaluation', """summary = validation.compare(CONFIG, dataset,
    WORK/'predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl', WORK/'scores')
print('All 15 provisional validation pages. Separate sessions; no gold or SOTA claim.')
for label, report in summary['reports'].items():
    print(f"{label}: CER={report['cer_micro']:.2%}, WER={report['wer_micro']:.2%}, "
          f"errors/missing={report['errors_or_missing']}, EOS={report['eos_pages']}, "
          f"capped={report['token_limit_pages']}")
from training.full_page_pilot import read_rows
for row in read_rows(WORK/'scores/per-page.jsonl'):
    print(row['engine'], row['id'], f"CER={row['cer']:.2%}", row['finish_reason'])
print('Grid and historical-glyph diagnostics:', WORK/'scores/input-and-glyph-diagnostics.json')
"""),
        'download-heading': markdown('download-heading', '## 7. Pobierz dowody\n'
            'Po zakończeniu prześlij **full-page-validation-4mp-v7-evidence.zip**. '
            'Tę komórkę można uruchomić również po błędzie instalacji, inferencji lub metryk, '
            'jeśli krok 2 się zakończył. Nie usuwaj częściowych wyników.\n'),
        'download': code_cell('download', """from google.colab import files
archive = comparison.package_comparison(WORK, CONFIG['evidence_archive_name'])
print('Pobierz i prześlij do Codex:', archive.name)
print('Predykcje, metryki, logi i pochodzenie; bez skanów, wag i plików programu.')
files.download(str(archive))
""")}
    for cell in notebook['cells']:
        if cell['id'] == 'inference':
            source = ''.join(cell['source']).replace('comparison.package_comparison(WORK)',
                "comparison.package_comparison(WORK, CONFIG['evidence_archive_name'])")
            replacements['inference'] = code_cell('inference', source)
    notebook['cells'] = [replacements.get(cell['id'], cell) for cell in notebook['cells']]
    Path(target).write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-config', action='store_true')
    parser.add_argument('--code-revision')
    args = parser.parse_args()
    if args.freeze_config:
        freeze_config()
    if args.code_revision:
        build(ROOT/'training/colab_full_page_validation_4mp_v7.ipynb', args.code_revision)
