"""Build the single no-upload paired resolution Colab."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from training.build_full_page_comparison_colab import ROOT, CONFIG_PATH as V5_CONFIG, build as build_base
from training.build_historical_recognizer_colab import code_cell
from training.build_full_page_pilot_colab import markdown


CONFIG_PATH = ROOT/'experiments/2026-10-03/full-page-resolution-v6/config.json'


def freeze_config():
    config = copy.deepcopy(json.loads(V5_CONFIG.read_text(encoding='utf-8')))
    config.update(schema='slayer-full-page-resolution-v6',
                  work_name='slayer-full-page-resolution-v6',
                  evidence_archive_name='full-page-resolution-v6-evidence.zip',
                  arms={'one-mp': 1048576, 'four-mp': 4194304},
                  seed=42, post_hoc_selection=True,
                  selection=[
                      {'id': 'Slawna_wiktoria_FT__437089', 'reason': 'v5 numeric loop on title page'},
                      {'id': 'Wiesc_FT__436884', 'reason': 'dense historical body and glyph errors'},
                      {'id': 'Choragiew_FT__436799', 'reason': 'two user-confirmed Po\u017f\u0142a\u0142 instances'}])
    config['scope_policy'].update(
        comparison_kind='Same-session paired resolution arms; only max_pixels changes in model spec',
        metrics='Post-hoc diagnostic against exact v2 references, NFC and whitespace only',
        actual_input_grid_verification_required=True,
        character_counts_are_recall=False,
        runtime_comparison='Sequential same-session diagnostics, not controlled performance measurement')
    config['previous_evidence'] = {
        'repo': 'PiotrSty/slayer-ocr-experiment-evidence',
        'revision': '5f30a4d0e32f3e4606c6a37690e3e21b59022d7b',
        'path': 'experiments/2026-10-03/full-page-comparison-v5-result/full-page-comparison-v5-evidence.zip',
        'sha256': 'bf6ddc8cb25c6041f034b2fff05a933d5f9197ba1025a9f2c71c6c63e25e62e7',
        'used_as_rerun_baseline': False}
    config['scope_policy'].pop('baseline_prompt_matched')
    config['baseline']['role_in_v6'] = 'Bundle integrity check only; not a scored resolution arm'
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return config


def build(target, code_revision):
    build_base(target, code_revision, CONFIG_PATH)
    notebook = json.loads(Path(target).read_text(encoding='utf-8'))
    replacements = {
        'scope': markdown('scope', '# OCR: kontrolowany test 1 MP kontra 4 MP\n\n'
            'Wybierz GPU i **Uruchom wszystko**. Niczego nie wgrywaj. Dane pobierają się z HF. '
            'Trzy strony zostaną odczytane dwukrotnie tym samym Qwen3-VL-4B, '
            'ze zmienionym wyłącznie budżetem pikseli.\n\n'
            'To diagnostyka wybrana po analizie błędów v5, nie niezależny benchmark ani SOTA. '
            'Referencje v2 są prowizoryczne. Zachowujemy ſ, á i dawną pisownię. '
            '4 MP nie zostało jeszcze sprawdzone na T4; błędy pamięci zostaną zachowane.\n'),
        'inputs': code_cell('inputs', """from training import full_page_resolution as resolution
dataset = WORK/'dataset'
staged = comparison.stage_bundle(CONFIG, dataset)
selection = resolution.prepare_subset(CONFIG, dataset)
write_json(WORK/'resolution-code-provenance.json', {
    'code_revision': CODE_REVISION,
    'resolution_sha256': digest(repo/'training/full_page_resolution.py')})
print(json.dumps(selection, indent=2))
print('Model receives images only, without reference text, notes or boxes.')
"""),
        'inference-heading': markdown('inference-heading', '## 5. Dwa odczyty tych samych trzech stron\n'
            'Najpierw 1 MP, potem 4 MP. Każdy wariant działa w osobnym procesie, '
            'a wyniki zapisują się po stronie. Powtórzenie komórki nie ponawia zapisanych błędów.\n'),
        'inference': code_cell('inference', """for arm, arm_config in resolution.arm_configs(CONFIG).items():
    arm_root = WORK/'arms'/arm
    destination = arm_root/'predictions'
    destination.mkdir(parents=True, exist_ok=True)
    arm_config_path = arm_root/'config.json'
    if arm_config_path.exists():
        assert json.loads(arm_config_path.read_text()) == arm_config
    write_json(arm_config_path, arm_config)
    command = [str(python), '-u', '-m', 'training.full_page_pilot', 'worker',
        '--config', str(arm_config_path), '--engine', 'qwen3-vl-4b',
        '--inputs', str(dataset/'resolution-inputs.jsonl'), '--output', str(destination)]
    process = None
    print('START:', arm)
    try:
        with (arm_root/'worker.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command,
                env={**os.environ, 'PYTHONPATH': str(repo), 'TOKENIZERS_PARALLELISM': 'false'},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)
                    log.flush()
                exit_status = process.wait()
                write_json(arm_root/'process-result.json', {'arm': arm, 'exit_status': exit_status})
                print('Worker exit status:', exit_status)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
    finally:
        print('PARTIAL_EVIDENCE_READY:', comparison.package_comparison(
            WORK, CONFIG['evidence_archive_name']))
"""),
        'evaluation-heading': markdown('evaluation-heading', '## 6. Wszystkie wyniki i ślady wejścia\n'
            'Błędy i brakujące wyniki liczymy jako pusty tekst, pętli nie usuwamy. '
            'Liczby znaków są diagnostyką, nie miarą recall. '
            'Sprawdź faktyczne piksele wejściowe, zanim przypiszesz różnicę rozdzielczości.\n'),
        'evaluation': code_cell('evaluation', """summary = resolution.compare(CONFIG, dataset, WORK)
print('Post-hoc development diagnostics; no gold, teacher promotion or SOTA.')
for arm, report in summary['reports'].items():
    print(f"{arm}: CER={report['cer_micro']:.2%}, WER={report['wer_micro']:.2%}, "
          f"errors/missing={report['errors_or_missing']}, EOS={report['eos_pages']}, "
          f"capped={report['token_limit_pages']}")
from training.full_page_pilot import read_rows
for row in read_rows(WORK/'scores/per-page.jsonl'):
    print(row['arm'], row['id'], f"CER={row['cer']:.2%}",
          row['finish_reason'], json.dumps(row['input_geometry']))
    print('Raw character counts (not recall):',
          {'long_s': row['raw_long_s_count'], 'a_acute': row['raw_a_acute_count'],
           'exact_Poslal': row['raw_poslal_count']})
"""),
        'chart': code_cell('chart', """import matplotlib.pyplot as plt
labels = list(summary['reports'])
values = [summary['reports'][label]['cer_micro']*100 for label in labels]
fig, ax = plt.subplots(figsize=(9, 3))
ax.barh(labels, values, color=['#53758e', '#b25346'])
ax.set_xlim(0, max(values+[1])*1.2)
ax.set_xlabel('CER (%)')
ax.set_title('3 strony diagnostyczne: wszystkie wyniki, referencje v2 nie gold')
for index, value in enumerate(values):
    ax.text(value, index, f' {value:.2f}%', va='center')
fig.tight_layout()
fig.savefig(WORK/'cer-comparison.png', dpi=140)
plt.show()
"""),
        'download-heading': markdown('download-heading', '## 7. Pobierz dowody\n'
            'Tę komórkę uruchom również po błędzie instalacji, inferencji lub metryk, '
            'jeśli krok 2 się zakończył. Prześlij **full-page-resolution-v6-evidence.zip**.\n'),
        'download': code_cell('download', """from google.colab import files
archive = comparison.package_comparison(WORK, CONFIG['evidence_archive_name'])
print('Pobierz i prześlij do Codex:', archive.name)
print('Predykcje, metryki, konfiguracje i logi; bez skanów i wag.')
files.download(str(archive))
""")}
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
        build(ROOT/'training/colab_full_page_resolution_v6.ipynb', args.code_revision)
