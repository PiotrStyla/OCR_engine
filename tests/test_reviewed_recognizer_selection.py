from copy import deepcopy
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training.reviewed_recognizer_selection import (
    DOMAINS, checkpoint_metrics, eligibility, normalize, select, summarize,
)


def baseline():
    return {domain: dict(cer=0.2, wer=0.3, lines=count, replacement_characters=0)
            for domain, count in zip((*DOMAINS, 'combined'), (9, 75, 84))}


def candidate():
    scores = baseline()
    scores[DOMAINS[0]]['cer'] = 0.1
    scores['combined']['cer'] = 0.18
    return scores


def test_baseline_is_selected_when_every_candidate_regresses():
    scores = candidate()
    scores[DOMAINS[1]]['cer'] = 0.20001
    report = select(baseline(), [dict(id='bad', metrics=scores)])
    assert report['selected'] == 'unchanged-baseline'
    assert report['baseline_retained']
    assert not report['production_promoted']
    assert not report['sota_claim']


@pytest.mark.parametrize('domain,metric,value', [
    (DOMAINS[0], 'cer', 0.2), (DOMAINS[1], 'cer', 0.21),
    (DOMAINS[1], 'wer', 0.31), (DOMAINS[0], 'replacement_characters', 1),
    (DOMAINS[1], 'replacement_characters', 1), ('combined', 'cer', 0.2),
])
def test_every_protected_slice_is_required(domain, metric, value):
    scores = candidate()
    scores[domain][metric] = value
    assert not eligibility(scores, baseline())['eligible']


def test_selection_prefers_best_eligible_over_better_regressing_candidate():
    bad = candidate()
    bad['combined']['cer'] = 0.01
    bad[DOMAINS[1]]['cer'] = 0.21
    good = candidate()
    result = select(baseline(), [dict(id='bad', metrics=bad), dict(id='good', metrics=good)])
    assert result['selected'] == 'good'
    assert not result['baseline_retained']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1])
def test_invalid_metrics_fail(value):
    scores = candidate()
    scores[DOMAINS[0]]['cer'] = value
    with pytest.raises(ValueError, match='Invalid'):
        eligibility(scores, baseline())


def test_coverage_drift_fails():
    scores = candidate()
    scores[DOMAINS[0]]['lines'] = 8
    with pytest.raises(ValueError, match='coverage'):
        eligibility(scores, baseline())


def test_historical_glyphs_are_not_modernized():
    raw = 'Po\u017f\u0142a\u0142 \u00e1 \u0292\u0307'
    assert normalize(' ' + raw + '\n') == raw
    rows = [dict(domain=domain, id='line', reference=raw, prediction='Pos\u0142a\u0142 a z') for domain in DOMAINS]
    assert summarize(rows)['combined']['cer'] > 0
    rows[0]['prediction'] = '\ufffd'
    assert summarize(rows)[DOMAINS[0]]['replacement_characters'] == 1


def test_duplicate_predictions_rejected():
    rows = [dict(domain=domain, id='line', reference='a', prediction='a') for domain in DOMAINS]
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(rows + [deepcopy(rows[0])])


def test_checkpoint_metrics_use_raw_reference_order_and_preserve_padding(tmp_path):
    class Tokenizer:
        pad_token_id = 1

        def batch_decode(self, rows, **kwargs):
            assert kwargs['clean_up_tokenization_spaces'] is False
            return ['a' if row[0] == 2 else 'b' for row in rows]

    references = [dict(domain=domain, id='line', reference='a') for domain in DOMAINS]
    base = baseline()
    for domain, count in zip((*DOMAINS, 'combined'), (1, 1, 2)):
        base[domain]['lines'] = count
    metric = checkpoint_metrics(Tokenizer(), references, base, tmp_path)
    labels = np.array([[2, -100], [2, -100]])
    score = metric(SimpleNamespace(predictions=np.array([[2, 1], [2, 1]]), label_ids=labels))
    assert score['eligible'] == 1
    assert score['selection_score'] == 0
    assert labels[0, 1] == -100
    assert (tmp_path / 'checkpoint-evaluation-001.jsonl').exists()
    with pytest.raises(ValueError, match='order'):
        metric(SimpleNamespace(predictions=np.array([[2, 1], [2, 1]]),
                               label_ids=np.array([[3, 1], [2, 1]])))
    rejected = metric(SimpleNamespace(predictions=np.array([[2, 1], [3, 1]]), label_ids=labels))
    assert rejected['eligible'] == 0
    assert rejected['selection_score'] >= 1e6


def test_config_is_bounded_and_replay_subset_is_stable():
    from training.run_reviewed_recognizer_colab_v2 import CONFIG
    config = json.loads(CONFIG.read_text())
    variants = config['variants']
    assert [(row['learning_rate'], row['replay_count']) for row in variants] == [
        (1e-5, 500), (3e-6, 500), (3e-6, 2000)]
    assert config['training']['epochs'] == 3
    assert config['synthetic_order_prefix'] == 'reviewed-colab-v1:'
    assert not config['sota_claim']


def test_v2_notebook_has_one_runner_auto_inputs_and_drive_fallback(tmp_path):
    import nbformat
    from training.build_reviewed_recognizer_colab import build

    path = tmp_path / 'v2.ipynb'
    build(path, 'a' * 40, version='v2')
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
    assert code.count("'training.run_reviewed_recognizer_colab_v2'") == 1
    assert "'training.run_reviewed_recognizer_colab'," not in code
    assert 'files.upload' not in code
    assert 'with_pip=False' in code
    assert 'BACKUP_TO_DRIVE = False' in code
    assert 'sha256(target) == expected' in code
    assert 'files.download(str(EVIDENCE_ZIP))' in code
    assert 'files.download(str(result_zip))' not in code
    assert 'tests/test_training_protocol.py' in code
