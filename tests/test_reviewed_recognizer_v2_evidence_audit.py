from copy import deepcopy
import hashlib

import pytest

from training.audit_reviewed_recognizer_v2_evidence import (
    assert_close, verify_checkpoint_metrics, verify_pair_files, verify_references, verify_training_lineage,
)
from training.reviewed_recognizer_selection import DOMAINS, eligibility, summarize


def lineage():
    def record(name):
        return dict(id=name, image_sha256=hashlib.sha256(name.encode()).hexdigest(),
                    text_sha256=hashlib.sha256(b'a').hexdigest())
    manifests = {'reviewed': [record('train')], 'synthetic': [record('synth')],
                 DOMAINS[0]: [record('hist')], DOMAINS[1]: [record('ordinary')]}
    recipe = dict(historical_repeats=4, epochs=3, batch_size=2, gradient_accumulation_steps=4,
                  seed=42, max_target_length=128)
    variant = dict(learning_rate=3e-6)
    run = dict(train=manifests['reviewed'] * 4 + manifests['synthetic'],
               validation=[dict(row, id=domain + '__' + row['id']) for domain in DOMAINS
                           for row in manifests[domain]], source_commit='revision',
               use_4bit=False, include_mlp=False, max_target_length=128,
               gradient_accumulation_steps=4,
               training_arguments=dict(num_train_epochs=3, per_device_train_batch_size=2,
                   gradient_accumulation_steps=4, learning_rate=3e-6, seed=42, data_seed=42,
                   metric_for_best_model='selection_score', load_best_model_at_end=True, fp16=True))
    return run, variant, recipe, manifests


def test_intentional_historical_repeats_keep_exact_lineage():
    run, variant, recipe, manifests = lineage()
    verify_training_lineage(run, variant, recipe, manifests, manifests['synthetic'], 'revision')


@pytest.mark.parametrize('field', ['train', 'validation', 'source_commit', 'learning_rate', 'metric', 'use_4bit', 'length'])
def test_lineage_or_recipe_drift_is_rejected(field):
    run, variant, recipe, manifests = lineage()
    run = deepcopy(run)
    if field in ('train', 'validation'):
        run[field][0]['text_sha256'] = 'bad'
    elif field == 'source_commit':
        run[field] = 'other'
    elif field == 'learning_rate':
        run['training_arguments']['learning_rate'] = 1e-5
    elif field == 'metric':
        run['training_arguments']['metric_for_best_model'] = 'cer'
    elif field == 'use_4bit':
        run[field] = True
    else:
        run['max_target_length'] = 64
    with pytest.raises(ValueError):
        verify_training_lineage(run, variant, recipe, manifests, manifests['synthetic'], 'revision')


def prediction_fixture():
    text = 'Po\u017f\u0142a\u0142 \u00e1'
    rows = [dict(domain=domain, id='line', reference=text, prediction=text) for domain in DOMAINS]
    manifests = {domain: [dict(id='line', text_sha256=hashlib.sha256(text.encode()).hexdigest())]
                 for domain in DOMAINS}
    return rows, manifests


def test_raw_historical_references_bind_to_input_hashes():
    rows, manifests = prediction_fixture()
    verify_references(rows, rows, manifests)


@pytest.mark.parametrize('mutation', ['reference', 'hash', 'missing', 'duplicate-prediction', 'duplicate-input'])
def test_reference_identity_and_hash_drift_are_rejected(mutation):
    baseline, manifests = prediction_fixture()
    rows = deepcopy(baseline)
    if mutation == 'reference':
        rows[0]['reference'] = 'Pos\u0142a\u0142 a'
    elif mutation == 'hash':
        manifests[DOMAINS[0]][0]['text_sha256'] = 'bad'
    elif mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate-prediction':
        rows.append(deepcopy(rows[0]))
    else:
        manifests[DOMAINS[0]].append(deepcopy(manifests[DOMAINS[0]][0]))
    with pytest.raises(ValueError):
        verify_references(rows, baseline, manifests)


def checkpoint_fixture():
    rows, _ = prediction_fixture()
    scores = summarize(rows)
    base = deepcopy(scores)
    base[DOMAINS[0]]['cer'] = 0.2
    base['combined']['cer'] = 0.1
    gate = eligibility(scores, base)
    logged = dict(eval_cer=0, eval_wer=0, eval_eligible=1, eval_selection_score=0, eval_loss=1.0)
    for domain in DOMAINS:
        for metric in ('cer', 'wer', 'replacement_characters'):
            logged['eval_' + domain + '_' + metric] = scores[domain][metric]
    return scores, gate, logged


def test_checkpoint_domain_metrics_and_score_are_recomputed():
    scores, gate, logged = checkpoint_fixture()
    assert verify_checkpoint_metrics(scores, gate, logged) == 0


@pytest.mark.parametrize('field', ['eval_cer', 'eval_eligible', 'eval_selection_score', 'eval_loss',
                                 'eval_ordinary-development_wer'])
def test_checkpoint_metric_drift_is_rejected(field):
    scores, gate, logged = checkpoint_fixture()
    logged[field] = float('nan') if field == 'eval_loss' else logged[field] + 0.1
    with pytest.raises(ValueError):
        verify_checkpoint_metrics(scores, gate, logged)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0.10001])
def test_metric_comparison_rejects_invalid_or_drifting_numbers(value):
    with pytest.raises(ValueError):
        assert_close(value, 0.1, 'test')


def test_local_pair_bytes_bind_images_and_raw_glyphs(tmp_path):
    from training.full_page_pilot import digest
    (tmp_path / 'line.png').write_bytes(b'image')
    (tmp_path / 'line.txt').write_text('Po\u017f\u0142a\u0142 \u00e1', encoding='utf-8')
    records = [dict(id='line', image_sha256=digest(tmp_path / 'line.png'),
                    text_sha256=digest(tmp_path / 'line.txt'))]
    verify_pair_files(tmp_path, records)
    (tmp_path / 'line.txt').write_text('Pos\u0142a\u0142 a', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        verify_pair_files(tmp_path, records)


def test_missing_local_pair_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='coverage'):
        verify_pair_files(tmp_path, [dict(id='missing')])
