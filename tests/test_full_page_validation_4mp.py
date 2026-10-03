import copy
import io
import json
from pathlib import Path

import pytest

from training import full_page_validation_4mp as validation
from training.build_full_page_validation_4mp_colab import CONFIG_PATH, build
from training.full_page_pilot import digest, read_rows, write_json, write_rows
from tests.test_full_page_comparison import bundle_fixture
from training.full_page_comparison import stage_bundle


def test_frozen_v7_model_equals_v6_four_mp():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    validation.validate_profile(config)
    v6 = json.loads((CONFIG_PATH.parent.parent/'full-page-resolution-v6/config.json').read_text(encoding='utf-8'))
    v6['models']['qwen3-vl-4b']['max_pixels'] = 4194304
    assert config['models'] == v6['models']
    assert config['dataset']['pages'] == 15
    assert config['scope_policy']['same_session_comparison'] is False
    assert config['comparison_profile']['labels'] == ['qwen-v5-1mp-retained', 'qwen-v7-4mp']
    config['models']['qwen3-vl-4b']['max_new_tokens'] = 2048
    with pytest.raises(ValueError, match='exact v5'):
        validation.validate_profile(config)


def fixture(tmp_path, monkeypatch):
    data_config, opener = bundle_fixture(tmp_path)
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    config['dataset'] = data_config['dataset']
    config['baseline'] = data_config['baseline']
    previous = copy.deepcopy(config)
    previous['models']['qwen3-vl-4b']['max_pixels'] = 1048576
    previous_path = tmp_path/'v5.json'
    write_json(previous_path, previous)
    monkeypatch.setattr(validation, 'V5_CONFIG', previous_path)
    dataset = tmp_path/'dataset'
    stage_bundle(config, dataset, opener)
    baseline = dataset/'qwen-v5-1mp.jsonl'
    write_rows(baseline, [{'id': 'p', 'status': 'ok', 'format': 'plain',
        'text': 'b\u0142\u00e1waty \u017f\u0142owo', 'finish_reason': 'eos', 'token_limit_reached': False,
        'input_geometry': {'processed_pixels': 1048576}}])
    config['retained_baseline']['raw_predictions_sha256'] = digest(baseline)
    return config, dataset


def test_scoring_retains_historical_errors_caps_and_missing_pages(tmp_path, monkeypatch):
    config, dataset = fixture(tmp_path, monkeypatch)
    candidate = tmp_path/'candidate.jsonl'
    write_rows(candidate, [{'id': 'p', 'status': 'ok', 'format': 'plain',
        'text': 'blawaty slowo '*50, 'finish_reason': 'length', 'token_limit_reached': True,
        'input_geometry': {'processed_pixels': 4194304}}])
    summary = validation.compare(config, dataset, candidate, tmp_path/'scores')
    assert summary['reports']['qwen-v5-1mp-retained']['cer_micro'] == 0
    assert summary['reports']['qwen-v7-4mp']['cer_micro'] > 1
    assert summary['reports']['qwen-v7-4mp']['token_limit_pages'] == 1
    assert summary['schema'] == 'slayer-full-page-validation-4mp-v7-result'
    assert summary['baseline_rerun'] is summary['sota_claim'] is False
    assert all(report['runtime_measurements_comparable'] is False for report in summary['reports'].values())
    geometry = json.loads((tmp_path/'scores/input-and-glyph-diagnostics.json').read_text())
    assert geometry['pages'][0]['processed_pixels_increased'] is True
    assert geometry['pages'][0]['reference_long_s_count'] == 1
    assert geometry['pages'][0]['four_mp_long_s_count'] == 0
    candidate.unlink()
    missing = validation.compare(config, dataset, candidate, tmp_path/'missing')
    assert missing['reports']['qwen-v7-4mp']['errors_or_missing'] == 1
    assert missing['reports']['qwen-v7-4mp']['cer_micro'] == 1


def test_scoring_rejects_changed_baseline_and_markdown(tmp_path, monkeypatch):
    config, dataset = fixture(tmp_path, monkeypatch)
    candidate = tmp_path/'candidate.jsonl'
    write_rows(candidate, [{'id': 'p', 'status': 'ok', 'text': '# modified', 'format': 'markdown'}])
    with pytest.raises(ValueError, match='plain-text'):
        validation.compare(config, dataset, candidate, tmp_path/'scores')
    (dataset/'qwen-v5-1mp.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        validation.compare(config, dataset, candidate, tmp_path/'scores')


@pytest.mark.parametrize('labels', [None, ['same', 'same'], ['baseline', 1]])
def test_invalid_comparison_labels_rejected(tmp_path, monkeypatch, labels):
    config, dataset = fixture(tmp_path, monkeypatch)
    config['comparison_profile']['labels'] = labels
    with pytest.raises(ValueError, match='labels'):
        validation.compare(config, dataset, tmp_path/'missing.jsonl', tmp_path/'scores')


def test_baseline_download_checks_hash_and_uses_independent_audit(tmp_path, monkeypatch):
    import hashlib
    config, dataset = fixture(tmp_path, monkeypatch)
    baseline_data = (dataset/'qwen-v5-1mp.jsonl').read_bytes()
    archive_data = b'synthetic verified archive handed to mock auditor'
    config['retained_baseline']['archive_sha256'] = hashlib.sha256(archive_data).hexdigest()
    calls = []
    def auditor(archive, images, previous_config, output, code):
        assert archive.read_bytes() == archive_data
        assert images == dataset and code == '0d74a667f94901516dd9d3f75f9e478ca63502be'
        source = output/'evidence/predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl'
        source.parent.mkdir(parents=True)
        source.write_bytes(baseline_data)
        calls.append(output)
        return {'members_verified': 30, 'metrics_recomputed': True}
    monkeypatch.setattr(validation, 'audit', auditor)
    result = validation.stage_baseline(config, dataset, tmp_path/'work', lambda *a, **kw: io.BytesIO(archive_data))
    assert result['members_verified'] == 30 and result['baseline_rerun'] is False
    validation.stage_baseline(config, dataset, tmp_path/'work', lambda *a, **kw: pytest.fail('Use cached archive'))
    assert len(calls) == 2 and calls[0] != calls[1]
    with pytest.raises(ValueError, match='checksum'):
        validation.stage_baseline(config, dataset, tmp_path/'bad', lambda *a, **kw: io.BytesIO(b'bad'))


def test_notebook_is_no_upload_and_compiles(tmp_path):
    target = tmp_path/'v7.ipynb'
    build(target, 'a'*40)
    notebook = json.loads(target.read_text(encoding='utf-8'))
    sources = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            compile(source, cell['id'], 'exec')
            sources.append(source)
    code = '\n'.join(sources)
    assert 'files.upload' not in code and 'stage_baseline(CONFIG' in code
    assert 'EnvBuilder(with_pip=True' not in code
    assert "validation.compare(CONFIG" in code
    assert "comparison.package_comparison(WORK, CONFIG['evidence_archive_name'])" in code
    assert 'files.download(str(archive))' in code
