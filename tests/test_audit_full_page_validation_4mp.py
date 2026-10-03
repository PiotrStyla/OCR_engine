import hashlib
import json
import shutil
import subprocess
import zipfile

import pytest

from tests.test_full_page_validation_4mp import fixture
from training import audit_full_page_validation_4mp as auditor
from training import full_page_validation_4mp as validation
from training.full_page_comparison import package_comparison
from training.full_page_pilot import digest, write_json, write_rows


REVISION = '990146abb83005205e50dc28238c9767915a7efb'


def make_evidence(tmp_path, monkeypatch):
    config, dataset = fixture(tmp_path, monkeypatch)
    original_baseline = tmp_path/'original-v5.zip'
    with zipfile.ZipFile(original_baseline, 'w') as stream:
        stream.writestr('note.json', '{}')
    config['retained_baseline']['archive_sha256'] = digest(original_baseline)
    work = tmp_path/'work'
    shutil.copytree(dataset, work/'dataset')
    write_json(work/'config.json', config)
    config_path = tmp_path/'frozen-v7.json'
    write_json(config_path, config)
    code = {key: hashlib.sha256(subprocess.check_output(
        ['git', 'show', f'{REVISION}:training/{name}'], cwd=auditor.ROOT)).hexdigest()
        for key, name in [('runner_sha256', 'full_page_pilot.py'),
                          ('comparison_sha256', 'full_page_comparison.py'),
                          ('validation_sha256', 'full_page_validation_4mp.py')]}
    write_json(work/'code-provenance.json', {'code_revision': REVISION, **code})
    write_json(work/'validation-code-provenance.json',
               {'code_revision': REVISION, 'validation_sha256': code['validation_sha256']})
    spec = config['models']['qwen3-vl-4b']
    (work/'preflight.log').write_text(json.dumps({'packages': dict(item.split('==') for item in spec['packages'])}))
    write_json(work/'runtime.json', {'gpu': 'synthetic-test-no-GPU-execution'})
    write_json(work/'process-result.json', {'exit_status': 0})
    audit_directory = 'baseline/audit-synthetic'
    retained = config['retained_baseline']
    write_json(work/'baseline-provenance.json', {'archive_sha256': retained['archive_sha256'],
        'raw_predictions_sha256': retained['raw_predictions_sha256'],
        'inference_code_revision': retained['inference_code_revision'], 'v5_metrics_recomputed': True,
        'members_verified': 1, 'baseline_rerun': False, 'same_session_comparison': False,
        'runtime_measurements_comparable': False, 'audit_directory': audit_directory})
    baseline_result = {'members_verified': 1, 'metrics_recomputed': True, 'pages': 1}
    (work/audit_directory).mkdir(parents=True)
    write_json(work/audit_directory/'audit.json', baseline_result)
    monkeypatch.setattr(auditor, 'audit_v5', lambda *args: baseline_result)
    destination = work/'predictions/qwen3-vl-4b'
    destination.mkdir(parents=True)
    write_json(destination/'identity.json', {'engine': 'qwen3-vl-4b', 'spec': spec,
        'input_sha256': digest(work/'dataset/inference-inputs.jsonl'), 'runner_sha256': code['runner_sha256']})
    write_json(destination/'environment.json', {'reference_text_sent_to_model': False,
        'packages': dict(item.split('==') for item in spec['packages'])})
    prediction = {'id': 'p', 'text': 'b\u0142\u00e1waty \u017f\u0142owo', 'format': 'plain', 'status': 'ok',
        'generated_tokens': 1, 'finish_reason': 'eos', 'token_limit_reached': False,
        'elapsed_seconds': 0.1, 'peak_allocated_bytes': 1,
        'generation_trace': {'generated_token_ids': [1], 'eos_token_ids': [1]},
        'input_geometry': {'image_grid_thw': [[1, 32, 32]], 'patch_size': 16, 'merge_size': 2,
            'processed_width': 512, 'processed_height': 512, 'processed_pixels': 262144,
            'visual_tokens': 256, 'original_width': 16, 'original_height': 32,
            'min_pixels': spec['min_pixels'], 'max_pixels': spec['max_pixels']}}
    predictions = destination/'qwen3-vl-4b-full-page.jsonl'
    write_rows(predictions, [prediction])
    validation.compare(config, work/'dataset', predictions, work/'scores')
    return package_comparison(work, 'v7.zip'), dataset, original_baseline, config_path


def test_audit_preserves_original_and_recomputes(tmp_path, monkeypatch):
    archive, dataset, baseline, config = make_evidence(tmp_path, monkeypatch)
    before = archive.read_bytes()
    output = tmp_path/'audited'
    result = auditor.audit(archive, dataset, baseline, config, output, REVISION)
    assert result['metrics_recomputed'] and result['pages'] == 1
    assert result['reports']['qwen-v7-4mp']['cer_micro'] == 0
    assert result['glyph_counts']['\u017f']['candidate'] == 1
    assert result['old_uncapped_secondary_diagnostic']['primary_metric'] is False
    assert result['model_promotion'] is result['sota_claim'] is False
    assert (output/'source-evidence.zip').read_bytes() == before == archive.read_bytes()


@pytest.mark.parametrize('change', ['checksum', 'scores', 'inputs', 'baseline', 'geometry', 'code', 'preflight', 'nested-audit'])
def test_audit_rejects_mutations(tmp_path, monkeypatch, change):
    archive, dataset, baseline, config = make_evidence(tmp_path, monkeypatch)
    with zipfile.ZipFile(archive) as stream:
        payload = {name: stream.read(name) for name in stream.namelist()}
    if change == 'checksum':
        payload['runtime.json'] = b'{}'
    else:
        name = {'scores': 'scores/metrics.json', 'inputs': 'dataset/inference-inputs.jsonl',
                'baseline': 'dataset/qwen-v5-1mp.jsonl',
                'geometry': 'predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl',
                'code': 'code-provenance.json', 'preflight': 'preflight.log',
                'nested-audit': 'baseline/audit-synthetic/audit.json'}[change]
        if name.endswith('.jsonl'):
            rows = [json.loads(line) for line in payload[name].splitlines()]
            if change == 'geometry':
                rows[0]['input_geometry']['processed_pixels'] = 1
            else:
                rows[0]['text'] = 'changed or leaked reference'
            payload[name] = ('\n'.join(json.dumps(row) for row in rows)+'\n').encode()
        elif change == 'preflight':
            payload[name] = b'{"packages":{}}'
        else:
            row = json.loads(payload[name])
            if change == 'scores':
                row['reports']['qwen-v7-4mp']['cer_micro'] = 0.5
            elif change == 'code':
                row['code_revision'] = '0'*40
            else:
                row['metrics_recomputed'] = False
            payload[name] = json.dumps(row).encode()
        payload['checksums.json'] = json.dumps({name: hashlib.sha256(data).hexdigest()
            for name, data in payload.items() if name != 'checksums.json'}).encode()
    changed = tmp_path/'changed.zip'
    with zipfile.ZipFile(changed, 'w') as stream:
        for name, data in payload.items():
            stream.writestr(name, data)
    with pytest.raises(ValueError):
        auditor.audit(changed, dataset, baseline, config, tmp_path/'rejected', REVISION)
