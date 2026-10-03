import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from PIL import Image
import pytest

from training.audit_full_page_resolution import audit, ROOT
from training.build_full_page_resolution_colab import CONFIG_PATH
from training.full_page_comparison import package_comparison
from training.full_page_pilot import digest, write_json, write_rows
from training.full_page_resolution import ENGINE, PAGE_IDS, arm_configs, compare, prepare_subset


REVISION = '6a9896b1c5ebdb941b29dfd3d48c4cb2690658dd'


def make_evidence(tmp_path):
    work = tmp_path/'work'
    dataset = work/'dataset'
    dataset.mkdir(parents=True)
    image = dataset/'page.png'
    Image.new('RGB', (32, 64), 'white').save(image)
    records = [{'id': key, 'image': 'page.png', 'sha256': digest(image),
                'width': 32, 'height': 64, 'split': 'validation', 'final_test': False,
                'eligible_for_training': False, 'text': 'Po\u017f\u0142a\u0142 b\u0142\u00e1waty'} for key in PAGE_IDS]
    write_rows(dataset/'manifest.jsonl', records)
    baseline = dataset/'retained-projected-predictions.jsonl'
    write_rows(baseline, [{'id': row['id'], 'status': 'ok', 'text': row['text']} for row in records])
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    config['dataset'].update(pages=3, manifest_sha256=digest(dataset/'manifest.jsonl'),
                             baseline_sha256=digest(baseline))
    config_path = tmp_path/'frozen-config.json'
    write_json(config_path, config)
    write_json(work/'config.json', config)
    prepare_subset(config, dataset)
    inputs = [{key: row[key] for key in ('id', 'image', 'sha256', 'width', 'height')}
              | {'source_regions': []} for row in records]
    write_rows(dataset/'inference-inputs.jsonl', inputs)
    write_json(dataset/'staging-provenance.json', {
        'bundle_sha256': config['dataset']['archive_sha256'],
        'inputs_sha256': digest(dataset/'inference-inputs.jsonl'),
        'reference_text_sent_to_model': False, 'source_geometry_sent_to_model': False})
    hashes = {key: hashlib.sha256(subprocess.check_output(
        ['git', 'show', f'{REVISION}:training/{filename}'], cwd=ROOT)).hexdigest()
        for key, filename in [('runner_sha256', 'full_page_pilot.py'),
                              ('comparison_sha256', 'full_page_comparison.py'),
                              ('resolution_sha256', 'full_page_resolution.py')]}
    write_json(work/'code-provenance.json', {'code_revision': REVISION, **hashes})
    write_json(work/'resolution-code-provenance.json',
               {'code_revision': REVISION, 'resolution_sha256': hashes['resolution_sha256']})
    write_json(work/'runtime.json', {'gpu': 'synthetic-test-not-a-GPU-run'})
    (work/'preflight.log').write_text(json.dumps({
        'packages': dict(item.split('==') for item in config['models'][ENGINE]['packages'])})+'\n')
    for arm, arm_config in arm_configs(config).items():
        destination = work/'arms'/arm/'predictions'
        destination.mkdir(parents=True)
        write_json(destination.parent/'config.json', arm_config)
        write_json(destination.parent/'process-result.json', {'arm': arm, 'exit_status': 0})
        spec = arm_config['models'][ENGINE]
        write_json(destination/'identity.json', {'engine': ENGINE, 'spec': spec,
            'input_sha256': digest(dataset/'resolution-inputs.jsonl'), 'runner_sha256': hashes['runner_sha256']})
        write_json(destination/'environment.json', {'gpu': 'synthetic', 'reference_text_sent_to_model': False,
            'packages': dict(item.split('==') for item in spec['packages'])})
        predictions = [{'id': row['id'], 'status': 'ok', 'text': row['text'], 'format': 'plain',
            'generated_tokens': 1, 'finish_reason': 'eos', 'token_limit_reached': False,
            'elapsed_seconds': 0.1, 'peak_allocated_bytes': 1,
            'generation_trace': {'generated_token_ids': [1], 'eos_token_ids': [1]},
            'input_geometry': {'image_grid_thw': [[1, 32, 32]], 'patch_size': 16, 'merge_size': 2,
                'processed_width': 512, 'processed_height': 512, 'processed_pixels': 262144,
                'visual_tokens': 256, 'original_width': 32, 'original_height': 64,
                'min_pixels': spec['min_pixels'], 'max_pixels': spec['max_pixels']}} for row in records]
        write_rows(destination/f'{ENGINE}-full-page.jsonl', predictions)
    compare(config, dataset, work)
    archive = package_comparison(work, 'evidence.zip')
    return archive, dataset, config_path


def test_audit_recomputes_and_preserves_evidence(tmp_path):
    archive, dataset, config = make_evidence(tmp_path)
    before = archive.read_bytes()
    output = tmp_path/'audit'
    result = audit(archive, dataset, config, output, REVISION)
    assert result['metrics_recomputed'] and result['pages_per_arm'] == 3
    assert result['reports']['four-mp']['cer_micro'] == 0
    assert (output/'source-evidence.zip').read_bytes() == before == archive.read_bytes()
    assert result['glyph_counts']['\u017f']['reference'] == 3
    assert result['body_only_diagnostic']['primary_metric'] is False


@pytest.mark.parametrize('change', ['checksum', 'reference-input', 'metrics', 'geometry', 'code', 'preflight'])
def test_audit_rejects_mutations(tmp_path, change):
    archive, dataset, config = make_evidence(tmp_path)
    with zipfile.ZipFile(archive) as stream:
        payload = {name: stream.read(name) for name in stream.namelist()}
    if change == 'checksum':
        payload['runtime.json'] = b'{}'
    elif change == 'preflight':
        payload['preflight.log'] = b'{"packages":{}}\n'
        payload['checksums.json'] = json.dumps({name: hashlib.sha256(data).hexdigest()
            for name, data in payload.items() if name != 'checksums.json'}).encode()
    else:
        name = {'reference-input': 'dataset/resolution-inputs.jsonl', 'metrics': 'scores/metrics.json',
                'geometry': f'arms/four-mp/predictions/{ENGINE}-full-page.jsonl',
                'code': 'resolution-code-provenance.json'}[change]
        if name.endswith('.jsonl'):
            rows = [json.loads(line) for line in payload[name].splitlines()]
            if change == 'reference-input':
                rows[0]['text'] = 'ground truth leak'
            else:
                rows[0]['input_geometry']['processed_pixels'] = 1
            payload[name] = ('\n'.join(json.dumps(row) for row in rows)+'\n').encode()
        else:
            row = json.loads(payload[name])
            if change == 'metrics':
                row['reports']['four-mp']['cer_micro'] = 0.5
            else:
                row['resolution_sha256'] = '0'*64
            payload[name] = json.dumps(row).encode()
        payload['checksums.json'] = json.dumps({name: hashlib.sha256(data).hexdigest()
            for name, data in payload.items() if name != 'checksums.json'}).encode()
    altered = tmp_path/'altered.zip'
    with zipfile.ZipFile(altered, 'w') as stream:
        for name, data in payload.items():
            stream.writestr(name, data)
    with pytest.raises(ValueError):
        audit(altered, dataset, config, tmp_path/'rejected', REVISION)
