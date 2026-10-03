import copy
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from training.build_full_page_resolution_colab import CONFIG_PATH, build
from training.full_page_comparison import package_comparison
from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.full_page_resolution import ARMS, ENGINE, PAGE_IDS, arm_configs, compare, prepare_subset


def fixture(tmp_path):
    dataset = tmp_path/'dataset'
    dataset.mkdir()
    image = dataset/'page.png'
    Image.new('RGB', (32, 64), 'white').save(image)
    rows = [{'id': page_id, 'image': 'page.png', 'sha256': digest(image),
             'width': 32, 'height': 64, 'split': 'validation',
             'eligible_for_training': False, 'text': 'Po\u017f\u0142a\u0142 b\u0142\u00e1waty',
             'annotation_notes': 'private model input exclusion'} for page_id in reversed(PAGE_IDS)]
    write_rows(dataset/'manifest.jsonl', rows)
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    config['dataset']['manifest_sha256'] = digest(dataset/'manifest.jsonl')
    return config, dataset


def test_model_specs_change_only_max_pixels():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    original = copy.deepcopy(config)
    arms = arm_configs(config)
    assert config == original
    one, four = arms.values()
    assert four['models'][ENGINE]['max_pixels'] == 4194304
    four['models'][ENGINE]['max_pixels'] = 1048576
    assert one == four == config
    v5 = json.loads((CONFIG_PATH.parent.parent/'full-page-comparison-v5/config.json').read_text(encoding='utf-8'))
    assert config['models'] == v5['models'] and config['dataset'] == v5['dataset']


def test_subset_is_ordered_reference_free_and_repeatable(tmp_path):
    config, dataset = fixture(tmp_path)
    original = (dataset/'manifest.jsonl').read_bytes()
    prepare_subset(config, dataset)
    prepare_subset(config, dataset)
    assert (dataset/'manifest.jsonl').read_bytes() == original
    inputs = read_rows(dataset/'resolution-inputs.jsonl')
    assert tuple(row['id'] for row in inputs) == PAGE_IDS
    assert all(set(row) == {'id','image','sha256','width','height','source_regions'}
               and row['source_regions'] == [] for row in inputs)
    assert 'private' not in json.dumps(inputs) and 'text' not in json.dumps(inputs)
    assert all(row['text'] == 'Po\u017f\u0142a\u0142 b\u0142\u00e1waty'
               for row in read_rows(dataset/'resolution-manifest.jsonl'))


@pytest.mark.parametrize('change', ['hash', 'training', 'selection', 'arms'])
def test_invalid_preparation_rejected(tmp_path, change):
    config, dataset = fixture(tmp_path)
    if change == 'hash':
        config['dataset']['manifest_sha256'] = '0'*64
    elif change == 'training':
        rows = read_rows(dataset/'manifest.jsonl')
        rows[0]['eligible_for_training'] = True
        write_rows(dataset/'manifest.jsonl', rows)
        config['dataset']['manifest_sha256'] = digest(dataset/'manifest.jsonl')
    elif change == 'selection':
        config['selection'].reverse()
    else:
        config['arms']['four-mp'] = 2000000
    with pytest.raises(ValueError):
        prepare_subset(config, dataset)


def test_metrics_include_errors_missing_loops_and_historical_spelling(tmp_path):
    config, dataset = fixture(tmp_path)
    prepare_subset(config, dataset)
    one = tmp_path/'arms/one-mp/predictions'/f'{ENGINE}-full-page.jsonl'
    one.parent.mkdir(parents=True)
    write_rows(one, [
        {'id': PAGE_IDS[0], 'text': '44. '*128, 'status': 'ok', 'format': 'plain',
         'finish_reason': 'length', 'token_limit_reached': True,
         'input_geometry': {'processed_pixels': 1000000}},
        {'id': PAGE_IDS[1], 'text': 'Po\u017f\u0142a\u0142 b\u0142\u00e1waty', 'status': 'ok', 'format': 'plain'},
        {'id': PAGE_IDS[2], 'text': 'Po\u0142a\u0142 b\u0142awaty', 'status': 'ok', 'format': 'plain'}])
    four = tmp_path/'arms/four-mp/predictions'/f'{ENGINE}-full-page.jsonl'
    four.parent.mkdir(parents=True)
    write_rows(four, [{'id': PAGE_IDS[0], 'text': 'fabricated', 'status': 'error'}])
    summary = compare(config, dataset, tmp_path)
    assert summary['reports']['one-mp']['cer_micro'] > 1
    assert summary['reports']['four-mp']['errors_or_missing'] == 3
    assert summary['reports']['four-mp']['cer_micro'] == 1
    table = read_rows(tmp_path/'scores/per-page.jsonl')
    assert len(table) == 6 and table[2]['cer'] > 0 and table[1]['cer'] == 0
    assert table[0]['input_geometry']['processed_pixels'] == 1000000
    assert table[1]['raw_poslal_count'] == 1 and table[2]['raw_poslal_count'] == 0
    assert summary['sota_claim'] is summary['model_promotion'] is False
    four.unlink()
    assert compare(config, dataset, tmp_path)['reports']['four-mp']['errors_or_missing'] == 3


def test_reference_change_rejected(tmp_path):
    config, dataset = fixture(tmp_path)
    prepare_subset(config, dataset)
    rows = read_rows(dataset/'resolution-manifest.jsonl')
    rows[0]['text'] = 'modernized'
    write_rows(dataset/'resolution-manifest.jsonl', rows)
    with pytest.raises(ValueError, match='references'):
        compare(config, dataset, tmp_path)


def test_worker_identity_and_actual_pixel_increase(tmp_path):
    config, dataset = fixture(tmp_path)
    prepare_subset(config, dataset)
    for label, arm in arm_configs(config).items():
        destination = tmp_path/'arms'/label/'predictions'
        destination.mkdir(parents=True)
        write_json(destination/'identity.json', {'engine': ENGINE,
            'spec': arm['models'][ENGINE],
            'input_sha256': digest(dataset/'resolution-inputs.jsonl')})
        write_rows(destination/f'{ENGINE}-full-page.jsonl', [
            {'id': key, 'status': 'ok', 'text': 'Po\u017f\u0142a\u0142 b\u0142\u00e1waty',
             'input_geometry': {'processed_pixels': ARMS[label]}} for key in PAGE_IDS])
    summary = compare(config, dataset, tmp_path)
    assert all(row['processed_pixels_increased'] is True for row in summary['paired'])
    assert all(row['cer_delta_four_minus_one'] == 0 for row in summary['paired'])
    assert all(report['worker_identity_verified'] for report in summary['reports'].values())
    identity = tmp_path/'arms/four-mp/predictions/identity.json'
    row = json.loads(identity.read_text())
    row['spec']['max_pixels'] = 1048576
    write_json(identity, row)
    with pytest.raises(ValueError, match='identity'):
        compare(config, dataset, tmp_path)


def test_notebook_compiles_with_one_link_no_uploads(tmp_path):
    target = tmp_path/'resolution.ipynb'
    build(target, 'a'*40)
    notebook = json.loads(target.read_text(encoding='utf-8'))
    code = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['id'], 'exec')
    assert 'files.upload' not in code and 'EnvBuilder(with_pip=True' not in code
    assert 'resolution-inputs.jsonl' in code and 'arm_configs(CONFIG)' in code
    assert "CONFIG['evidence_archive_name']" in code and 'files.download(str(archive))' in code
    assert 'prepare_subset' in code and 'v6/config.json' in code


def test_evidence_basename_and_filtering(tmp_path):
    (tmp_path/'raw.jsonl').write_text('{}\n')
    (tmp_path/'scan.png').write_bytes(b'image')
    (tmp_path/'weights.safetensors').write_bytes(b'weights')
    archive = package_comparison(tmp_path, 'full-page-resolution-v6-evidence.zip')
    package_comparison(tmp_path, archive.name)
    with zipfile.ZipFile(archive) as stream:
        assert set(stream.namelist()) == {'raw.jsonl', 'checksums.json'}
    for name in ('../escape.zip', 'C:\\escape.zip', '.hidden.zip', 'bad.txt'):
        with pytest.raises(ValueError):
            package_comparison(tmp_path, name)
