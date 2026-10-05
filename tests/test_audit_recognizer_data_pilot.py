import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from tests.test_recognizer_data_pilot import bundle, predictions
from training import audit_recognizer_data_pilot as audit_module
from training import recognizer_data_pilot as pilot
from training.build_annotation_review import build
from training.full_page_pilot import digest, read_rows, write_json, write_rows


def generation(engine):
    geometry = {'original_width': 120, 'original_height': 20}
    if engine == 'qwen3-vl-4b':
        geometry.update(image_grid_thw=[[1, 16, 64]], patch_size=16, merge_size=2,
            processed_width=1024, processed_height=256, processed_pixels=262144,
            visual_tokens=256, min_pixels=262144, max_pixels=1048576)
    else:
        geometry['processor_tensor_shape'] = [1, 3, 384, 384]
    return {'id': 'line', 'status': 'ok', 'text': '\u017f\u00e1\u0247', 'generated_tokens': 2,
        'finish_reason': 'eos', 'token_limit_reached': False,
        'generation_trace': {'generated_token_ids': [42, audit_module.EOS[engine][0]],
                             'eos_token_ids': audit_module.EOS[engine]}, 'input_geometry': geometry}


@pytest.mark.parametrize('engine', pilot.ENGINES)
def test_generation_valid(engine):
    audit_module.validate_generation(generation(engine), {'width': 120, 'height': 20},
        {'max_new_tokens': 8, 'min_pixels': 262144, 'max_pixels': 1048576}, engine)


@pytest.mark.parametrize('field', ['eos', 'tokens', 'length', 'finish', 'cap', 'dimensions', 'grid'])
def test_generation_rejects_inconsistent_trace(field):
    row = generation('qwen3-vl-4b')
    if field == 'eos':
        row['generation_trace']['eos_token_ids'] = [2]
    elif field == 'tokens':
        row['generation_trace']['generated_token_ids'] = [151645, 42]
    elif field == 'length':
        row['generated_tokens'] = 20
    elif field == 'finish':
        row['finish_reason'] = 'length'
    elif field == 'cap':
        row['token_limit_reached'] = True
    elif field == 'dimensions':
        row['input_geometry']['original_width'] = 300
    else:
        row['input_geometry']['visual_tokens'] = 999
    with pytest.raises(ValueError):
        audit_module.validate_generation(row, {'width': 120, 'height': 20},
            {'max_new_tokens': 8, 'min_pixels': 262144, 'max_pixels': 1048576}, 'qwen3-vl-4b')


def test_capped_generation_and_error_are_preserved():
    row = generation('qwen3-vl-4b')
    row.update(generated_tokens=8, finish_reason='length', token_limit_reached=True)
    row['generation_trace']['generated_token_ids'] = [42]*8
    spec = {'max_new_tokens': 8, 'min_pixels': 262144, 'max_pixels': 1048576}
    audit_module.validate_generation(row, {'width': 120, 'height': 20}, spec, 'qwen3-vl-4b')
    assert not pilot.healthy(row)
    audit_module.validate_generation({'status': 'error', 'text': '', 'error': 'OOM'}, {}, {}, 'qwen3-vl-4b')
    with pytest.raises(ValueError):
        audit_module.validate_generation({'status': 'error', 'text': 'fake', 'error': 'OOM'}, {}, {}, 'qwen3-vl-4b')


@pytest.fixture
def evidence(bundle, tmp_path, monkeypatch):
    source, input_archive, config = bundle
    write_rows(source/'excluded.jsonl', [])
    write_json(source/'report.json', {'lines': 1, 'training_examples_created': 0})
    paths = [p for p in source.rglob('*') if p.is_file() and p.name != 'checksums.json']
    write_json(source/'checksums.json', {p.relative_to(source).as_posix(): digest(p) for p in paths})
    with zipfile.ZipFile(input_archive, 'w') as stream:
        for p in source.rglob('*'):
            if p.is_file():
                stream.write(p, p.relative_to(source).as_posix())
    config['dataset']['archive_sha256'] = digest(input_archive)
    config['packages'] = ['transformers==4.57.6', 'pillow==11.3.0']
    for engine in pilot.ENGINES:
        config['models'][engine].update(max_new_tokens=8, min_pixels=262144, max_pixels=1048576)
    config_path = tmp_path/'frozen.json'
    write_json(config_path, config)
    work = tmp_path/'run'
    work.mkdir()
    predictions(source, work/'predictions', config, ['\u017f\u00e1\u0247']*2)
    for engine in pilot.ENGINES:
        folder = work/'predictions'/engine
        write_rows(folder/pilot.FILES[engine], [generation(engine)])
        write_json(folder/'environment.json', {'packages': {'transformers': '4.57.6', 'Pillow': '11.3.0', 'torch': 'mock'},
            'gpu': 'mock-GPU', 'reference_text_sent_to_model': False, 'load_error': None})
    pilot.combine(config, source, work/'predictions', work/'combined')
    write_json(work/'config.json', config)
    write_json(work/'code-provenance.json', {'code_revision': 'a'*40,
        'runner_sha256': digest(pilot.__file__), 'qwen_runner_sha256': digest(Path(pilot.__file__).with_name('full_page_pilot.py')),
        'reference_text_sent_to_models': False, 'gpu_execution_validated_locally': False})
    write_json(work/'staging-report.json', {'lines': 1, 'labels_sent_to_models': False,
        'training_examples_created': 0, 'input_sha256': digest(source/'inference-inputs.jsonl')})
    write_json(work/'bootstrap.json', {'status': 'ok', 'gpu': 'mock-GPU'})
    write_json(work/'worker-exits.json', {e: {'status': 'finished', 'exit_code': 0} for e in pilot.ENGINES})
    (work/'preflight.log').write_text(json.dumps({'packages': {'transformers': '4.57.6', 'pillow': '11.3.0'},
        'torch': 'mock', 'gpu': 'mock-GPU'})+'\n', encoding='utf-8')
    (work/'dataset').mkdir()
    for name in ('manifest.jsonl', 'inference-inputs.jsonl', 'report.json', 'excluded.jsonl'):
        (work/'dataset'/name).write_bytes((source/name).read_bytes())
    def git_blob(command, **kwargs):
        name = command[-1].split(':training/')[1]
        return Path(pilot.__file__).with_name(name).read_bytes()
    monkeypatch.setattr(audit_module.subprocess, 'check_output', git_blob)
    return work, input_archive, config_path


def test_full_audit_recomputes_from_raw_predictions(evidence, tmp_path):
    work, bundle_path, config = evidence
    report = audit_module.audit(pilot.package(work), bundle_path, config, tmp_path/'audit', 'a'*40)
    assert report['statuses'] == {'teacher-agreement-proposal': 1}
    assert report['proposals_and_report_recomputed'] and report['generation_traces_verified']
    assert report['training_examples_created'] == report['gold_labels_created'] == 0
    assert not report['sota_claim'] and not report['glyph_counts_are_recall_or_accuracy']


@pytest.mark.parametrize('file,key,value', [('code-provenance.json', 'reference_text_sent_to_models', True),
    ('staging-report.json', 'input_sha256', 'wrong'), ('bootstrap.json', 'status', 'error'),
    ('predictions/qwen3-vl-4b/identity.json', 'runner_sha256', 'wrong'),
    ('combined/report.json', 'training_examples_created', 1)])
def test_full_audit_rejects_semantic_tampering(evidence, tmp_path, file, key, value):
    work, bundle_path, config = evidence
    target = work/file
    data = json.loads(target.read_text(encoding='utf-8'))
    data[key] = value
    write_json(target, data)
    with pytest.raises(ValueError):
        audit_module.audit(pilot.package(work), bundle_path, config, tmp_path/'audit', 'a'*40)


def test_context_review_has_separate_schema_and_unapproved_geometry(bundle, tmp_path):
    source, _, _ = bundle
    region = source/'region.jpg'
    Image.new('RGB', (240, 100), 'white').save(region)
    context = {'line': {'image': 'region.jpg', 'sha256': digest(region), 'width': 240, 'height': 100,
                        'text': 'Whole region text'}}
    build(source/'manifest.jsonl', tmp_path/'review', contexts=context, geometry_review=True)
    html = (tmp_path/'review/index.html').read_text(encoding='utf-8')
    payload = json.loads(html.split('<script id="source-data" type="application/json">')[1].split('</script>')[0])
    assert payload['geometry_review_required'] is True and len(payload['pages']) == 1
    assert payload['pages'][0]['text'] == '\u017f\u00e1\u0247'
    assert digest(tmp_path/'review'/payload['pages'][0]['context']['image']) == digest(region)
    js = (tmp_path/'review/app.js').read_text(encoding='utf-8')
    assert 'slayer-recognizer-line-review-v1' in js and 'geometry_decision' in js
    assert 'context_image_sha256' in js


def test_context_review_rejects_missing_or_mismatched_context(bundle, tmp_path):
    source, _, _ = bundle
    with pytest.raises(ValueError):
        build(source/'manifest.jsonl', tmp_path/'review', geometry_review=True)
    with pytest.raises(ValueError):
        build(source/'manifest.jsonl', tmp_path/'review', contexts={}, geometry_review=True)
