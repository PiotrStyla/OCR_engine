import hashlib
import io
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import zipfile

from PIL import Image
import pytest

from training.full_page_comparison import compare, package_comparison, repetition_flags, stage_bundle
from training.full_page_pilot import digest, load_qwen, read_rows, run_worker, write_json, write_rows
from training.build_full_page_comparison_colab import build


def bundle_fixture(tmp_path, extra=None):
    image = io.BytesIO()
    Image.new('RGB', (16, 32), 'white').save(image, format='PNG')
    row = {'id':'p', 'image':'images/0000.png', 'sha256':hashlib.sha256(image.getvalue()).hexdigest(),
           'width':16, 'height':32, 'split':'validation', 'collection':'C', 'final_test':False,
           'reference_status':'single-review-draft-not-gold', 'eligible_for_training':False,
           'text':'błáwaty ſłowo', 'annotation_notes':'Do not send this note to the model'}
    prediction = {'id':'p', 'text':row['text'], 'raw_text':row['text'], 'status':'ok',
                  'finish_reason':'eos', 'token_limit_reached':False}
    files = {'images/0000.png':image.getvalue(),
             'manifest.jsonl':(json.dumps(row)+'\n').encode(),
             'retained-projected-predictions.jsonl':(json.dumps(prediction)+'\n').encode()}
    if extra:
        files.update(extra)
    files['checksums.json'] = json.dumps({name:hashlib.sha256(data).hexdigest() for name,data in files.items()}).encode()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as stream:
        for name,data in files.items():
            stream.writestr(name, data)
    data = archive.getvalue()
    config = {'dataset':{'repo':'Owner/repo', 'revision':'a'*40, 'path':'data/bundle.zip',
                        'archive_sha256':hashlib.sha256(data).hexdigest(), 'pages':1,
                        'manifest_sha256':hashlib.sha256(files['manifest.jsonl']).hexdigest(),
                        'baseline_sha256':hashlib.sha256(files['retained-projected-predictions.jsonl']).hexdigest()},
              'baseline':{'source':'retained-projected-predictions.jsonl'}, 'scope_policy':{'sota_claim':False}}
    opener = lambda *args,**kwargs: io.BytesIO(data)
    return config, opener


def test_frozen_bundle_is_repeatable_and_has_reference_free_inputs(tmp_path):
    config, opener = bundle_fixture(tmp_path)
    dataset = tmp_path/'dataset'
    stage_bundle(config, dataset, opener)
    before = (dataset/'manifest.jsonl').read_bytes()
    stage_bundle(config, dataset, lambda *args,**kwargs: pytest.fail('Must use checked cache'))
    assert (dataset/'manifest.jsonl').read_bytes() == before
    inputs = read_rows(dataset/'inference-inputs.jsonl')
    assert set(inputs[0]) == {'id','image','sha256','width','height','source_regions'}
    assert inputs[0]['source_regions'] == []
    assert 'błáwaty' not in json.dumps(inputs) and 'note' not in json.dumps(inputs)
    provenance = json.loads((dataset/'staging-provenance.json').read_text())
    assert provenance['reference_text_sent_to_model'] is provenance['training_eligibility'] is False


@pytest.mark.parametrize('name', ['../escape.json', 'C:/escape.json', 'images\\escape.png', 'runner.py'])
def test_unsafe_bundle_rejected_before_images_materialize(tmp_path, name):
    config, opener = bundle_fixture(tmp_path, {name:b'bad'})
    with pytest.raises(ValueError):
        stage_bundle(config, tmp_path/'dataset', opener)
    assert not (tmp_path/'dataset/images').exists()


def test_hash_and_existing_staged_changes_are_rejected(tmp_path):
    config, opener = bundle_fixture(tmp_path)
    config['dataset']['archive_sha256'] = '0'*64
    with pytest.raises(ValueError, match='checksum'):
        stage_bundle(config, tmp_path/'wrong', opener)
    config, opener = bundle_fixture(tmp_path)
    stage_bundle(config, tmp_path/'dataset', opener)
    (tmp_path/'dataset/manifest.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='differs'):
        stage_bundle(config, tmp_path/'dataset', opener)


def test_comparison_keeps_errors_missing_and_historical_spelling(tmp_path):
    config, opener = bundle_fixture(tmp_path)
    dataset = tmp_path/'dataset'
    stage_bundle(config, dataset, opener)
    predictions = tmp_path/'candidate.jsonl'
    write_rows(predictions, [{'id':'p', 'status':'error', 'text':'ignored'}])
    summary = compare(config, dataset, predictions, tmp_path/'scores')
    assert summary['reports']['ovis-retained']['cer_micro'] == 0
    assert summary['reports']['qwen3-vl-4b']['cer_micro'] == 1
    assert summary['reports']['qwen3-vl-4b']['errors_or_missing'] == 1
    assert summary['sota_claim'] is summary['model_promotion'] is False
    assert summary['spatial_reading_order_metrics_available'] is False
    predictions.unlink()
    assert compare(config, dataset, predictions, tmp_path/'missing')['reports']['qwen3-vl-4b']['cer_micro'] == 1
    write_rows(predictions, [{'id':'p', 'status':'ok', 'text':'bławaty słowo', 'format':'plain'}])
    assert compare(config, dataset, predictions, tmp_path/'modernized')['reports']['qwen3-vl-4b']['cer_micro'] > 0


def test_evidence_is_rerunnable_and_never_contains_scans_weights_or_code(tmp_path):
    write_json(tmp_path/'config.json', {'scope':'test'})
    (tmp_path/'per-page.csv').write_text('id,cer\np,1\n')
    (tmp_path/'scan.png').write_bytes(b'scan')
    (tmp_path/'weights.safetensors').write_bytes(b'weights')
    (tmp_path/'runner.py').write_text('print(1)')
    (tmp_path/'cer-comparison.png').write_bytes(b'chart')
    for _ in range(2):
        with zipfile.ZipFile(package_comparison(tmp_path)) as stream:
            assert set(stream.namelist()) == {'config.json','per-page.csv','cer-comparison.png','checksums.json'}
            checksums = json.loads(stream.read('checksums.json'))
            assert all(hashlib.sha256(stream.read(name)).hexdigest() == sha for name,sha in checksums.items())


def test_repetition_is_flag_only_and_does_not_rewrite_text():
    text = 'ſ'*64
    assert repetition_flags(text)['suspicious_repetition'] is True
    assert repetition_flags('ſłowo błáwaty')['suspicious_repetition'] is False
    assert text == 'ſ'*64


def test_notebook_is_one_click_frozen_and_cells_compile(tmp_path):
    target = tmp_path/'comparison.ipynb'
    build(target, 'a'*40)
    notebook = json.loads(target.read_text(encoding='utf-8'))
    combined = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    assert 'files.upload' not in combined and 'with_pip=True' not in combined
    assert 'ensurepip' not in '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code') or 'lack ensurepip' in combined
    assert 'Qwen3VLForConditionalGeneration' in combined
    assert 'd7a0bed7eabb68ae1231ea68f2a4461dd4c190ef' in combined
    assert 'package_comparison(WORK)' in combined
    assert 'files.download(str(archive))' in combined
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and not cell['source'][0].startswith('%'):
            compile(''.join(cell['source']), cell['id'], 'exec')
        assert not cell.get('outputs')


@pytest.mark.parametrize('ended', [True, False])
def test_qwen_loader_uses_explicit_class_and_retains_raw_tokens(tmp_path, monkeypatch, ended):
    class Tokens(list):
        def __getitem__(self, key):
            value = super().__getitem__(key)
            return Tokens(value) if isinstance(key, slice) else value
        def tolist(self):
            return list(self)
    class Inputs(dict):
        def to(self, device):
            return self
    calls = {}
    ids = [1,2,99] if ended else [1,2,3,4]
    model = SimpleNamespace(device='cuda', generation_config=SimpleNamespace(eos_token_id=99))
    model.eval = lambda: model
    def generate(**kwargs):
        calls['generate'] = kwargs
        return [Tokens([10,11,12]+ids)]
    model.generate = generate
    def load_model(*args, **kwargs):
        calls['model'] = kwargs
        return model
    processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16, merge_size=2),
                                tokenizer=SimpleNamespace(pad_token_id=0))
    def template(messages, **kwargs):
        calls['messages'] = messages
        return Inputs(input_ids=SimpleNamespace(shape=(1,3)),
                      image_grid_thw=SimpleNamespace(tolist=lambda:[[1,4,2]]))
    processor.apply_chat_template = template
    processor.decode = lambda *args,**kwargs: 'błáwaty ſłowo'
    fake = SimpleNamespace(float16='float16', inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=load_model),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args,**kwargs:processor),
        BitsAndBytesConfig=lambda **kwargs:kwargs))
    path = tmp_path/'page.png'
    Image.new('RGB', (32,64)).save(path)
    spec = {'repo':'Qwen/model', 'revision':'a'*40, 'min_pixels':256, 'max_pixels':1024,
            'prompt':'Preserve ſ and á', 'max_new_tokens':4}
    prediction = load_qwen(spec)(path, {'id':'p', 'text':'SECRET'})['qwen3-vl-4b-full-page']
    assert 'SECRET' not in str(calls['messages'])
    assert prediction['text'] == 'błáwaty ſłowo'
    assert prediction['generation_trace']['generated_token_ids'] == ids
    assert prediction['finish_reason'] == ('eos' if ended else 'length')
    assert prediction['token_limit_reached'] is not ended
    assert prediction['input_geometry']['processed_pixels'] == 2048
    assert calls['model']['device_map'] == {'':0} and calls['model']['attn_implementation'] == 'sdpa'
    assert calls['model']['quantization_config']['bnb_4bit_compute_dtype'] == 'float16'


def test_qwen_worker_load_failure_is_saved_and_not_retried_on_resume(tmp_path, monkeypatch):
    from training import full_page_pilot as pilot
    config, opener = bundle_fixture(tmp_path)
    dataset = tmp_path/'dataset'
    stage_bundle(config, dataset, opener)
    fake = SimpleNamespace(manual_seed=lambda seed:None, cuda=SimpleNamespace(
        is_available=lambda:True, reset_peak_memory_stats=lambda:None,
        synchronize=lambda:None, max_memory_allocated=lambda:0, get_device_name=lambda index:'fixture'))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setattr(pilot.importlib.metadata, 'version', lambda name:'fixture')
    calls = []
    def fail(spec):
        calls.append(spec)
        raise ImportError('fixture Qwen load failed')
    monkeypatch.setattr(pilot, 'load_qwen', fail)
    config['models'] = {'qwen3-vl-4b':{'revision':'a'*40}}
    output = tmp_path/'predictions'
    result = run_worker('qwen3-vl-4b', config, dataset/'inference-inputs.jsonl', output)
    assert result['failed_pages'] == 1
    before = (output/'qwen3-vl-4b-full-page.jsonl').read_bytes()
    run_worker('qwen3-vl-4b', config, dataset/'inference-inputs.jsonl', output)
    assert len(calls) == 1 and (output/'qwen3-vl-4b-full-page.jsonl').read_bytes() == before
