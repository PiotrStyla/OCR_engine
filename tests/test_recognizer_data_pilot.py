import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from training import recognizer_data_pilot as pilot
from training.prepare_recognizer_data_pilot import choose
from training.full_page_pilot import digest, read_rows, write_json, write_rows


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path/'source'
    (source/'images').mkdir(parents=True)
    Image.new('L', (120, 20), 240).save(source/'images/line.png')
    row = {'id': 'line', 'image': 'images/line.png', 'sha256': digest(source/'images/line.png'),
           'width': 120, 'height': 20, 'text': '\u017f\u00e1\u0247', 'page_id': 'page', 'collection': 'train',
           'source_split': 'train', 'split': 'training-review', 'eligible_for_training': False,
           'final_test': False, 'dataset': 'source-repo', 'revision': 'source-revision'}
    write_rows(source/'manifest.jsonl', [row])
    inputs = [{key: row[key] for key in ('id', 'image', 'sha256', 'width', 'height')}
              | {'source_regions': []}]
    write_rows(source/'inference-inputs.jsonl', inputs)
    files = list(source.rglob('*'))
    write_json(source/'checksums.json', {p.relative_to(source).as_posix(): digest(p) for p in files if p.is_file()})
    archive = tmp_path/'input.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        for path in source.rglob('*'):
            if path.is_file():
                stream.write(path, path.relative_to(source).as_posix())
    config = {'dataset': {'archive_sha256': digest(archive), 'manifest_sha256': digest(source/'manifest.jsonl'),
                         'lines': 1, 'forbidden_collections': ['test'], 'source_repo': 'source-repo',
                         'source_revision': 'source-revision'},
              'models': {engine: {'revision': 'frozen'} for engine in pilot.ENGINES}}
    return source, archive, config


def test_stage_reference_free_and_repeat_safe(bundle, tmp_path):
    _, archive, config = bundle
    target = tmp_path/'staged'
    result = pilot.stage(config, archive, target)
    assert result['training_examples_created'] == 0
    assert result['labels_sent_to_models'] is False
    assert 'text' not in read_rows(target/'inference-inputs.jsonl')[0]
    with pytest.raises(FileExistsError):
        pilot.stage(config, archive, target)


@pytest.mark.parametrize('kind', ['archive-hash', 'manifest-hash', 'forbidden', 'revision', 'count'])
def test_stage_rejects_changed_provenance(bundle, tmp_path, kind):
    _, archive, config = bundle
    changes = {'archive-hash': ('archive_sha256', 'bad'), 'manifest-hash': ('manifest_sha256', 'bad'),
               'forbidden': ('forbidden_collections', ['train']), 'revision': ('source_revision', 'bad'),
               'count': ('lines', 2)}
    key, value = changes[kind]
    config['dataset'][key] = value
    with pytest.raises(ValueError):
        pilot.stage(config, archive, tmp_path/'staged')
    assert not (tmp_path/'staged').exists()


@pytest.mark.parametrize('name', ['../outside.png', 'images\\bad.png', 'images/./line.png', '/absolute.png', 'model.py'])
def test_stage_rejects_unsafe_members(bundle, tmp_path, name):
    _, archive, config = bundle
    with zipfile.ZipFile(archive, 'a') as stream:
        stream.writestr(name, b'bad')
    config['dataset']['archive_sha256'] = digest(archive)
    with pytest.raises(ValueError):
        pilot.stage(config, archive, tmp_path/'staged')


def predictions(source, root, config, texts, capped=False):
    for engine, text in zip(pilot.ENGINES, texts):
        folder = root/engine
        folder.mkdir(parents=True)
        runner = Path(pilot.__file__).with_name('full_page_pilot.py') if engine == 'qwen3-vl-4b' else Path(pilot.__file__)
        write_json(folder/'identity.json', {'engine': engine, 'spec': config['models'][engine],
                   'input_sha256': digest(source/'inference-inputs.jsonl'), 'runner_sha256': digest(runner)})
        write_rows(folder/pilot.FILES[engine], [{'id': 'line', 'status': 'ok', 'text': text,
                   'finish_reason': 'eos', 'token_limit_reached': capped}])


@pytest.mark.parametrize('texts,status', [(['\u017f\u00e1\u0247', '\u017f\u00e1\u0247'], 'teacher-agreement-proposal'),
    (['\u017f\u00e1\u0247', 'sae'], 'teacher-disagreement'), (['', ''], 'teacher-abstention'),
    ([None, '\u017f\u00e1\u0247'], 'teacher-abstention'), (['\u00e1', 'a\u0301'], 'teacher-agreement-proposal')])
def test_agreement_never_promotes_training_or_modernizes(bundle, tmp_path, texts, status):
    source, _, config = bundle
    root = tmp_path/'predictions'
    predictions(source, root, config, texts)
    report = pilot.combine(config, source, root, tmp_path/'combined')
    row = read_rows(tmp_path/'combined/proposals.jsonl')[0]
    assert row['status'] == status
    assert row['review_required'] and not row['gold_label'] and not row['eligible_for_training']
    assert not row['line_geometry_verified']
    assert report['training_examples_created'] == report['gold_labels_created'] == 0
    assert row['teachers'][pilot.ENGINES[0]]['text'] == texts[0]


def test_missing_and_capped_predictions_abstain(bundle, tmp_path):
    source, _, config = bundle
    report = pilot.combine(config, source, tmp_path/'missing', tmp_path/'empty')
    assert report['statuses'] == {'teacher-abstention': 1}
    predictions(source, tmp_path/'predictions', config, ['same', 'same'], capped=True)
    report = pilot.combine(config, source, tmp_path/'predictions', tmp_path/'capped')
    assert report['statuses'] == {'teacher-abstention': 1}


@pytest.mark.parametrize('kind', ['engine', 'runner', 'inputs', 'duplicate', 'unknown', 'no-identity'])
def test_consensus_rejects_wrong_identity_or_coverage(bundle, tmp_path, kind):
    source, _, config = bundle
    root = tmp_path/'predictions'
    predictions(source, root, config, ['same', 'same'])
    folder = root/pilot.ENGINES[0]
    identity = folder/'identity.json'
    if kind == 'no-identity':
        identity.unlink()
    elif kind in ('engine', 'runner', 'inputs'):
        value = json.loads(identity.read_text())
        key = {'engine': 'engine', 'runner': 'runner_sha256', 'inputs': 'input_sha256'}[kind]
        value[key] = 'wrong'
        write_json(identity, value)
    else:
        path = folder/pilot.FILES[pilot.ENGINES[0]]
        rows = read_rows(path)
        if kind == 'duplicate':
            rows += rows
        else:
            rows[0]['id'] = 'unknown'
        write_rows(path, rows)
    with pytest.raises(ValueError):
        pilot.combine(config, source, root, tmp_path/'combined')


def test_package_keeps_failures_and_excludes_scans_weights_code(tmp_path):
    write_json(tmp_path/'error.json', {'error': 'failed'})
    for name in ('scan.png', 'model.safetensors', 'code.py'):
        (tmp_path/name).write_bytes(b'not evidence')
    archive = pilot.package(tmp_path)
    with zipfile.ZipFile(archive) as stream:
        assert set(stream.namelist()) == {'error.json', 'checksums.json'}
        checksums = json.loads(stream.read('checksums.json'))
        assert checksums['error.json'] == hashlib.sha256(stream.read('error.json')).hexdigest()


def test_selection_deterministic_and_bounded():
    rows = [{'id': str(i), 'collection': 'c'+str(i%4), 'page_id': 'p'+str(i%8),
             'text': '\u017f\u00e1\u0247' if i%2 else '\u00e1'} for i in range(32)]
    result = choose(rows, 12, 4, 2)
    assert result == choose(list(reversed(rows)), 12, 4, 2)
    from collections import Counter
    assert max(Counter(r['page_id'] for r in result).values()) <= 2
    assert max(Counter(r['collection'] for r in result).values()) <= 4
    with pytest.raises(ValueError):
        choose(rows, 32, 1, 1)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    from training import prepare_recognizer_data_pilot as prepare
    root = tmp_path/'corpus'
    (root/'train').mkdir(parents=True)
    Image.new('L', (100, 20), 230).save(root/'train/line.png')
    (root/'train/line.txt').write_text('\u017f\u00e1\u0247\n', encoding='utf-8')
    row = {'id': 'line', 'split': 'train', 'collection': 'train', 'page_id': 'page',
           'source_region_id': 'region', 'source_image_sha256': 'source-hash',
           'image_sha256': digest(root/'train/line.png'), 'text_sha256': digest(root/'train/line.txt'),
           'license': 'CC-BY-3.0'}
    write_rows(root/'manifest.jsonl', [row])
    train, test, validation, holdout = [tmp_path/name for name in ('train.jsonl', 'test.jsonl', 'validation.jsonl', 'holdout.json')]
    write_rows(train, [{'id': 'region', 'split': 'train', 'page_id': 'page', 'collection': 'train',
                       'image_sha256': 'source-hash', 'bbox': [0, 0, 100, 20], 'text': '\u017f\u00e1\u0247'}])
    write_rows(test, [{'id': 'test', 'collection': 'test', 'page_id': 'test-page', 'image_sha256': 'test-hash'}])
    write_rows(validation, [{'id': 'val', 'collection': 'validation', 'page_id': 'val-page', 'sha256': 'val-hash'}])
    write_json(holdout, {'dataset': prepare.DATASET, 'revision': prepare.REVISION,
                         'regions': [{'id': 'hold', 'collection': 'holdout', 'page_id': 'hold-page'}]})
    monkeypatch.setattr(prepare, 'SOURCE_MANIFEST', digest(root/'manifest.jsonl'))
    monkeypatch.setattr(prepare, 'METADATA_SHA256', {'train': digest(train), 'test': digest(test)})
    return prepare, root, train, test, validation, holdout


def test_preparer_preserves_text_and_no_training_approval(corpus, tmp_path):
    prepare, *paths = corpus
    result = prepare.prepare(*paths, tmp_path/'pilot', count=1)
    assert result['lines'] == 1 and result['training_examples_created'] == 0
    row = read_rows(tmp_path/'pilot/manifest.jsonl')[0]
    assert row['text'] == '\u017f\u00e1\u0247'
    assert not row['eligible_for_training'] and not row['eligible_for_evaluation']
    assert result['forbidden_collections'] == ['holdout', 'test', 'validation']


@pytest.mark.parametrize('kind', ['collection', 'page', 'crop-hash', 'source-hash'])
def test_preparer_rejects_validation_overlap(corpus, tmp_path, kind):
    prepare, root, train, test, validation, holdout = corpus
    row = read_rows(validation)[0]
    if kind == 'collection':
        row['collection'] = 'train'
    elif kind == 'page':
        row['page_id'] = 'page'
    elif kind == 'crop-hash':
        row['sha256'] = digest(root/'train/line.png')
    else:
        row['sha256'] = 'source-hash'
    write_rows(validation, [row])
    with pytest.raises(ValueError, match='overlap'):
        prepare.prepare(root, train, test, validation, holdout, tmp_path/'pilot', count=1)
    assert not (tmp_path/'pilot').exists()


def test_notebook_builder_compiles_and_pins_code(bundle, tmp_path):
    from training.build_recognizer_data_pilot_colab import build
    _, _, config = bundle
    path, notebook = tmp_path/'config.json', tmp_path/'pilot.ipynb'
    write_json(path, config)
    build(notebook, path, 'a'*40)
    cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
    code = '\n'.join(''.join(cell['source']) for cell in cells if cell['cell_type'] == 'code')
    for cell in cells:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['id'], 'exec')
    assert "'fetch', 'origin', 'main'" in code
    assert 'venv.EnvBuilder(with_pip=False, system_site_packages=True)' in code
    assert "'-m', 'ensurepip'" not in code
    assert 'files.upload' not in code
    assert 'CODE_REVISION = '+repr('a'*40) in code
    assert 'package(WORK)' in code and 'worker-exits.json' in code
    with pytest.raises(ValueError):
        build(notebook, path, 'short-sha')


def test_notebook_failure_path_still_downloads_evidence(bundle, tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from training.build_recognizer_data_pilot_colab import build
    source, _, config = bundle
    config['packages'] = ['transformers==4.57.6']
    path, notebook = tmp_path/'config.json', tmp_path/'pilot.ipynb'
    write_json(path, config)
    build(notebook, path, 'a'*40)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    downloads = []
    google, colab = ModuleType('google'), ModuleType('google.colab')
    colab.files = SimpleNamespace(download=lambda path: downloads.append(path))
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    scope = {'WORK': tmp_path, 'CONFIG': config, 'dataset': source, 'ENGINES': pilot.ENGINES,
             'write_json': write_json, 'combine': pilot.combine, 'package': pilot.package, 'json': json}
    for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']:
        if cell['id'] in ('environment', 'inference', 'consensus', 'download'):
            exec(''.join(cell['source']), scope)
    assert len(downloads) == 1 and Path(downloads[0]).exists()
    report = json.loads((tmp_path/'combined/report.json').read_text())
    assert report['statuses'] == {'teacher-abstention': 1}
    assert json.loads((tmp_path/'bootstrap.json').read_text())['status'] == 'error'
    assert all(row['status'] == 'not-started' for row in json.loads((tmp_path/'worker-exits.json').read_text()).values())


@pytest.mark.parametrize('failed', [False, True])
def test_trocr_worker_reference_free_resume_and_error_coverage(bundle, tmp_path, monkeypatch, failed):
    import sys
    from types import SimpleNamespace
    source, _, config = bundle
    cuda = SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None, max_memory_allocated=lambda: 0, get_device_name=lambda _: 'mock-GPU')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda, manual_seed=lambda _: None))
    monkeypatch.setattr(pilot.importlib.metadata, 'version', lambda _: 'mock')
    seen = []
    def loader(_spec):
        if failed:
            raise RuntimeError('mock model failed')
        def infer(_path, row):
            seen.append(row)
            return {'text': '\u017f\u00e1\u0247', 'finish_reason': 'eos', 'token_limit_reached': False}
        return infer
    monkeypatch.setattr(pilot, 'load_trocr', loader)
    out = tmp_path/'worker'
    result = pilot.worker('trocr-mixed-v3', config, source/'inference-inputs.jsonl', out)
    assert result == {'lines': 1, 'errors': int(failed)}
    assert not seen or 'text' not in seen[0]
    assert pilot.worker('trocr-mixed-v3', config, source/'inference-inputs.jsonl', out) == result
    rows = read_rows(out/'trocr-lines.jsonl')
    assert len(rows) == 1 and rows[0]['status'] == ('error' if failed else 'ok')
    if failed:
        assert rows[0]['text'] == '' and 'mock model failed' in rows[0]['error']
