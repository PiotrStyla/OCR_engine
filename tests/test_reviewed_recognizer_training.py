import pytest

from training import prepare_reviewed_recognizer_training as module
from training.adjudicate_reviews import text_hash
from training.full_page_pilot import digest, write_json, write_rows
from training.merge_recognizer_reviewed_pool import verified_package


def fixture(tmp_path):
    root = tmp_path/'source'
    root.mkdir()
    rows = []
    for name, collection in [('a', 'train'), ('b', 'dev'), ('c', 'risk')]:
        image = root/(name+'.png')
        image.write_bytes(name.encode())
        text = '\u017f\u0292\u0307\u00e1'
        rows.append({'id': name, 'image': image.name, 'sha256': digest(image), 'text': text,
            'reviewed_text_sha256': text_hash(text), 'review_status': 'verified',
            'line_geometry_verified': True, 'geometry_decision': 'complete-line',
            'source_split': 'train', 'final_test': False, 'gold': False,
            'eligible_for_training': True, 'collection': collection, 'page_id': name})
    write_rows(root/'manifest.jsonl', rows)
    write_json(root/'checksums.json', {p.name: digest(p) for p in root.iterdir()})
    cfg = {'schema': 'slayer-recognizer-reviewed-exploratory-training-v1',
        'independent_benchmark': False, 'sota_claim': False,
        'bibliographic_independence_certified': False, 'automatic_production_promotion': False,
        'sources': [{'manifest_sha256': digest(root/'manifest.jsonl'),
            'checksums_sha256': digest(root/'checksums.json'), 'count': 3, 'kind': 'original-verified-pool'}],
        'excluded_collections': ['risk'], 'development_collection': 'dev',
        'expected_counts': {'approved_source': 3, 'excluded_work_family': 1,
            'historical_train': 1, 'historical_development': 1}}
    config = tmp_path/'config.json'
    write_json(config, cfg)
    return root, config, tmp_path/'output', rows, cfg


def test_exploratory_selection_preserves_raw_gates_and_text(tmp_path):
    root, config, out, rows, _ = fixture(tmp_path)
    before = digest(root/'checksums.json')
    result = module.prepare([root], config, out)
    assert result['counts']['historical_train'] == 1
    assert digest(root/'checksums.json') == before
    assert (out/'train/a.txt').read_text(encoding='utf-8') == rows[0]['text']
    assert not (out/'train/c.png').exists()
    manifest = module.read_rows(out/'manifest.jsonl')
    assert all(not r['eligible_for_training'] and not r['eligible_for_evaluation'] for r in manifest)
    assert not result['independent_benchmark']
    verified_package(out)
    with pytest.raises(FileExistsError):
        module.prepare([root], config, out)


@pytest.mark.parametrize('field', ['independent_benchmark', 'sota_claim',
    'bibliographic_independence_certified', 'automatic_production_promotion'])
def test_cannot_silently_promote_experimental_policy(tmp_path, field):
    root, config, out, _, cfg = fixture(tmp_path)
    cfg[field] = True
    write_json(config, cfg)
    with pytest.raises(ValueError, match='explicitly exploratory'):
        module.prepare([root], config, out)
    assert not out.exists()


@pytest.mark.parametrize('mutation', ['unverified', 'unchecked', 'test-source', 'modified-text', 'duplicate', 'shared-page'])
def test_rejects_invalid_or_overlapping_sources_before_output(tmp_path, mutation):
    root, config, out, rows, cfg = fixture(tmp_path)
    if mutation == 'unverified':
        rows[0]['review_status'] = 'proposed'
    elif mutation == 'unchecked':
        rows[0]['geometry_decision'] = 'unreviewed'
    elif mutation == 'test-source':
        rows[0]['source_split'] = 'test'
    elif mutation == 'modified-text':
        rows[0]['text'] += 'changed'
    elif mutation == 'duplicate':
        rows[1]['id'] = rows[0]['id']
    else:
        rows[1]['page_id'] = rows[0]['page_id']
    write_rows(root/'manifest.jsonl', rows)
    write_json(root/'checksums.json', {p.name: digest(p) for p in root.iterdir() if p.name != 'checksums.json'})
    cfg['sources'][0].update(manifest_sha256=digest(root/'manifest.jsonl'),
        checksums_sha256=digest(root/'checksums.json'))
    write_json(config, cfg)
    with pytest.raises(ValueError):
        module.prepare([root], config, out)
    assert not out.exists()


def test_stale_source_manifest_rejected(tmp_path):
    root, config, out, rows, _ = fixture(tmp_path)
    write_rows(root/'manifest.jsonl', rows[::-1])
    with pytest.raises(ValueError):
        module.prepare([root], config, out)
    assert not out.exists()


def test_colab_notebook_is_valid_pinned_and_has_no_manual_uploads(tmp_path):
    import nbformat
    from training.build_reviewed_recognizer_colab import build
    target = tmp_path/'notebook.ipynb'
    build(target, 'a'*40)
    notebook = nbformat.read(target, as_version=4)
    nbformat.validate(notebook)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
    for code in sources:
        compile(code, 'notebook-cell', 'exec')
    assert "with_pip=False" in sources[0]
    assert "'fetch', 'origin', 'main'" in sources[0]
    assert "'2.11': '0.17.0'" in sources[0]
    assert "'2.10': '0.16.0'" in sources[0]
    assert "'--no-deps'" in sources[0]
    assert 'training.preflight_reviewed_recognizer' in sources[0]
    assert 'files.upload' not in '\n'.join(sources)
    assert 'run_reviewed_recognizer_colab' in sources[1]
    assert 'result.zip' in sources[2]
    with pytest.raises(ValueError):
        build(target, 'short-sha')


def test_development_normalization_never_modernizes_historical_spelling():
    from training.run_reviewed_recognizer_colab import normalize
    assert normalize('\u017f \u00e1 \u0292\u0307') == '\u017f \u00e1 \u0292\u0307'


def test_actual_model_preflight_precedes_baseline_and_uses_training_sample():
    import inspect
    from training import run_reviewed_recognizer_colab as runner
    source = inspect.getsource(runner.run)
    assert source.index("preflight(base, corpus/'train'") < source.index('baseline = evaluate(')
    assert "'torchao'" in source
    assert 'adapter-preflight.json' in source
