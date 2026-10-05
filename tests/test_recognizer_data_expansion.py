import json
import shutil
import zipfile

from PIL import Image
import pytest

from tests.test_recognizer_data_pilot import corpus, bundle
from training import prepare_recognizer_data_pilot as preparer
from training import recognizer_data_pilot as runner
from training.full_page_pilot import digest, read_rows, write_json, write_rows


def seal(root):
    write_json(root/'checksums.json', {p.relative_to(root).as_posix(): digest(p)
        for p in root.rglob('*') if p.is_file() and p.name != 'checksums.json'})


def prior_pool(tmp_path, source):
    pool = tmp_path/'pool'
    (pool/'images').mkdir(parents=True)
    original = read_rows(source/'manifest.jsonl')[0]
    shutil.copyfile(source/'train/line.png', pool/'images/old.png')
    Image.new('L', (140, 20), 10).save(pool/'images/replacement.png')
    row = {**original, 'id': 'line__context-v1', 'root_line_id': 'line',
        'image': 'images/replacement.png', 'sha256': digest(pool/'images/replacement.png'),
        'source_split': 'train', 'dataset': preparer.DATASET, 'revision': preparer.REVISION,
        'final_test': False, 'eligible_for_training': True, 'eligible_for_evaluation': False, 'gold': False}
    write_rows(pool/'manifest.jsonl', [row])
    write_rows(pool/'review-history.jsonl', [{**row, 'id': 'line', 'image': 'images/old.png',
        'sha256': digest(pool/'images/old.png'), 'eligible_for_training': False}])
    write_json(pool/'pool-report.json', {'schema': 'slayer-recognizer-reviewed-pool-v1',
        'manifest_sha256': digest(pool/'manifest.jsonl'), 'training_candidates': 1})
    seal(pool)
    return pool


def test_expansion_excludes_original_root_of_replacement(corpus, tmp_path, monkeypatch):
    prepare, root, *paths = corpus
    pool = prior_pool(tmp_path, root)
    Image.new('L', (100, 20), 130).save(root/'train/new.png')
    (root/'train/new.txt').write_text('\u017f\u00e1\u0247\n', encoding='utf-8')
    rows = read_rows(root/'manifest.jsonl')
    rows.append({**rows[0], 'id': 'new', 'image_sha256': digest(root/'train/new.png'),
        'text_sha256': digest(root/'train/new.txt')})
    write_rows(root/'manifest.jsonl', rows)
    monkeypatch.setattr(prepare, 'SOURCE_MANIFEST', digest(root/'manifest.jsonl'))
    result = prepare.prepare(root, *paths, tmp_path/'expansion', count=1, reviewed_pool=pool)
    assert result['previously_reviewed_roots_excluded'] == 1
    assert result['remaining_source_train_lines'] == 1
    assert result['previous_reviewed_root_overlap'] == []
    selected = read_rows(tmp_path/'expansion/manifest.jsonl')
    assert [r['id'] for r in selected] == ['new']
    assert selected[0]['eligible_for_training'] is selected[0]['eligible_for_evaluation'] is False
    assert selected[0]['text'] == '\u017f\u00e1\u0247'
    assert read_rows(tmp_path/'expansion/excluded.jsonl')[0]['reason'] == 'previously-reviewed-root'
    again = prepare.prepare(root, *paths, tmp_path/'reproduced', count=1, reviewed_pool=pool)
    assert result['manifest_sha256'] == again['manifest_sha256']
    assert digest(tmp_path/'expansion/report.json') == digest(tmp_path/'reproduced/report.json')


@pytest.mark.parametrize('kind', ['unknown-root', 'source-hash', 'revision', 'evaluation', 'gold', 'unapproved', 'manifest', 'bytes'])
def test_unbound_prior_pool_fails_closed(corpus, tmp_path, kind):
    _, root, *_ = corpus
    pool = prior_pool(tmp_path, root)
    rows = read_rows(pool/'manifest.jsonl')
    if kind == 'bytes':
        (pool/'images/replacement.png').write_bytes(b'changed')
    elif kind == 'manifest':
        report = json.loads((pool/'pool-report.json').read_text(encoding='utf-8'))
        report['manifest_sha256'] = 'bad'
        write_json(pool/'pool-report.json', report)
        seal(pool)
    else:
        key, value = {'unknown-root': ('root_line_id', 'unknown'), 'source-hash': ('source_image_sha256', 'bad'),
            'revision': ('revision', 'bad'), 'evaluation': ('eligible_for_evaluation', True),
            'gold': ('gold', True), 'unapproved': ('eligible_for_training', False)}[kind]
        rows[0][key] = value
        write_rows(pool/'manifest.jsonl', rows)
        report = json.loads((pool/'pool-report.json').read_text(encoding='utf-8'))
        report['manifest_sha256'] = digest(pool/'manifest.jsonl')
        write_json(pool/'pool-report.json', report)
        seal(pool)
    with pytest.raises(ValueError):
        preparer.reviewed_exclusions(pool, read_rows(root/'manifest.jsonl'))


@pytest.mark.parametrize('kind', ['id', 'hash'])
def test_worker_staging_rechecks_review_exclusion(bundle, tmp_path, kind):
    source, archive, config = bundle
    row = read_rows(source/'manifest.jsonl')[0]
    config['selection'] = {'reviewed_root_ids': [row['id']] if kind == 'id' else [],
        'reviewed_crop_hashes': [row['sha256']] if kind == 'hash' else []}
    with pytest.raises(ValueError, match='previously reviewed'):
        runner.stage(config, archive, tmp_path/'staged')
    assert not (tmp_path/'staged').exists()


@pytest.mark.parametrize('name', ['../outside.zip', 'dir/result.zip', 'dir\\result.zip', 'result.py'])
def test_evidence_archive_names_are_safe(tmp_path, name):
    with pytest.raises(ValueError):
        runner.package(tmp_path, name)


def test_expansion_notebook_compiles_and_uses_distinct_archive(bundle, tmp_path):
    from training.build_recognizer_data_pilot_colab import build
    _, _, config = bundle
    config['selection'] = {'reviewed_root_ids': ['old']}
    config['evidence_archive_name'] = 'recognizer-data-v3-expansion-v1-evidence.zip'
    cfg, notebook = tmp_path/'config.json', tmp_path/'expansion.ipynb'
    write_json(cfg, config)
    build(notebook, cfg, 'a'*40)
    document = json.loads(notebook.read_text(encoding='utf-8'))
    code = '\n'.join(''.join(c['source']) for c in document['cells'] if c['cell_type'] == 'code')
    for c in document['cells']:
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), c['id'], 'exec')
    assert 'package(WORK, EVIDENCE_ARCHIVE_NAME)' in code
    assert 'files.upload' not in code
    assert config['evidence_archive_name'] in code
    write_json(tmp_path/'partial-error.json', {'status': 'not-started'})
    with zipfile.ZipFile(runner.package(tmp_path, config['evidence_archive_name'])) as archive:
        assert 'partial-error.json' in archive.namelist()
