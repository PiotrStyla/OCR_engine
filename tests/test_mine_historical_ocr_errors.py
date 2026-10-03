import hashlib
import json
import zipfile

import pytest
from PIL import Image

from training.full_page_pilot import digest, write_json, write_rows
from training.mine_historical_ocr_errors import mismatches, prepare


def fixture(tmp_path):
    audit = tmp_path/'audit'
    dataset = audit/'verified-dataset'
    evidence = audit/'evidence'
    dataset.mkdir(parents=True)
    (evidence/'dataset').mkdir(parents=True)
    (evidence/'predictions/qwen3-vl-4b').mkdir(parents=True)
    Image.new('L', (12, 16), 255).save(dataset/'page.png')
    text = 'Po\u017f\u0142a\u0142 w b\u0142\u00e1waty \u0247'
    row = {'id': 'PAGE_1', 'text': text, 'image': 'page.png',
           'sha256': digest(dataset/'page.png'), 'collection': 'collection',
           'split': 'validation', 'eligible_for_training': False,
           'final_test': False, 'reference_status': 'single-review-draft-not-gold'}
    write_rows(dataset/'manifest.jsonl', [row])
    prediction = {'id': row['id'], 'status': 'ok', 'text': 'Pofal w blawaty e'}
    candidate = evidence/'predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl'
    baseline = evidence/'dataset/qwen-v5-1mp.jsonl'
    write_rows(candidate, [prediction])
    write_rows(baseline, [prediction])
    payload = {'dataset/manifest.jsonl': (dataset/'manifest.jsonl').read_bytes(),
               'predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl': candidate.read_bytes(),
               'dataset/qwen-v5-1mp.jsonl': baseline.read_bytes()}
    with zipfile.ZipFile(audit/'source-evidence.zip', 'w') as stream:
        for name, data in payload.items():
            stream.writestr(name, data)
        stream.writestr('checksums.json', json.dumps({name: hashlib.sha256(data).hexdigest()
                                                    for name, data in payload.items()}))
    write_json(audit/'audit.json', {'schema': 'slayer-full-page-validation-4mp-v7-audit',
        'metrics_recomputed': True, 'gold_pages': 0, 'pages': 1,
        'source_archive_sha256': digest(audit/'source-evidence.zip')})
    return audit


def test_raw_glyphs_not_modernized_and_equal_ignored():
    assert mismatches('\u017f\u00e1\u0247', '\u017f\u00e1\u0247') == []
    items = mismatches('\u017f\u00e1\u0247', 'sae')
    assert ''.join(item['reference'] for item in items) == '\u017f\u00e1\u0247'
    assert all(item['eligible_for_training'] is False for item in items)


def test_deletion_insertion_and_utf16_offsets():
    item = mismatches('\U0001f600 A\u017f B', '\U0001f600 A B')[0]
    assert item['kind'] == 'delete' and item['offset'] == 4 and item['length'] == 1
    inserted = mismatches('A B', 'A\u017f B')[0]
    assert inserted['kind'] == 'insert' and inserted['length'] == 0
    assert mismatches('A\nb', 'A b') == []


def test_preparation_preserves_sources_and_no_training(tmp_path):
    source = fixture(tmp_path)
    manifest = source/'verified-dataset/manifest.jsonl'
    original = manifest.read_bytes()
    output = tmp_path/'mining'
    result = prepare(source, output)
    assert result['pages'] == result['pages_with_mismatches'] == 1
    assert result['candidate_edit_spans'] > 0 and result['word_context_edit_spans'] > 0
    assert result['training_examples_created'] == result['gold_pages'] == 0
    assert manifest.read_bytes() == original == (output/'original-manifest.jsonl').read_bytes()
    row = json.loads((output/'original-manifest.jsonl').read_text(encoding='utf-8'))
    assert digest(output/row['image']) == row['sha256']
    assert (output/'raw-v7-predictions.jsonl').read_bytes() == (
        source/'evidence/predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl').read_bytes()
    assert (output/'review/images/0000.png').read_bytes() == (source/'verified-dataset/page.png').read_bytes()
    html = (output/'review/index.html').read_text(encoding='utf-8')
    assert 'diagnostics' in html and 'Qwen v7 4MP' in html
    checksums = json.loads((output/'checksums.json').read_text())
    assert all(digest(output/name) == expected for name, expected in checksums.items())
    with pytest.raises(FileExistsError):
        prepare(source, output)


@pytest.mark.parametrize('file', ['source-evidence.zip', 'verified-dataset/manifest.jsonl',
    'evidence/predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl',
    'evidence/dataset/qwen-v5-1mp.jsonl', 'verified-dataset/page.png'])
def test_changed_source_rejected_before_output(tmp_path, file):
    source = fixture(tmp_path)
    path = source/file
    path.write_bytes(path.read_bytes()+b' changed')
    with pytest.raises(ValueError):
        prepare(source, tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


def test_duplicate_predictions_rejected(tmp_path):
    source = fixture(tmp_path)
    path = source/'evidence/predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl'
    path.write_bytes(path.read_bytes()*2)
    sync_archive(source)
    with pytest.raises(ValueError, match='coverage'):
        prepare(source, tmp_path/'rejected')


def sync_archive(source):
    files = {'dataset/manifest.jsonl': source/'verified-dataset/manifest.jsonl',
        'dataset/qwen-v5-1mp.jsonl': source/'evidence/dataset/qwen-v5-1mp.jsonl',
        'predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl':
            source/'evidence/predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl'}
    with zipfile.ZipFile(source/'source-evidence.zip', 'w') as stream:
        for name, path in files.items():
            stream.writestr(name, path.read_bytes())
        stream.writestr('checksums.json', json.dumps({name: digest(path) for name, path in files.items()}))
    report = json.loads((source/'audit.json').read_text())
    report['source_archive_sha256'] = digest(source/'source-evidence.zip')
    write_json(source/'audit.json', report)


@pytest.mark.parametrize(('key', 'value'), [
    ('split', 'train'), ('eligible_for_training', True), ('final_test', True)])
def test_split_firewall_rejected_before_output(tmp_path, key, value):
    source = fixture(tmp_path)
    path = source/'verified-dataset/manifest.jsonl'
    row = json.loads(path.read_text(encoding='utf-8'))
    row[key] = value
    write_rows(path, [row])
    sync_archive(source)
    with pytest.raises(ValueError, match='non-training validation'):
        prepare(source, tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()
