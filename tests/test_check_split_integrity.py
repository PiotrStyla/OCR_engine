import hashlib
import io
import json

import pytest

from training.check_split_integrity import (
    check,
    dhash,
    hamming,
    holdout_value,
    load_split,
    shingles,
    text_similarity,
)

from PIL import Image, ImageDraw


def write_manifest(tmp_path, name, rows):
    manifest = tmp_path / name
    manifest.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows),
                        encoding='utf-8', newline='\n')
    return manifest


def page(tmp_path, name, offset=0, invert=False):
    image = Image.new('RGB', (200, 200), 'black' if invert else 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle([20 + offset, 20, 180, 60], fill='white' if invert else 'black')
    draw.rectangle([20, 100 + offset, 180, 140], fill='white' if invert else 'black')
    path = tmp_path / name
    image.save(path)
    return path


def row_for(path, id_, text, tmp_path):
    return {'id': id_, 'image': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'text': text}


def test_dhash_survives_reencoding_and_detects_drift(tmp_path):
    path = page(tmp_path, 'clean.png')
    reencoded = tmp_path / 'reencoded.jpg'
    Image.open(path).convert('RGB').save(reencoded, quality=40)
    other = page(tmp_path, 'other.png', invert=True)
    clean, again, moved = (dhash(item.read_bytes()) for item in (path, reencoded, other))
    assert hamming(clean, again) <= 4  # copy with different bytes
    assert hamming(clean, moved) > 6   # different page


def test_text_similarity_catches_reformatting_and_subpages():
    original = shingles('Faktura VAT numer FV dwadzieścia cztery z dnia trzeciego marca')
    reformatted = shingles('Faktura   VAT numer FV dwadzieścia\ncztery z dnia trzeciego marca')
    jaccard, containment = text_similarity(original, reformatted)
    assert jaccard == 1.0 and containment == 1.0
    subpage = shingles('Faktura VAT numer FV')
    jaccard, containment = text_similarity(original, subpage)
    assert containment == 1.0 and jaccard < 0.5
    assert text_similarity(original, shingles('Zupełnie inna treść dokumentu w całości')) == (0.0, 0.0)


def test_clean_splits_have_no_violations(tmp_path):
    first = page(tmp_path, 'a.png')
    second = page(tmp_path, 'b.png', invert=True)
    splits = [
        load_split('train', write_manifest(tmp_path, 'train.jsonl',
                                           [row_for(first, 't1', 'Pierwszy dokument w całości', tmp_path)])),
        load_split('testA', write_manifest(tmp_path, 'testA.jsonl',
                                           [row_for(second, 'a1', 'Drugi dokument o innej treści', tmp_path)])),
    ]
    report = check(splits)
    assert report['violations'] == 0
    assert report['splits']['train']['images_missing'] == 0


def test_cross_split_copies_are_flagged(tmp_path):
    shared = page(tmp_path, 'shared.png')
    copy_bytes = tmp_path / 'copy.jpg'
    Image.open(shared).convert('RGB').save(copy_bytes, quality=90)  # same pixels, new bytes
    text = 'Umowa w sprawie wykonania projektu technicznego'
    splits = [
        load_split('train', write_manifest(tmp_path, 'train.jsonl',
                                           [row_for(copy_bytes, 't1', text, tmp_path)])),
        load_split('testA', write_manifest(tmp_path, 'testA.jsonl',
                                           [row_for(shared, 'a1', text + ' z dopiskiem', tmp_path)])),
    ]
    report = check(splits, near_text_similarity=0.5)
    pair = report['pairs'][0]
    assert pair['near_image'] == [{'a': 't1', 'b': 'a1', 'distance': 0}]
    assert pair['near_text'][0]['containment'] > 0.5
    assert pair['violations'] == 1 and pair['warnings'] == 1  # text leak hard, look-alike soft


def test_exact_text_and_image_duplicates_are_flagged(tmp_path):
    shared = page(tmp_path, 'same.png')
    splits = [
        load_split('train', write_manifest(tmp_path, 'train.jsonl',
                                           [row_for(shared, 't1', 'Identyczny opis', tmp_path)])),
        load_split('testB', write_manifest(tmp_path, 'testB.jsonl',
                                           [row_for(shared, 'b1', 'Identyczny opis', tmp_path)])),
    ]
    report = check(splits)
    assert report['pairs'][0]['exact_image'] == [['t1', 'b1']]
    assert report['pairs'][0]['exact_text'] == [['t1', 'b1']]
    assert report['violations'] == 2


def test_record_ids_shared_between_splits_are_violations(tmp_path):
    first = page(tmp_path, 'id-a.png')
    second = page(tmp_path, 'id-b.png', invert=True)
    splits = [
        load_split('train', write_manifest(tmp_path, 'train.jsonl',
                                           [row_for(first, 'formularz-3', 'Pierwszy', tmp_path)])),
        load_split('testB', write_manifest(tmp_path, 'testB.jsonl',
                                           [row_for(second, 'formularz-3', 'Drugi', tmp_path)])),
    ]
    report = check(splits)
    assert report['pairs'][0]['shared_ids'] == ['formularz-3']
    assert report['violations'] == 1


def test_missing_images_are_counted_but_text_checks_run(tmp_path):
    ghost = {'id': 'g1', 'image': 'brak.png', 'sha256': 'x', 'text': 'Osierocony opis'}
    real = page(tmp_path, 'real.png')
    splits = [
        load_split('train', write_manifest(tmp_path, 'train.jsonl', [ghost])),
        load_split('testA', write_manifest(tmp_path, 'testA.jsonl',
                                           [row_for(real, 'a1', 'Osierocony opis', tmp_path)])),
    ]
    report = check(splits)
    assert report['splits']['train']['images_missing'] == 1
    assert report['pairs'][0]['exact_text'] == [['g1', 'a1']]


def test_holdout_field_enforces_disjoint_test_b(tmp_path):
    image = page(tmp_path, 'x.png')
    other = page(tmp_path, 'y.png', invert=True)
    train = write_manifest(tmp_path, 'train.jsonl', [
        {**row_for(image, 't1', 'tekst jeden', tmp_path), 'degradation': 'clean'},
        {**row_for(image, 't2', 'tekst dwa', tmp_path), 'degradation': 'scan'}])
    clean_b = write_manifest(tmp_path, 'testB.jsonl', [
        {**row_for(other, 'b1', 'tekst trzy', tmp_path), 'degradation': 'photo'}])
    leaky_b = write_manifest(tmp_path, 'testB-leaky.jsonl', [
        {**row_for(other, 'b1', 'tekst cztery', tmp_path), 'degradation': 'scan'}])
    report = check([load_split('train', train), load_split('testB', clean_b)],
                   holdout_field='degradation', holdout_split='testB')
    assert report['holdout'] == {'field': 'degradation', 'split': 'testB',
                                 'overlaps': [], 'rows_without_value': 0}
    assert report['violations'] == 0
    report = check([load_split('train', train), load_split('testB', leaky_b)],
                   holdout_field='degradation', holdout_split='testB')
    assert report['holdout']['overlaps'] == [{'value': 'scan', 'id': 'b1', 'other_ids': ['t2']}]
    assert report['violations'] == 1


def test_holdout_values_come_from_generation_metadata(tmp_path):
    image = page(tmp_path, 'm.png')
    other = page(tmp_path, 'n.png', invert=True)
    train = write_manifest(tmp_path, 'train.jsonl', [row_for(image, 't1', 'jeden', tmp_path)])
    test_b = write_manifest(tmp_path, 'testB.jsonl', [row_for(other, 'b1', 'dwa', tmp_path)])
    metadata = tmp_path / 'generation.json'
    metadata.write_text(json.dumps({'samples': [
        {'id': 't1', 'degradation': 'clean', 'recipe': {'kind': 'clean'}},
        {'id': 'b1', 'recipe': {'kind': 'photo'}}]}), encoding='utf-8')
    splits = [load_split('train', train, metadata), load_split('testB', test_b, metadata)]
    assert holdout_value(splits[1]['samples'][0], 'degradation') == 'photo'  # recipe fallback
    report = check(splits, holdout_field='degradation', holdout_split='testB')
    assert report['violations'] == 0
