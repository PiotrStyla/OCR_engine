import json

import pytest

from training.build_layout_consensus import build_consensus, digest


CATEGORIES = ['text_region', 'heading', 'table']


def proposal(page, teacher, detections, *, image_hash='a' * 64,
             score_kind='model-confidence'):
    return {
        'schema': 'slayer-layout-teacher-proposal-v1',
        'page_id': page,
        'image': {'file_name': f'images/{page}.png', 'sha256': image_hash,
                  'width': 100, 'height': 200},
        'teacher': {'id': teacher, 'revision': 'rev-1', 'run_id': f'run-{teacher}',
                    'prompt_sha256': 'b' * 64},
        'detections': [
            {'id': f'{teacher}-{index}', 'label': label, 'bbox_xyxy': box, 'score': score,
             'score_kind': score_kind}
            for index, (label, box, score) in enumerate(detections)
        ],
    }


def write(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    return path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_two_teachers_create_median_consensus_and_coco(tmp_path):
    source = write(tmp_path / 'teachers.jsonl', [
        proposal('p1', 'a', [('text_region', [10, 20, 50, 80], 0.9)]),
        proposal('p1', 'b', [('text_region', [12, 18, 52, 82], 0.8)]),
    ])
    output = tmp_path / 'out'
    report = build_consensus([source], output, categories=CATEGORIES)
    assert report['accepted_objects'] == 1
    assert report['review_objects'] == 0
    item = read_jsonl(output / 'consensus.jsonl')[0]['objects'][0]
    assert item['bbox_xyxy'] == [11.0, 19.0, 51.0, 81.0]
    assert item['teachers'] == ['a', 'b']
    assert item['score_kinds'] == ['model-confidence']
    coco = json.loads((output / 'annotations.coco.json').read_text(encoding='utf-8'))
    assert coco['annotations'][0]['bbox'] == [11.0, 19.0, 40.0, 62.0]
    assert [item['name'] for item in coco['categories']] == CATEGORIES
    for line in (output / 'checksums.sha256').read_text().splitlines():
        checksum, name = line.split('  ')
        assert digest(output / name) == checksum


def test_singleton_and_low_score_go_to_review_and_hard_examples(tmp_path):
    source = write(tmp_path / 'teachers.jsonl', [proposal('p1', 'a', [
        ('heading', [1, 2, 40, 20], 0.9),
        ('table', [5, 40, 90, 160], 0.2),
    ])])
    output = tmp_path / 'out'
    report = build_consensus([source], output, categories=CATEGORIES, min_score=0.5)
    assert report['accepted_objects'] == 0
    reviews = read_jsonl(output / 'review-queue.jsonl')
    assert {reason for row in reviews for reason in row['reasons']} == {
        'below-quorum', 'below-score-threshold'}
    hard = read_jsonl(output / 'hard-examples.jsonl')
    assert hard[0]['page_id'] == 'p1'


def test_conflicting_labels_demote_otherwise_valid_consensus(tmp_path):
    rows = [
        proposal('p1', 'a', [('text_region', [10, 10, 80, 80], 0.9)]),
        proposal('p1', 'b', [('text_region', [11, 11, 81, 81], 0.9)]),
        proposal('p1', 'c', [('table', [10, 10, 80, 80], 0.9)]),
        proposal('p1', 'd', [('table', [11, 11, 81, 81], 0.9)]),
    ]
    output = tmp_path / 'out'
    report = build_consensus([write(tmp_path / 'teachers.jsonl', rows)], output,
                             categories=CATEGORIES)
    assert report['accepted_objects'] == 0
    assert report['review_objects'] == 2
    assert all('label-conflict' in row['reasons']
               for row in read_jsonl(output / 'review-queue.jsonl'))


@pytest.mark.parametrize('mutate,match', [
    (lambda rows: rows.append(rows[0]), 'Duplicate teacher vote'),
    (lambda rows: rows[1]['image'].__setitem__('sha256', 'c' * 64), 'Image identity mismatch'),
    (lambda rows: rows[0]['detections'][0].__setitem__('label', 'unknown'), 'Unknown category'),
    (lambda rows: rows[1]['teacher'].__setitem__('id', 'A'), 'lowercase model-family slug'),
    (lambda rows: rows[1]['detections'][0].__setitem__('score_kind', 'invented'),
     'score_kind'),
])
def test_invalid_provenance_is_rejected_before_writes(tmp_path, mutate, match):
    rows = [
        proposal('p1', 'a', [('text_region', [10, 10, 80, 80], 0.9)]),
        proposal('p1', 'b', [('text_region', [11, 11, 81, 81], 0.9)]),
    ]
    mutate(rows)
    source = write(tmp_path / 'teachers.jsonl', rows)
    output = tmp_path / 'out'
    with pytest.raises(ValueError, match=match):
        build_consensus([source], output, categories=CATEGORIES)
    assert not output.exists()


def test_teacher_revision_cannot_change_between_pages(tmp_path):
    rows = [
        proposal('p1', 'a', [('text_region', [10, 10, 80, 80], 0.9)]),
        proposal('p2', 'a', [('text_region', [10, 10, 80, 80], 0.9)]),
    ]
    rows[1]['teacher']['revision'] = 'rev-2'
    source = write(tmp_path / 'teachers.jsonl', rows)
    with pytest.raises(ValueError, match='Inconsistent teacher run'):
        build_consensus([source], tmp_path / 'out', categories=CATEGORIES)


def test_output_is_never_overwritten(tmp_path):
    source = write(tmp_path / 'teachers.jsonl', [
        proposal('p1', 'a', [('text_region', [10, 10, 80, 80], 0.9)]),
        proposal('p1', 'b', [('text_region', [11, 11, 81, 81], 0.9)]),
    ])
    output = tmp_path / 'out'
    build_consensus([source], output, categories=CATEGORIES)
    with pytest.raises(FileExistsError):
        build_consensus([source], output, categories=CATEGORIES)
