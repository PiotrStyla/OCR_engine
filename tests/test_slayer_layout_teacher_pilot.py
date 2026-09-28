import json
import inspect

import pytest

from training.slayer_layout_teacher_pilot import (
    PROMPT,
    canonicalize,
    make_proposal,
    normalize_label,
    parse_qwen_grounding,
    select_pages,
    sha256_bytes,
    _load_qwen,
    run,
)


def rows():
    return [
        {'id': f'p{index:02d}', 'split': 'train', 'collection': f'c{index % 4}',
         'license': 'CC-BY-3.0'}
        for index in range(12)
    ]


def test_selection_is_deterministic_balanced_and_train_only():
    selected = select_pages(rows(), 8, 'salt')
    assert [row['id'] for row in selected] == [
        row['id'] for row in select_pages(list(reversed(rows())), 8, 'salt')]
    counts = {collection: sum(row['collection'] == collection for row in selected)
              for collection in {row['collection'] for row in selected}}
    assert set(counts.values()) == {2}
    bad = rows(); bad[0]['split'] = 'test'
    with pytest.raises(ValueError, match='train pages'):
        select_pages(bad, 8, 'salt')


@pytest.mark.parametrize('raw,expected', [
    ('text_region', 'text_region'), ('heading', 'heading'),
    ('table', 'table'), ('figure', 'figure'), ('caption', 'caption'),
    ('marginalia', 'marginalia'), ('header', 'header'), ('footer', 'footer'),
    ('page_number', 'page_number'),
    ('plain text', 'text_region'), ('Section-header', 'heading'),
    ('FIGURE_CAPTION', 'caption'), ('page footer', 'footer'),
    ('unknown thing', None),
])
def test_label_mapping_is_explicit(raw, expected):
    assert normalize_label(raw) == expected


def test_qwen_json_is_scaled_to_original_pixels_without_transcription():
    text = '```json\n[{"label":"text_region","bbox_2d":[100,200,900,800]}]\n```'
    assert parse_qwen_grounding(text, 2000, 3000) == [{
        'raw_label': 'text_region',
        'bbox_xyxy': [200.0, 600.0, 1800.0, 2400.0],
        'score': 0.5,
        'score_kind': 'neutral-unavailable',
    }]
    with pytest.raises(ValueError, match='normalized range'):
        parse_qwen_grounding('[{"label":"text","bbox_2d":[0,0,1001,20]}]', 100, 100)
    assert 'Do not transcribe' in PROMPT


def test_unmapped_and_invalid_boxes_are_quarantined():
    accepted, rejected = canonicalize([
        {'raw_label': 'Title', 'bbox_xyxy': [1, 2, 50, 30], 'score': 0.9,
         'score_kind': 'model-confidence'},
        {'raw_label': 'abandon', 'bbox_xyxy': [1, 2, 50, 30], 'score': 0.8},
        {'raw_label': 'text', 'bbox_xyxy': [-1, 2, 50, 30], 'score': 0.8},
        {'raw_label': 'text', 'bbox_xyxy': [1, 2, 50, 30], 'score': 0.8,
         'score_kind': 'invented'},
    ], 100, 100)
    assert accepted[0]['label'] == 'heading'
    assert [row['reason'] for row in rejected] == [
        'unmapped-label', 'invalid-bbox', 'invalid-score-kind']


def test_proposal_has_provenance_and_no_reference_text():
    page = {'id': 'p1', 'file_name': 'images/p1.jpg', 'image_sha256': 'a' * 64,
            'width': 100, 'height': 200, 'text': 'must not leak'}
    teacher = {'revision': 'rev'}
    proposal = make_proposal(
        page, 'qwen3-vl-4b', teacher, 'run', sha256_bytes(PROMPT.encode()), [{
            'label': 'text_region', 'bbox_xyxy': [1.0, 2.0, 50.0, 100.0],
            'score': 0.5, 'score_kind': 'neutral-unavailable', 'raw_label': 'text_region'}])
    serialized = json.dumps(proposal)
    assert proposal['teacher']['id'] == 'qwen3-vl-4b'
    assert proposal['detections'][0]['score_kind'] == 'neutral-unavailable'
    assert 'must not leak' not in serialized and 'text' not in proposal


def test_qwen_uses_transformers_457_image_text_auto_loader():
    source = inspect.getsource(_load_qwen)
    assert 'AutoModelForImageTextToText' in source
    assert 'AutoModelForMultimodalLM' not in source


def test_teacher_run_records_and_validates_code_revision():
    source = inspect.getsource(run)
    assert "'code_revision': code_revision" in source
    assert "Invalid code revision" in source
