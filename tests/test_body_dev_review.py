import pytest

from training.prepare_body_dev_review import align_candidates, choose


def row(id='r1', page='p1', **changes):
    return dict(dict(id=id, page_id=page, collection='dev', split='validation',
                     image_sha256=id, region_type='paragraph', text='one\ntwo\nthree'), **changes)


def test_page_round_robin_and_quota():
    selected, excluded = choose([row('r1'), row('r2'), row('r3', 'p2')], [], 2)
    assert [r['id'] for r in selected] == ['r1', 'r3']
    assert excluded == [{'id': 'r2', 'reason': 'deterministic collection quota'}]


@pytest.mark.parametrize('changes', [{'collection': 'NA2_FT'}, {'split': 'train'}, {'id': '../x'}])
def test_invalid_source(changes):
    with pytest.raises(ValueError):
        choose([row(**changes)], [])


def test_overlap_and_duplicates():
    with pytest.raises(ValueError):
        choose([row()], [{'image_sha256': 'r1'}])
    with pytest.raises(ValueError):
        choose([row(), row()], [])


def test_title_and_heading_exclusion():
    selected, excluded = choose([row(), row('r2', 'Choragiew_FT__436794'), row('r3', region_type='heading')], [])
    assert len(selected) == 1
    assert len(excluded) == 2


def test_mismatch_does_not_assign_reference():
    assert align_candidates([(0, 0, 10, 5)], 'one\ntwo', (10, 20)) == ([], 'line-count-mismatch')


def test_overlap_requires_review():
    assert align_candidates([(0, 0, 10, 6), (0, 5, 10, 10)], 'one\ntwo', (10, 20))[1] == 'vertically-overlapping-candidates'


def test_matches_are_only_proposals_and_keep_unicode():
    pairs, status = align_candidates([(0, 8, 10, 14), (0, 0, 10, 6)], 'one\ntw\ufffd', (10, 20))
    assert status == 'unverified-count-matched'
    assert pairs[0] == ((0, 0, 10, 7), 'one')
    assert pairs[1][0] == (0, 7, 10, 20)
    assert pairs[1][1] == 'tw\ufffd'


def test_geometry_bounds():
    with pytest.raises(ValueError):
        align_candidates([(0, -1, 10, 6)], 'text', (10, 20))
