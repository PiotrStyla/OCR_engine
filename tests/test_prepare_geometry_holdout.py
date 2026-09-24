import pytest

from training.prepare_geometry_holdout import has_broken_reference, private_use_count, select, source_path


def row(id, collection='c', page='p', text='ábc\ndef\nſgh', **changes):
    return dict({'id': id, 'split': 'train', 'collection': collection, 'page_id': page,
                 'text': text, 'region_type': 'paragraph', 'image_sha256': id,
                 'license': 'CC-BY-3.0', 'file_name': 'images/' + id + '.jpg'}, **changes)


def test_historical_characters_are_not_broken():
    assert not has_broken_reference('á ſ ɇ \ue123')
    assert has_broken_reference('\ufffd')
    assert private_use_count('á \ue123 \ue456') == 2


def test_selection_is_deterministic_and_collection_disjoint():
    rows = [row('a1', 'a'), row('a2', 'a'), row('b1', 'b'), row('c1', 'c')]
    first, _ = select(rows, [], set(), sample_collections=2)
    second, _ = select(list(reversed(rows)), [], set(), sample_collections=2)
    assert [r['id'] for r in first] == [r['id'] for r in second]
    assert len({r['collection'] for r in first}) == 2


def test_used_page_and_bad_reference_are_excluded():
    rows = [row('a1', 'a', page='used'), row('b1', 'b', text='one\ntwo\n\ufffd')]
    with pytest.raises(ValueError, match='Insufficient'):
        select(rows, [], {'used'}, sample_collections=1)


def test_exact_test_overlap_and_frozen_collection_rejected():
    with pytest.raises(ValueError, match='overlap'):
        select([row('a1')], [{'image_sha256': 'a1'}], set(), 1)
    with pytest.raises(ValueError, match='frozen'):
        select([row('a1', collection='NA2_FT')], [], set(), 1)


def test_safe_source_path():
    assert source_path(row('a1')) == 'regions/train/images/a1.jpg'
    with pytest.raises(ValueError):
        source_path(row('a1', file_name='../x.jpg'))
