from training.audit_geometry_holdout import paired_bootstrap, percentile


def test_percentile_is_bounded():
    assert percentile([1, 2, 3, 4], .025) == 1
    assert percentile([1, 2, 3, 4], .975) == 4


def test_paired_bootstrap_direction_and_reproducibility():
    rows = [{'id': 'a', 'text': 'abcd'}, {'id': 'b', 'text': 'abcdef'}]
    rectangle = {'a': 2, 'b': 3}
    bands = {'a': 1, 'b': 2}
    first = paired_bootstrap(rows, rectangle, bands, samples=100, seed=7)
    second = paired_bootstrap(rows, rectangle, bands, samples=100, seed=7)
    assert first == second
    assert first['cer_delta_line_band_minus_rectangle'] < 0
    assert first['bootstrap_fraction_line_band_better'] == 1
