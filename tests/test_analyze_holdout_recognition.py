from training.analyze_holdout_recognition import base_character, classify, edit_alignment


def test_alignment_counts_primary_errors():
    operations = edit_alignment('abc', 'adc!')
    assert sum(op != 'M' for op, _, _ in operations) == 2


def test_historical_errors_are_classified_but_not_removed():
    assert classify('S', 'á', 'a') == 'diacritic'
    assert classify('S', 'ſ', 's') == 'long_s_or_s'
    assert classify('S', '\ue123', 'x') == 'private_use_reference'
    assert base_character('á') == 'a'


def test_case_and_punctuation_categories():
    assert classify('S', 'A', 'a') == 'case_only'
    assert classify('I', None, ',') == 'punctuation'
