import pytest

from training.composite_score import composite


def report(prefix, unit, total, metric, value, results=()):
    return {'protocol_version': prefix + 'x', unit: total, 'errors_or_missing': 0,
            metric: value, 'results': list(results)}


REPORT_A = report('polocrbench-transcription-', 'pages', 10, 'cer_micro', 0.25)
REPORT_B = report('polocrbench-table-teds-', 'tables', 4, 'teds_mean', 0.75)
REPORT_C = report('polocrbench-kie-', 'documents', 5, 'f1_micro', 0.5)


def test_composite_is_the_mean_of_submitted_subtasks():
    result = composite({'A': REPORT_A, 'B': REPORT_B, 'C': REPORT_C})
    assert result['scores'] == {'A': 0.75, 'B': 0.75, 'C': 0.5}
    assert result['composite'] == pytest.approx((0.75 + 0.75 + 0.5) / 3)
    assert result['components']['A']['metric'] == 'cer_micro'
    assert result['components']['B']['records'] == 4


def test_partial_entry_scores_only_submitted_subtasks():
    result = composite({'B': REPORT_B, 'C': REPORT_C})
    assert result['subtasks'] == ['B', 'C']
    assert result['composite'] == pytest.approx(0.625)


def test_cer_above_one_keeps_the_negative_score():
    result = composite({'A': report('polocrbench-transcription-', 'pages', 2,
                                    'cer_micro', 1.4)})
    assert result['scores']['A'] == pytest.approx(-0.4)
    assert result['composite'] == pytest.approx(-0.4)


def test_wrong_or_missing_reports_raise():
    with pytest.raises(ValueError, match='at least one'):
        composite({})
    with pytest.raises(ValueError, match='subtasks'):
        composite({'D': REPORT_A})
    with pytest.raises(ValueError, match='protocol version'):
        composite({'A': REPORT_B})


def test_timing_report_covers_cost_per_page():
    timed = report('polocrbench-transcription-', 'pages', 2, 'cer_micro', 0.0,
                   results=[{'elapsed_seconds': 1.0}, {'elapsed_seconds': 3.0},
                            {'elapsed_seconds': True}, {}])
    result = composite({'A': timed})
    assert result['timing']['A'] == {'records': 2, 'timed_records': 2,
                                     'mean_elapsed_seconds': 2.0,
                                     'max_elapsed_seconds': 3.0}
    assert 'timing' not in composite({'A': REPORT_A})
