from training.probe_correction import distance, score, summarize


def test_failed_correction_is_not_counted_as_success():
    row={'id':'case','input':'abc','reference':'abc','status':'incomplete','output':'a',
         'requested_model':'test','category':'preserve_clean'}
    assert score(row)['accepted'] is False
    assert score(row)['raw_output_edits']==2
    assert score(row)['effective_edits']==0
    report=summarize([row])['test']
    assert report['completed']==0


def test_literal_ascii_and_diacritics_are_distinct():
    assert distance('Zolw','Żółw')==3
    row={'input':'Zolw','reference':'Zolw','status':'ok','output':'Żółw'}
    assert score(row)['changed_clean']
