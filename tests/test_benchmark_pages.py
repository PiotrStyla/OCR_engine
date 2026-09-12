import hashlib
import json
import pytest
from training.benchmark_pages import evaluate


def test_missing_page_counts_as_deletion(tmp_path):
    image=tmp_path/'page.png'
    image.write_bytes(b'fixture')
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps({'id':'page','image':'page.png',
                        'sha256':hashlib.sha256(b'fixture').hexdigest(),'text':'Ala ma kota'}))
    prediction=tmp_path/'pred.jsonl'
    prediction.write_text('')
    result=evaluate(manifest,prediction)
    assert result['errors_or_missing']==1
    assert result['cer_micro']==1
    prediction.write_text(json.dumps({'id':'page','status':'ok','text':'Ala ma kota'}))
    assert evaluate(manifest,prediction)['cer_micro']==0
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        evaluate(manifest,prediction)
