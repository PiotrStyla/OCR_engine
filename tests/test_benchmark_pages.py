import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
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


def test_real_lines_v1_manifest_is_complete_and_hashed():
    root = Path(__file__).parents[1] / "benchmarks" / "real-lines-v1"
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 75
    assert len({row["id"] for row in rows}) == 75
    assert {row["source_page"] for row in rows} == {"odezwa-12", "torun-74"}
    assert sum(row["source_page"] == "odezwa-12" for row in rows) == 33
    assert sum(row["source_page"] == "torun-74" for row in rows) == 42
    for row in rows:
        image_path = root / row["image"]
        text_path = root / row["text_file"]
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == row["sha256"]
        assert text_path.read_text(encoding="utf-8") == row["text"]
        with Image.open(image_path) as image:
            assert image.width > 0 and image.height > 0
