import io
import json
import zipfile

import pytest
from PIL import Image

from training import audit_geometry_run as module
from training.geometry_comparison import pair_inputs, paired_metrics
from training.kaggle_body_dev_diagnostic import digest, load_input, metrics


def test_historical_spelling_counts_as_error():
    assert module.edit_count('bl\u00e1waty \u017f', 'blawaty s') == 2
    assert module.edit_count('a\u0301  b', '\u00e1 b') == 0


def fixture_archives(tmp_path, monkeypatch):
    image = io.BytesIO()
    Image.new('RGB', (40, 20), 'white').save(image, format='PNG')
    content = image.getvalue()
    row = {'id': 'dev__line000', 'text': 'abc', 'original_text': 'abc',
           'collection': 'development', 'page_id': 'page1', 'image': 'line.png',
           'sha256': digest(content), 'source_review_decision': 'proposed',
           'eligible_for_evaluation': False, 'geometry_status': 'auto-proposal',
           'bbox_in_region': [0, 0, 40, 20]}
    source = tmp_path / 'source.zip'
    with zipfile.ZipFile(source, 'w') as z:
        z.writestr('manifest.json', json.dumps([row]))
        z.writestr('provenance.json', json.dumps({'scope': 'diagnostic-only'}))
        z.writestr('line.png', content)
    data = source.read_bytes()
    sha = digest(data)
    monkeypatch.setattr(module, 'OLD_HASH', sha)
    monkeypatch.setattr(module, 'NEW_HASH', sha)
    paired = pair_inputs(data, sha, data, sha, load_input)
    rows, _, provenance = load_input(paired, digest(paired))
    preds = [{'id': r['id'], 'text': 'abc' if i == 0 else 'abd', 'status': 'ok'}
             for i, r in enumerate(rows)]
    model = 'PiotrSty/trocr-pl-mixed-v3'
    report = {'models': {model: 'test'}, 'input_zip_sha256': digest(paired),
              'environment': {}, 'results': {model: paired_metrics(rows, preds, metrics)}}
    payloads = {'input-manifest.json': rows, 'provenance.json': provenance,
                'report.json': report, 'PiotrSty--trocr-pl-mixed-v3.json': preds}
    payloads = {name: json.dumps(value).encode() for name, value in payloads.items()}
    run = tmp_path / 'run.zip'
    with zipfile.ZipFile(run, 'w') as z:
        for name, value in payloads.items():
            z.writestr(name, value)
        z.writestr('checksums.json', json.dumps({name: digest(value) for name, value in payloads.items()}))
    return run, source


def test_audit_round_trip(tmp_path, monkeypatch):
    run, source = fixture_archives(tmp_path, monkeypatch)
    output = tmp_path / 'audit'
    result = module.audit(run, source, source, output)
    assert result['regressed'] == 1
    assert result['net_edit_change'] == 1
    assert (output / 'regression-01.png').exists()
    assert (output / 'index.html').exists()
    with pytest.raises(FileExistsError):
        module.audit(run, source, source, output)


def test_rejects_corrupted_evidence(tmp_path, monkeypatch):
    run, source = fixture_archives(tmp_path, monkeypatch)
    with zipfile.ZipFile(run) as z:
        payloads = {name: z.read(name) for name in z.namelist()}
    payloads['report.json'] = b'{}'
    with zipfile.ZipFile(run, 'w') as z:
        for name, value in payloads.items():
            z.writestr(name, value)
    with pytest.raises(ValueError, match='Evidence checksum mismatch'):
        module.audit(run, source, source, tmp_path / 'audit')
