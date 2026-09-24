"""Recompute aggregate development metrics for publication, without raw OCR."""
import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

from training.audit_geometry_run import edit_count
from training.geometry_comparison import paired_metrics
from training.kaggle_body_dev_diagnostic import digest, metrics


def read_run(path):
    with zipfile.ZipFile(path) as archive:
        checks = json.loads(archive.read('checksums.json'))
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate ZIP entries')
        if set(archive.namelist()) != set(checks) | {'checksums.json'}:
            raise ValueError('Unlisted payload')
        if any(digest(archive.read(k)) != v for k, v in checks.items()):
            raise ValueError('Checksum mismatch')
        report = json.loads(archive.read('report.json'))
        rows = json.loads(archive.read('input-manifest.json'))
        predictions = {}
        for model in report['models']:
            p = json.loads(archive.read(model.replace('/', '--') + '.json'))
            if any(r['status'] != 'ok' or not r['text'].strip() for r in p):
                raise ValueError('Incomplete inference')
            if paired_metrics(rows, p, metrics) != report['results'][model]:
                raise ValueError('Metrics mismatch')
            predictions[model] = p
    return rows, report, predictions


def summarize(previous, current):
    old_rows, old_report, old_predictions = read_run(previous)
    rows, report, predictions = read_run(current)
    if len(rows) != 126 or len(old_rows) != 126:
        raise ValueError('Expected paired 63-line runs')
    if report['models'] != old_report['models']:
        raise ValueError('Model revisions changed')
    for a, b in zip(old_rows, rows):
        if any(a[k] != b[k] for k in ['id', 'text', 'source_review_decision']):
            raise ValueError('References or IDs changed')
    model = 'PiotrSty/trocr-pl-mixed-v3'
    before, after = old_predictions[model], predictions[model]
    counts = Counter()
    net = 0
    for i in range(63, 126):
        delta = edit_count(rows[i]['text'], after[i]['text']) - edit_count(rows[i]['text'], before[i]['text'])
        counts['improved' if delta < 0 else 'regressed' if delta > 0 else 'tied'] += 1
        net += delta
    return {'scope': 'post-hoc development diagnostic; draft references; not benchmark or SOTA',
            'source_zip_sha256': {Path(p).name: digest(Path(p).read_bytes()) for p in [previous, current]},
            'models': report['models'], 'environment': report['environment'],
            'normalization': report['normalization'], 'generation': report['generation'],
            'input_zip_sha256': report['input_zip_sha256'],
            'runner_sha256': report['runner_sha256'],
            'previous_results': old_report['results'], 'current_results': report['results'],
            'mixed_v3_original_records_identical': before[:63] == after[:63],
            'mixed_v3_band_vs_rectangle': dict(counts), 'net_character_edit_change': net,
            'reference_strings_changed': False, 'weights_changed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True)
    parser.add_argument('--current', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(args.previous, args.current)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Verified metrics; aggregate-only summary written.')
