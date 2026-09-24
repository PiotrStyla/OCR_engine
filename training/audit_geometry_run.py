"""Build a checksum-verified, read-only paired geometry audit from a run ZIP."""
import argparse
import html
import io
import json
from pathlib import Path
import unicodedata
import zipfile

from training.geometry_comparison import pair_inputs, paired_metrics
from training.kaggle_body_dev_diagnostic import digest, load_input, metrics
from training.build_geometry_colab import OLD_HASH, NEW_HASH


def edit_count(reference, prediction):
    from jiwer import process_characters
    normalize = lambda s: ' '.join(unicodedata.normalize('NFC', s).split())
    score = process_characters(normalize(reference), normalize(prediction))
    return score.substitutions + score.deletions + score.insertions


def audit(run_path, original_path, automatic_path, output):
    from PIL import Image, ImageDraw
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    run_data = Path(run_path).read_bytes()
    original = Path(original_path).read_bytes()
    automatic = Path(automatic_path).read_bytes()
    payload = pair_inputs(original, OLD_HASH, automatic, NEW_HASH, load_input)
    rows, images, _ = load_input(payload, digest(payload))
    with zipfile.ZipFile(io.BytesIO(run_data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate evidence members')
        checksums = json.loads(archive.read('checksums.json'))
        if set(names) != set(checksums) | {'checksums.json'}:
            raise ValueError('Unexpected evidence members')
        if any(digest(archive.read(name)) != sha for name, sha in checksums.items()):
            raise ValueError('Evidence checksum mismatch')
        report = json.loads(archive.read('report.json'))
        # ZIP compression bytes can differ across zlib versions; verify the
        # pinned source archives and exact reconstructed manifest instead.
        provenance = json.loads(archive.read('provenance.json'))
        if provenance['source_archive_hashes'] != [OLD_HASH, NEW_HASH]:
            raise ValueError('Source archive hash mismatch')
        if rows != json.loads(archive.read('input-manifest.json')):
            raise ValueError('Input manifest mismatch')
        predictions = {}
        for model in report['models']:
            predictions[model] = json.loads(archive.read(model.replace('/', '--') + '.json'))
            if paired_metrics(rows, predictions[model], metrics) != report['results'][model]:
                raise ValueError('Metric mismatch: ' + model)
            if any(p['status'] != 'ok' for p in predictions[model]):
                raise ValueError('Incomplete inference: ' + model)
    model = 'PiotrSty/trocr-pl-mixed-v3'
    preds = predictions[model]
    size = len(rows) // 2
    details = []
    for i in range(size):
        a, b = rows[i], rows[i + size]
        if a['comparison_id'] != b['comparison_id'] or a['text'] != b['text']:
            raise ValueError('Pair mismatch')
        before = edit_count(a['text'], preds[i]['text'])
        after = edit_count(b['text'], preds[i + size]['text'])
        details.append({'id': a['comparison_id'], 'index': i, 'reference': a['text'],
                        'before': preds[i]['text'], 'after': preds[i + size]['text'],
                        'edits_before': before, 'edits_after': after, 'delta': after - before,
                        'geometry_status': a['comparison_geometry_status'],
                        'automatic_bbox': b.get('bbox_in_region'),
                        'review_decision': a['source_review_decision']})
    changed = [r for r in details if r['geometry_status'] == 'auto-proposal']
    regressions = sorted([r for r in changed if r['delta'] > 0], key=lambda r: -r['delta'])
    summary = {'scope': 'Post-hoc development audit; draft references, not a held-out benchmark.',
               'run_sha256': digest(run_data), 'reported_input_sha256': report['input_zip_sha256'],
               'reconstructed_input_sha256': digest(payload),
               'environment': report['environment'], 'metrics': report['results'],
               'improved': sum(r['delta'] < 0 for r in changed),
               'regressed': len(regressions), 'tied': sum(r['delta'] == 0 for r in changed),
               'net_edit_change': sum(r['delta'] for r in details),
               'fallback_identical': all(r['before'] == r['after'] for r in details
                                         if r['geometry_status'] == 'fallback-original'),
               'lines': details}
    output.mkdir(parents=True)
    (output / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    sections = []
    for number, row in enumerate(regressions, 1):
        canvas = Image.new('RGB', (1500, 360), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 4), f"{number}. {row['id']} | edits {row['edits_before']} -> {row['edits_after']}", fill='black')
        for arm, offset in enumerate([0, size]):
            with Image.open(io.BytesIO(images[row['index'] + offset])) as source:
                picture = source.convert('RGB')
                picture.thumbnail((1470, 140))
                y = 30 + arm * 165
                draw.text((10, y), 'ORIGINAL' if arm == 0 else 'AUTOMATIC', fill='black')
                canvas.paste(picture, (10, y + 20))
        name = f'regression-{number:02d}.png'
        canvas.save(output / name)
        esc = html.escape
        sections.append(f'<section><h2>{esc(row["id"])}</h2><img src="{name}" alt="Original and automatic crop">'
                        f'<p>Reference: {esc(row["reference"])}</p><p>Before: {esc(row["before"])}</p>'
                        f'<p>After: {esc(row["after"])}</p><p>Edits: {row["edits_before"]} to {row["edits_after"]}; '
                        f'review: {esc(row["review_decision"])}</p></section>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>Geometry regression audit</title><style>body{font:16px system-ui;margin:24px;color:#222}'
            'section{border-top:1px solid #aaa;padding:16px 0}img{max-width:100%;height:auto}h2,p{overflow-wrap:anywhere}'
            '</style><h1>Geometry regression audit</h1><p>Draft references. Historical spelling retained. '
            'Post-hoc development inspection, not a benchmark.</p>' + ''.join(sections) + '</html>')
    (output / 'index.html').write_text(page, encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'original', 'automatic', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.original, args.automatic, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in {'lines', 'metrics'}}, indent=2))
    print(json.dumps([r for r in result['lines'] if r['delta'] > 0], ensure_ascii=True, indent=2))
