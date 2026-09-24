"""Compare image-only region geometry while preserving the 63 baseline IDs."""
import argparse
import io
import json
from pathlib import Path
import zipfile

from training.body_line_geometry import detect_lines, crop_line_band
from training.kaggle_body_dev_diagnostic import digest, load_input

INPUT_HASH = '913ded2bd5d742099567a2652460baaf3c4a8bb16625492d0931607396dbdd55'


def prepare(input_zip, region_manifest, output, *, follow_lines=False):
    from PIL import Image
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    original, old_images, _ = load_input(Path(input_zip).read_bytes(), INPUT_HASH)
    region_manifest = Path(region_manifest)
    sources = [json.loads(s) for s in region_manifest.read_text(encoding='utf-8').splitlines()]
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    outcomes, replacements = [], {}
    for row in sources:
        path = (region_manifest.parent / row['image']).resolve()
        if not path.is_relative_to(region_manifest.parent.resolve()) or digest(path.read_bytes()) != row['sha256']:
            raise ValueError('Source region checksum/path mismatch')
        with Image.open(path) as image:
            result = detect_lines(image, follow_lines=follow_lines)
            boxes = result['boxes']
            records = [r for r in original if r['id'].rsplit('__line', 1)[0] == row['id']]
            ordered = sorted(records, key=lambda r: r['id'])
            # Text is not used by the detector; counts only gate tentative ID alignment.
            overlap = any(max(0, a[3] - b[1]) > .3 * min(a[3] - a[1], b[3] - b[1])
                          for a, b in zip(boxes, boxes[1:]))
            usable = bool(ordered) and len(boxes) == len(ordered)
            status = 'candidate-count-matched' if usable else 'fallback-original' if ordered else 'outside-63-line-comparison'
            outcomes.append({'region_id': row['id'], 'source_sha256': row['sha256'], **result,
                             'baseline_lines': len(ordered), 'reference_lines': len(row['text'].splitlines()),
                             'overlapping_boxes': overlap, 'status': status})
            if usable:
                for index, (record, box, foreign) in enumerate(zip(ordered, boxes, result['foreign_ink_fraction'])):
                    if foreign > .10:
                        continue
                    data = io.BytesIO()
                    crop = crop_line_band(image, box, result['line_bands'][index]) if follow_lines else image.crop(box)
                    crop.save(data, format='PNG')
                    replacements[record['id']] = (data.getvalue(), box)
        print(row['id'], status, len(boxes), flush=True)
    rows = []
    for i, (row, old) in enumerate(zip(original, old_images)):
        content, box = replacements.get(row['id'], (old, None))
        relative = f'images/{i:04d}.png'
        (output / relative).write_bytes(content)
        rows.append({**row, 'image': relative, 'sha256': digest(content),
                     'baseline_sha256': row['sha256'], 'bbox_in_region': box,
                     'geometry_status': 'auto-proposal' if box else 'fallback-original'})
    report = {'scope': 'diagnostic-only', 'method': 'image-only connected components; no reference text/count in detection',
              'follow_lines': follow_lines,
              'pixel_operation': 'RGB conversion; outside-band pixels whitened' if follow_lines else 'unchanged source crop',
              'baseline_input_sha256': INPUT_HASH, 'lines': len(rows), 'regions': len(sources),
              'auto_lines': len(replacements), 'fallback_lines': len(rows) - len(replacements),
              'detector_sha256': digest(Path(__file__).with_name('body_line_geometry.py').read_bytes()),
              'outcomes': outcomes, 'reference_changed': False,
              'limitations': 'Post-hoc development experiment. Count matching is not verified alignment. Retain fallback lines in all metrics. Eight previously unmatched regions remain outside the 63-line score.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    archive = output / 'body-auto-geometry-input.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(rows, ensure_ascii=False))
        z.writestr('provenance.json', json.dumps(report))
        for row in rows:
            z.write(output / row['image'], row['image'])
    load_input(archive.read_bytes(), digest(archive.read_bytes()))
    print('ZIP SHA256:', digest(archive.read_bytes()))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-zip', required=True)
    parser.add_argument('--region-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--follow-lines', action='store_true')
    args = parser.parse_args()
    prepare(args.input_zip, args.region_manifest, args.output, follow_lines=args.follow_lines)
