"""Download frozen holdout regions and build an offline reference-review UI."""
import argparse
import io
import json
from pathlib import Path
from urllib.request import urlopen

from training.build_annotation_review import build
from training.geometry_holdout_runner import DATASET, REVISION, digest


def stage(manifest_path, output, opener=urlopen):
    from PIL import Image
    manifest_path, output = Path(manifest_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = manifest['regions']
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    staged = []
    root = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/'
    for index, row in enumerate(rows):
        with opener(root + row['source_path'], timeout=120) as response:
            data = response.read()
        if digest(data) != row['image_sha256']:
            raise ValueError('Source checksum mismatch: ' + row['id'])
        target = output / 'images' / f'{index:04d}.png'
        with Image.open(io.BytesIO(data)) as image:
            image.convert('RGB').save(target)
        staged.append({**row, 'image': target.relative_to(output).as_posix(),
                       'sha256': digest(target.read_bytes()), 'source_sha256': row['image_sha256']})
    review_manifest = output / 'manifest.jsonl'
    review_manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in staged),
                               encoding='utf-8')
    summary = build(review_manifest, output / 'review')
    report = {'scope': 'reference review only; no OCR text included', 'dataset': DATASET,
              'revision': REVISION, 'source_manifest_sha256': digest(manifest_path.read_bytes()),
              'regions': len(rows), 'private_use_characters': sum(r['reference_private_use_count'] for r in rows),
              'review': summary, 'policy': 'Do not modernize historical spelling or silently replace private-use glyphs.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.manifest, args.output), indent=2))
