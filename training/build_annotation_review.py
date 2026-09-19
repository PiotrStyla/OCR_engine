"""Build an offline annotation review workspace from a verified page manifest."""
import argparse
import hashlib
import json
import shutil
import unicodedata
from pathlib import Path

from training.stage_impact_benchmark import digest, read_rows


def issues(text):
    result = []
    offset = 0
    for index, character in enumerate(text):
        width = len(character.encode('utf-16-le')) // 2
        if character == '\ufffd' or unicodedata.category(character) == 'Co':
            result.append({'offset': offset, 'length': width,
                           'codepoint': f'U+{ord(character):04X}',
                           'kind': 'replacement' if character == '\ufffd' else 'private',
                           'context': text[max(0, index - 22):index + 23]})
        offset += width
    return result


def build(manifest, output):
    from PIL import Image
    manifest, output = Path(manifest), Path(output)
    rows = read_rows(manifest)
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Expected unique nonempty page IDs')
    if output.exists():
        raise FileExistsError('Use a new review directory')
    for row in rows:
        source = manifest.parent / row['image']
        if digest(source) != row['sha256']:
            raise ValueError(f"Image checksum mismatch: {row['id']}")
        with Image.open(source) as image:
            if image.format != 'PNG':
                raise ValueError('Review requires PNG inputs; run training.prepare_ocr_images first')
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    pages = []
    for index, row in enumerate(rows):
        relative = f'images/{index:04d}.png'
        shutil.copyfile(manifest.parent / row['image'], output / relative)
        pages.append({'id': row['id'], 'image': relative, 'image_sha256': row['sha256'],
                      'source_image_sha256': row.get('source_sha256', row['sha256']),
                      'text': row['text'],
                      'text_sha256': hashlib.sha256(row['text'].encode('utf-8')).hexdigest(),
                      'issues': issues(row['text'])})
    payload = {'schema': 'polocrbench-review-source-v1', 'manifest_sha256': digest(manifest),
               'pages': pages}
    assets = Path(__file__).resolve().parents[1] / 'tools' / 'annotation-review'
    template = (assets / 'index.html').read_text(encoding='utf-8')
    encoded = json.dumps(payload, ensure_ascii=True).replace('<', '\\u003c')
    (output / 'index.html').write_text(template.replace('<!-- SOURCE_DATA -->', encoded),
                                      encoding='utf-8')
    for name in ('app.js', 'style.css'):
        shutil.copyfile(assets / name, output / name)
    summary = {'manifest_sha256': payload['manifest_sha256'], 'pages': len(pages),
               'flagged_pages': sum(bool(page['issues']) for page in pages),
               'issues': sum(len(page['issues']) for page in pages)}
    (output / 'source-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.output), indent=2))


if __name__ == '__main__':
    main()
