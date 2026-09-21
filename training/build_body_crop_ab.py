"""Build a one-line, manually specified geometry diagnostic, not an evaluation release."""
import argparse
import io
import json
from pathlib import Path
import zipfile

from training.kaggle_body_dev_diagnostic import digest, load_input

INPUT_SHA256 = '913ded2bd5d742099567a2652460baaf3c4a8bb16625492d0931607396dbdd55'
LINE_ID = 'Slawna_wiktoria_FT__437103__r003__line003'
REGION_ID = 'Slawna_wiktoria_FT__437103__r003'
# Selected by visual inspection of the source region, not optimized on model output.
PROPOSED_BOX = (965, 375, 1950, 520)


def build(input_zip, regions_manifest, output):
    from PIL import Image
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, images, _ = load_input(Path(input_zip).read_bytes(), INPUT_SHA256)
    index = next(i for i, row in enumerate(rows) if row['id'] == LINE_ID)
    regions_manifest = Path(regions_manifest)
    sources = [json.loads(s) for s in regions_manifest.read_text(encoding='utf-8').splitlines()]
    source = next(r for r in sources if r['id'] == REGION_ID)
    source_path = (regions_manifest.parent / source['image']).resolve()
    if not source_path.is_relative_to(regions_manifest.parent.resolve()):
        raise ValueError('Source outside region workspace')
    content = source_path.read_bytes()
    if digest(content) != source['sha256']:
        raise ValueError('Region checksum mismatch')
    with Image.open(io.BytesIO(content)) as im:
        x1, y1, x2, y2 = PROPOSED_BOX
        if not 0 <= x1 < x2 <= im.width or not 0 <= y1 < y2 <= im.height:
            raise ValueError('Invalid proposed crop')
        buffer = io.BytesIO()
        im.crop(PROPOSED_BOX).save(buffer, format='PNG')
    output.mkdir(parents=True)
    before, after = images[index], buffer.getvalue()
    selected = []
    for variant, data in [('A-original', before), ('B-manual-geometry', after)]:
        image = variant + '.png'
        (output / image).write_bytes(data)
        selected.append({**rows[index], 'id': LINE_ID + '__' + variant, 'image': image,
                         'sha256': digest(data), 'variant': variant, 'original_line_id': LINE_ID})
    provenance = {'scope': 'diagnostic-only', 'experiment': 'one-line paired geometry control',
                  'original_input_sha256': INPUT_SHA256, 'source_region_sha256': source['sha256'],
                  'source_region_id': REGION_ID, 'original_bbox': [0, 435, 2209, 550],
                  'proposed_bbox': list(PROPOSED_BOX), 'reference_changed': False,
                  'limitations': 'Post-hoc manual geometry example. Not an automatic segmentation fix or quality benchmark. Do not interpret pooled two-variant CER.'}
    archive = output / 'body-crop-ab-input.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(selected, ensure_ascii=False))
        z.writestr('provenance.json', json.dumps(provenance, indent=2))
        for row, data in zip(selected, [before, after]):
            z.writestr(row['image'], data)
    checksum = digest(archive.read_bytes())
    load_input(archive.read_bytes(), checksum)
    (output / 'build.json').write_text(json.dumps({'input_sha256': checksum, **provenance}, indent=2), encoding='utf-8')
    return checksum


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-zip', required=True)
    parser.add_argument('--regions-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(build(args.input_zip, args.regions_manifest, args.output))
