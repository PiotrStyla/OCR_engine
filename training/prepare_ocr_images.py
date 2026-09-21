"""Re-encode verified page rasters as pixel-identical PNGs for OCR backends."""
import argparse
import hashlib
import json
from pathlib import Path

from training.stage_impact_benchmark import digest, read_rows


def prepare(manifest, output):
    from PIL import Image, __version__ as pillow_version
    manifest, output = Path(manifest), Path(output)
    rows = read_rows(manifest)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Expected unique nonempty records')
    if output.exists():
        raise FileExistsError('Use a new output directory')
    for row in rows:
        if digest(manifest.parent / row['image']) != row['sha256']:
            raise ValueError(f"Source checksum mismatch: {row['id']}")
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    converted, evidence = [], []
    for index, row in enumerate(rows):
        source = manifest.parent / row['image']
        relative = f'images/{index:04d}.png'
        target = output / relative
        with Image.open(source) as image:
            source_format, frames = image.format, image.n_frames if hasattr(image, 'n_frames') else 1
            # Frame zero is explicit; reject files whose later frame is larger.
            image.seek(0)
            size, mode = image.size, image.mode
            if mode not in ('1', 'L', 'RGB', 'RGBA'):
                raise ValueError(f'Unsupported pixel mode: {mode}')
            for frame in range(1, frames):
                image.seek(frame)
                if image.width > size[0] or image.height > size[1]:
                    raise ValueError('Frame zero is not the largest raster')
            image.seek(0)
            pixels = image.tobytes()
            image.save(target, format='PNG')
        with Image.open(target) as check:
            if check.mode != mode or check.size != size or check.tobytes() != pixels:
                raise ValueError('PNG pixel roundtrip mismatch')
        converted.append({**row, 'image': relative, 'sha256': digest(target),
                          'source_sha256': row['sha256']})
        evidence.append({'id': row['id'], 'source_format': source_format, 'source_frames': frames,
                         'selected_frame': 0, 'mode': mode, 'size': size,
                         'pixel_sha256': hashlib.sha256(pixels).hexdigest(),
                         'source_sha256': row['sha256'], 'png_sha256': digest(target)})
    target_manifest = output / 'manifest.jsonl'
    target_manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in converted),
                               encoding='utf-8', newline='\n')
    report = {'pillow_version': pillow_version, 'source_manifest_sha256': digest(manifest),
              'manifest_sha256': digest(target_manifest), 'pages': len(rows),
              'operation': 'frame 0 to PNG, unchanged mode/dimensions/pixels; no resizing',
              'results': evidence}
    (output / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = prepare(args.manifest, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
