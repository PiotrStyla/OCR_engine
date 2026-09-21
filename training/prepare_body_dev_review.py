"""Prepare unverified body-text line candidates; never create gold labels automatically."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.request import urlopen

from training.build_annotation_review import build as build_review
from training.kaggle_printed_dev_control import DATASET, REVISION, FROZEN, safe_file

TITLE_PAGES = {'Choragiew_FT__436794', 'Relacja_koronacji_FT__436547', 'Wiesc_FT__436868'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def choose(rows, test_rows, per_collection=4):
    if not 1 <= per_collection <= 10:
        raise ValueError('Use 1-10 regions per collection')
    hashes = {r['image_sha256'] for r in test_rows}
    groups = defaultdict(lambda: defaultdict(list))
    excluded, seen = [], set()
    for row in sorted(rows, key=lambda r: r['id']):
        if row['id'] in seen or not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']):
            raise ValueError('Invalid or duplicate region ID')
        seen.add(row['id'])
        if row['split'] != 'validation' or row['collection'] in FROZEN:
            raise ValueError('Frozen test collection or incorrect split')
        if row['image_sha256'] in hashes:
            raise ValueError('Exact crop overlap with test')
        reason = None
        if row['page_id'] in TITLE_PAGES:
            reason = 'previous heading-control page'
        elif row['region_type'] != 'paragraph':
            reason = 'not a paragraph'
        elif len(row['text'].splitlines()) < 3:
            reason = 'fewer than three reference lines'
        elif any(not line.strip() for line in row['text'].splitlines()):
            reason = 'blank reference line; requires separate alignment'
        if reason:
            excluded.append({'id': row['id'], 'reason': reason})
        else:
            groups[row['collection']][row['page_id']].append(row)
    selected = []
    # Round-robin across pages before selecting more regions from any one page.
    for collection in sorted(groups):
        pages = groups[collection]
        count = 0
        for offset in range(max(map(len, pages.values()))):
            for page in sorted(pages):
                if offset >= len(pages[page]):
                    continue
                row = pages[page][offset]
                if count < per_collection:
                    selected.append(row)
                    count += 1
                else:
                    excluded.append({'id': row['id'], 'reason': 'deterministic collection quota'})
    if not selected:
        raise ValueError('No eligible body-text regions')
    return selected, excluded


def align_candidates(boxes, reference, size):
    """Count agreement is a proposal only, never verification of line identity."""
    width, height = size
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    for x1, y1, x2, y2 in boxes:
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError('Invalid detected geometry')
    lines = reference.splitlines()
    if len(boxes) != len(lines):
        return [], 'line-count-mismatch'
    if any(b[1] < a[3] for a, b in zip(boxes, boxes[1:])):
        return [], 'vertically-overlapping-candidates'
    # Tight ink bands clip ascenders. Keep full width and split inter-line gaps;
    # this may retain neighbouring fragments, so visual verification is required.
    edges = [0] + [(a[3] + b[1]) // 2 for a, b in zip(boxes, boxes[1:])] + [height]
    crops = [(0, top, width, bottom) for top, bottom in zip(edges, edges[1:])]
    return list(zip(crops, lines)), 'unverified-count-matched'


def prepare(output, per_collection=4):
    from PIL import Image
    import cv2
    import numpy as np
    from ocr.opencv_detector import _character_height, _split_block
    from ocr.result import BBox

    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True)
    for folder in ['sources', 'regions', 'lines']:
        (output / folder).mkdir()

    def download(name):
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{safe_file(name)}'
        with urlopen(url, timeout=90) as response:
            return response.read()

    def save_json(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    def metadata(name):
        data = download(name)
        (output / 'sources' / name.replace('/', '--')).write_bytes(data)
        return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]

    rows = metadata('regions/validation/metadata.jsonl')
    test = metadata('regions/test/metadata.jsonl')
    pages = {r['id']: r for r in metadata('pages/validation/metadata.jsonl')}
    selected, excluded = choose(rows, test, per_collection)
    save_json('selection.json', selected)
    save_json('excluded.json', excluded)
    line_rows, region_rows, outcomes = [], [], []
    for row in selected:
        data = download('regions/validation/' + safe_file(row['file_name']))
        if sha(data) != row['image_sha256']:
            raise ValueError('Source crop checksum mismatch: ' + row['id'])
        source = output / 'sources' / (row['id'] + '.jpg')
        source.write_bytes(data)
        with Image.open(source) as image:
            image = image.convert('RGB')
            size = image.size
            region_path = output / 'regions' / (row['id'] + '.png')
            image.save(region_path)
            provenance = {'collection': row['collection'], 'page_id': row['page_id'],
                          'source_region_id': row['id'], 'source_sha256': sha(data),
                          'dataset': DATASET, 'revision': REVISION, 'license': row['license'],
                          'source_page': pages[row['page_id']], 'split': 'development-review',
                          'review_status': 'unverified', 'eligible_for_evaluation': False}
            region_rows.append(dict(provenance, id=row['id'], image=region_path.relative_to(output).as_posix(),
                                    sha256=sha(region_path.read_bytes()), text=row['text']))
            gray = np.array(image.convert('L'))
            boxes = _split_block(gray, BBox(0, 0, size[0], size[1]), _character_height(gray))
            coords = [(b.x1, b.y1, b.x2, b.y2) for b in boxes]
            pairs, status = align_candidates(coords, row['text'], size)
            outcomes.append({'id': row['id'], 'status': status, 'detected_boxes': coords,
                             'reference_lines': len(row['text'].splitlines()), 'candidates': len(pairs)})
            for index, (box, text) in enumerate(pairs):
                line_id = row['id'] + f'__line{index:03d}'
                line_path = output / 'lines' / (line_id + '.png')
                image.crop(box).save(line_path)
                line_rows.append(dict(provenance, id=line_id, image=line_path.relative_to(output).as_posix(),
                                      sha256=sha(line_path.read_bytes()), text=text, bbox_in_region=box,
                                      reference_line_index=index, alignment='count-only-proposal'))
        print(row['id'], status, len(pairs), flush=True)
    for name, records in [('region-manifest.jsonl', region_rows), ('candidate-lines.jsonl', line_rows)]:
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    build_review(output / 'region-manifest.jsonl', output / 'region-review')
    if line_rows:
        build_review(output / 'candidate-lines.jsonl', output / 'line-review')
    report = {'scope': 'Unverified development review candidates. NOT gold labels or a benchmark.',
              'regions': len(region_rows), 'pages': len({r['page_id'] for r in region_rows}),
              'collections': sorted({r['collection'] for r in region_rows}),
              'reference_lines': sum(o['reference_lines'] for o in outcomes),
              'candidate_lines': len(line_rows), 'outcomes': outcomes,
              'detector': 'existing OpenCV _character_height + _split_block; full-width crops split at inter-band midpoints; no reference-guided splitting',
              'opencv': cv2.__version__, 'numpy': np.__version__,
              'limits': 'No near-duplicate or upstream training audit. Collection IDs are not proof of different works. No human review performed.'}
    save_json('report.json', report)
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'docs/BODY_DEV_REVIEW.md', output / 'README.md')
    save_json('checksums.json', {p.relative_to(output).as_posix(): sha(p.read_bytes()) for p in output.rglob('*') if p.is_file()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-collection', type=int, default=4)
    args = parser.parse_args()
    report = prepare(args.output, args.per_collection)
    print(json.dumps({k: v for k, v in report.items() if k != 'outcomes'}, indent=2))


if __name__ == '__main__':
    main()
