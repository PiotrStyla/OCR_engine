"""Build the frozen IMPACT-Polish v2 benchmark split and manifests.

Splits are inherited from the frozen v1 subset (pages.jsonl of
PiotrSty/impact-psnc-polish-ocr, split assigned per source collection), so v2
results stay comparable with v1. Same collections -> same splits, always.

Outputs (under --outdir, e.g. benchmarks/impact-print-v2):
  split.json                     collection -> split (frozen)
  test_manifest.jsonl            benchmark_pages.py format: {id, image, text, sha256}
  train_regions.jsonl            per-region records for recognizer training:
                                 {id, image, text, bbox, source_page, sha256}
  val_regions.jsonl              same format, validation split
  stats.json                     counts per split/collection

Page text is built from PAGE XML TextRegion/TextEquiv (region-level GT; the
IMPACT corpus has no TextLine segmentation). Region order follows the
ReadingOrder when present, otherwise document order.

Usage:
  python -m training.impact_split --corpus-dir data/impact-corpus \
      --pages-jsonl pages.jsonl --outdir benchmarks/impact-print-v2
"""
import argparse
import hashlib
import json
import unicodedata
import xml.etree.ElementTree as ET
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def region_list(xml_path):
    """Return [(bbox, text)] per TextRegion, ordered by ReadingOrder when possible."""
    root = ET.parse(xml_path).getroot()
    ns = root.tag.split('}')[0].strip('{') if '}' in root.tag else None
    q = f'{{{ns}}}' if ns else ''

    regions = {}
    for reg in root.iter(f'{q}TextRegion' if ns else 'TextRegion'):
        coords = reg.find(f'{q}Coords' if ns else 'Coords')
        poly = (coords.get('points') or '') if coords is not None else ''
        xs, ys = [], []
        for pair in poly.split():
            x, _, y = pair.partition(',')
            if x.isdigit() and y.isdigit():
                xs.append(int(x))
                ys.append(int(y))
        bbox = (min(xs), min(ys), max(xs), max(ys)) if xs else None
        parts = [e.text.strip() for e in reg.iter(f'{q}Unicode' if ns else 'Unicode')
                 if (e.text or '').strip()]
        if parts:
            regions[reg.get('id')] = (bbox, unicodedata.normalize('NFC', ' '.join(parts)))

    if not regions:
        return []

    # ReadingOrder: regionRefIndexed order inside the top OrderedGroup
    order = []
    for ref in root.iter(f'{q}RegionRefIndexed' if ns else 'RegionRefIndexed'):
        rid = ref.get('regionRef')
        if rid in regions:
            order.append((int(ref.get('index', 0)), rid))
    if order:
        ordered = [regions[rid] for _, rid in sorted(order)]
        ordered += [v for k, v in regions.items() if k not in {rid for _, rid in order}]
    else:
        ordered = list(regions.values())
    return ordered


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', required=True, help='dir with download_manifest.jsonl')
    parser.add_argument('--pages-jsonl', required=True, help='frozen v1 pages.jsonl (split source)')
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    corpus = Path(args.corpus_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # frozen split: collection -> split, inherited from v1
    v1_split = {}
    for line in Path(args.pages_jsonl).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        v1_split[row['collection']] = row['split']

    rows = [json.loads(l) for l in (corpus / 'download_manifest.jsonl')
            .read_text(encoding='utf-8').splitlines() if l.strip()]
    unknown = {r['collection'] for r in rows} - set(v1_split)
    if unknown:
        raise SystemExit(f'Collections missing from v1 split: {sorted(unknown)}; '
                         'a frozen benchmark must not silently assign them')

    split_path = outdir / 'split.json'
    current = {c: s for c, s in v1_split.items()}
    if split_path.exists():
        frozen = json.loads(split_path.read_text(encoding='utf-8'))
        if frozen != current:
            raise SystemExit('split.json exists and differs from v1 assignment; refusing to rewrite')
    else:
        split_path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding='utf-8')
    print(f'Split (frozen): {dict(sorted((s, sum(1 for v in current.values() if v == s)) for s in set(current.values())))}')

    region_files = {'train': outdir / 'train_regions.jsonl', 'val': outdir / 'val_regions.jsonl'}
    test_manifest = outdir / 'test_manifest.jsonl'
    fhs = {k: v.open('w', encoding='utf-8') for k, v in region_files.items()}
    stats = defaultdict(lambda: defaultdict(int))
    test_fh = test_manifest.open('w', encoding='utf-8')

    def norm_split(s):
        return {'validation': 'val', 'dev': 'val'}.get(s, s)

    try:
        for row in sorted(rows, key=lambda r: r['id']):
            split = norm_split(v1_split[row['collection']])
            img_path = corpus / row['image']
            xml_path = corpus / row['pagexml']
            regions = region_list(xml_path)
            stats[split][row['collection']] += 1
            if split == 'test':
                text = unicodedata.normalize('NFC', '\n'.join(t for _, t in regions))
                if not text.strip():
                    raise SystemExit(f'Empty test page text: {row["id"]}')
                test_fh.write(json.dumps({
                    'id': row['id'],
                    'image': os.path.relpath(img_path, outdir),
                    'text': text,
                    'sha256': sha256(img_path),
                }, ensure_ascii=False) + '\n')
            else:
                for i, (bbox, text) in enumerate(regions):
                    fhs[split].write(json.dumps({
                        'id': f'{row["id"]}__r{i:02d}',
                        'image': os.path.relpath(img_path, outdir),
                        'text': text,
                        'bbox': list(bbox) if bbox else None,
                        'source_page': row['id'],
                        'sha256': sha256(img_path),
                    }, ensure_ascii=False) + '\n')
    finally:
        for fh in fhs.values():
            fh.close()
        test_fh.close()

    stats_out = {s: {c: n for c, n in sorted(cols.items())} for s, cols in sorted(stats.items())}
    (outdir / 'stats.json').write_text(json.dumps(stats_out, ensure_ascii=False, indent=2), encoding='utf-8')
    for name in ('test_manifest.jsonl', 'train_regions.jsonl', 'val_regions.jsonl'):
        n = sum(1 for _ in (outdir / name).open(encoding='utf-8'))
        print(f'{name}: {n} rows')


if __name__ == '__main__':
    main()
