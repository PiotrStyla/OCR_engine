"""Build the frozen EHRI-Polish benchmark split (document level) and manifests.

Splits at DOCUMENT level (EHRI-ET-ZIH3010201_01 -> doc ZIH3010201) so pages of
the same document never straddle train/val/test (no leakage). The split is
written once and then frozen: regenerating refuses to overwrite split.json.

Outputs (under --outdir, e.g. benchmarks/ehri-polish-v1):
  split.json            frozen document->split assignment
  train_manifest.txt    XML paths for ketos (-f xml), one per training page
  val_manifest.txt      XML paths for ketos validation pages
  test_manifest.jsonl   benchmark_pages.py format: {id, image, text, sha256}

Usage:
  python -m training.ehri_split --data-dir data/ehri-corpus --outdir benchmarks/ehri-polish-v1
"""
import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ALTO_NS = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}


def doc_of(stem):
    """EHRI-ET-ZIH3010884_02 -> ZIH3010884."""
    return stem.rsplit('_', 1)[0]


def page_text(alto_path):
    root = ET.parse(alto_path).getroot()
    lines = []
    for tl in root.findall('.//alto:TextLine', ALTO_NS):
        s = tl.find('alto:String', ALTO_NS)
        if s is not None and (s.get('CONTENT') or '').strip():
            lines.append(s.get('CONTENT'))
    text = unicodedata.normalize('NFC', '\n'.join(lines))
    if not text.strip():
        raise ValueError(f'No usable TextLines in {alto_path}')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--val-docs', type=int, default=4)
    parser.add_argument('--test-docs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tifs = sorted(data_dir.glob('*.tif'))
    if not tifs:
        raise SystemExit(f'No .tif files in {data_dir}')
    pages = {}
    for tif in tifs:
        alto = tif.with_suffix('.xml')
        if not alto.exists():
            raise SystemExit(f'Unpaired page (missing {alto.name})')
        pages[tif.stem] = alto
    docs = sorted({doc_of(stem) for stem in pages})
    print(f'{len(pages)} pages across {len(docs)} documents')

    split_path = outdir / 'split.json'
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding='utf-8'))
        missing = set(docs) - set(split)
        if missing:
            raise SystemExit(f'split.json exists but lacks documents {sorted(missing)}; '
                             'frozen split must not change - add new docs via a new benchmark version')
        print(f'Using frozen split from {split_path}')
    else:
        if len(docs) < args.val_docs + args.test_docs + 1:
            raise SystemExit(f'{len(docs)} documents is not enough for '
                             f'{args.val_docs} val + {args.test_docs} test documents')
        rng = random.Random(args.seed)
        shuffled = docs[:]
        rng.shuffle(shuffled)
        split = {d: 'test' for d in shuffled[:args.test_docs]}
        split.update({d: 'val' for d in shuffled[args.test_docs:args.test_docs + args.val_docs]})
        split.update({d: 'train' for d in shuffled[args.test_docs + args.val_docs:]})
        split_path.write_text(json.dumps(split, indent=2, sort_keys=True), encoding='utf-8')
        print(f'FROZEN new split -> {split_path} (never regenerate with different args)')

    counts = {'train': [], 'val': [], 'test': []}
    for stem, alto in pages.items():
        counts[split[doc_of(stem)]].append(stem)
    for name, stems in counts.items():
        print(f'  {name}: {len(stems)} pages / {len({doc_of(s) for s in stems})} docs')

    train_manifest = outdir / 'train_manifest.txt'
    val_manifest = outdir / 'val_manifest.txt'
    train_manifest.write_text('\n'.join(str(pages[s].resolve()) for s in sorted(counts['train'])), encoding='utf-8')
    val_manifest.write_text('\n'.join(str(pages[s].resolve()) for s in sorted(counts['val'])), encoding='utf-8')

    test_manifest = outdir / 'test_manifest.jsonl'
    with test_manifest.open('w', encoding='utf-8') as fh:
        for stem in sorted(counts['test']):
            image = data_dir / f'{stem}.tif'
            row = {'id': stem,
                   'image': str(image.relative_to(outdir.parent) if outdir.parent in image.parents else image),
                   'text': page_text(pages[stem]),
                   'sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Wrote {train_manifest.name}, {val_manifest.name}, {test_manifest.name} ({len(counts["test"])} test pages)')


if __name__ == '__main__':
    main()
