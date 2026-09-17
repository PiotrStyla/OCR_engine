"""Build a download index for the full Polish IMPACT ground truth (PAGE XML).

dLibra (ribes-80.man.poznan.pl) serves each annotated page as its own edition:
https://ribes-80.man.poznan.pl/Content/{edition}/Document/{doc}.tif and returns
404 for document/edition mismatches, so a HEAD probe verifies a mapping exactly.
Edition IDs are assigned in document order, so for a new document the edition is
found by interpolating between the known anchors from the frozen subset
(pages.jsonl of PiotrSty/impact-psnc-polish-ocr) and probing nearby editions.

Usage:
  git clone https://github.com/impactcentre/groundtruth-pol
  python -m training.impact_index --gt-dir groundtruth-pol \
      --pages-jsonl pages.jsonl --out impact_index.jsonl
"""
import argparse
import bisect
import json
import re
import ssl
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://ribes-80.man.poznan.pl'
IMAGE = BASE + '/Content/{eid}/Document/{doc}.tif'
RAW = 'https://raw.githubusercontent.com/impactcentre/groundtruth-pol/master/{collection}/{xml_name}'
UA = {'User-Agent': 'OCR-engine-research/0.1'}
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE  # dLibra TLS cert expired; SHA-256 checks cover integrity


def probe(eid, doc):
    try:
        req = Request(IMAGE.format(eid=eid, doc=doc), method='HEAD', headers=UA)
        with urlopen(req, timeout=30, context=_CTX) as response:
            return response.status == 200
    except Exception:
        return False


def find_edition(doc, anchors):
    """Interpolate between known (doc, edition) anchors, probe nearby editions."""
    docs = [a[0] for a in anchors]
    i = bisect.bisect_left(docs, doc)
    if i < len(anchors) and anchors[i][0] == doc:
        return anchors[i][1]
    guess = None
    if 0 < i < len(anchors):
        (d0, e0), (d1, e1) = anchors[i - 1], anchors[i]
        guess = round(e0 + (doc - d0) * (e1 - e0) / (d1 - d0))
    elif i == 0:
        (d0, e0), (d1, _) = anchors[0], anchors[1]
        guess = e0 - (d0 - doc) * (e1 - e0) / max(d1 - d0, 1)
        guess = round(guess)
    else:
        (d0, _), (d1, e1) = anchors[-2], anchors[-1]
        guess = e1 + (doc - d1) * (e1 - e0) / max(d1 - d0, 1)
        guess = round(guess)
    for delta in (0, 1, -1, 2, -2):
        if guess is not None and probe(guess + delta, doc):
            return guess + delta
    return None



def page_text_len(xml_path):
    total = 0
    for line in xml_path.read_text(encoding='utf-8', errors='replace').splitlines():
        if '<Unicode>' in line:
            inner = line.split('<Unicode>', 1)[1].split('</Unicode>', 1)[0]
            total += len(inner.strip())
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--pages-jsonl', required=True)
    parser.add_argument('--out', default=str(ROOT / 'data' / 'impact_index.jsonl'))
    parser.add_argument('--per-collection-cap', type=int, default=0,
                        help='Cap NEW pages per collection (subset pages always included; 0 = all)')
    parser.add_argument('--limit', type=int, default=0, help='Stop after N index rows')
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()

    anchors_by_col = defaultdict(list)  # collection -> sorted [(doc, edition)]
    subset_docs = set()
    for line in Path(args.pages_jsonl).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        eid = int(row['source_image_url'].split('/Content/')[1].split('/')[0])
        doc = int(row['document_id'])
        anchors_by_col[row['collection']].append((doc, eid))
        subset_docs.add(doc)
    for col in anchors_by_col:
        anchors_by_col[col].sort()

    gt_dir = Path(args.gt_dir)
    gt = {}  # doc -> (collection, xml name)
    empty = 0
    for xml_file in sorted(gt_dir.glob('*/*.xml')):
        m = re.fullmatch(r'0*(\d+)\.xml', xml_file.name)
        if not m:
            continue
        if page_text_len(xml_file) == 0:
            empty += 1
            continue
        gt[int(m.group(1))] = (xml_file.parent.name, xml_file.name)
    print(f'GT: {len(gt)} annotated pages (skipped {empty} empty) across '
          f'{len({c for c, _ in gt.values()})} collections')

    targets = defaultdict(dict)  # collection -> doc -> xml name
    for doc, (col, xml_name) in gt.items():
        targets[col][doc] = xml_name

    allowed_new = set()
    if args.per_collection_cap:
        for col in sorted(targets):
            new_docs = [d for d in sorted(targets[col]) if d not in subset_docs]
            allowed_new.update(new_docs[:args.per_collection_cap])

    def resolve(item):
        col, doc, xml_name = item
        if doc in subset_docs:
            return doc, col, xml_name, anchors_by_col[col]
        if args.per_collection_cap and doc not in allowed_new:
            return doc, col, xml_name, []  # cap reached: skip probing
        return doc, col, xml_name, anchors_by_col.get(col) or []

    work = [(col, doc, xml_name) for col in sorted(targets) for doc, xml_name in sorted(targets[col].items())]
    resolved = {}
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for doc, col, xml_name, anchors in pool.map(resolve, work):
            done += 1
            if done % 500 == 0:
                print(f'  resolve progress: {done}/{len(work)}', flush=True)
            if doc in subset_docs:
                resolved[doc] = anchors_by_col[col][bisect.bisect_left([a[0] for a in anchors], doc)][1]
            elif anchors:
                eid = find_edition(doc, anchors)
                if eid is not None:
                    resolved[doc] = eid
    print(f'Resolved editions: {len(resolved)}/{len(gt)}')

    rows = []
    new_per_col = defaultdict(int)
    for col in sorted(targets):
        for doc, xml_name in sorted(targets[col].items()):
            if doc not in resolved:
                continue
            is_subset = doc in subset_docs
            if not is_subset and args.per_collection_cap and \
               new_per_col[col] >= args.per_collection_cap:
                continue
            if not is_subset:
                new_per_col[col] += 1
            rows.append({
                'id': f'{col}__{doc}',
                'collection': col,
                'doc_id': doc,
                'edition_id': resolved[doc],
                'in_subset': is_subset,
                'image_url': IMAGE.format(eid=resolved[doc], doc=doc),
                'page_xml_url': RAW.format(collection=col, xml_name=xml_name),
            })
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Index: {len(rows)} rows ({len(subset_docs)} subset + {len(rows) - min(len(subset_docs), len(rows))} new) -> {out}')


if __name__ == '__main__':
    main()
