"""Download EHRI corpus pages (image + ALTO XML) from a JSONL index.

The index is a JSONL file with rows: {"id", "image_url", "alto_url"}.
Build the index from the eScriptorium export / EHRI transcription source.
Existing files are skipped, so the download is resumable.

Usage: python -m training.download_ehri_corpus --index ehri_index.jsonl --out data/ehri-corpus
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ALTO_NS = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
MAX_BYTES = 100_000_000
USER_AGENT = 'OCR-engine-research/0.1'


def fetch(url, dest):
    parsed = urlparse(url)
    if parsed.scheme != 'https':
        raise ValueError(f'Only https downloads allowed: {url}')
    with urlopen(Request(url, headers={'User-Agent': USER_AGENT}), timeout=120) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f'File exceeds {MAX_BYTES} bytes: {url}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return data


def validate_alto(path):
    """ALTO must contain at least one TextLine with nonempty String CONTENT and BASELINE."""
    root = ET.parse(path).getroot()
    lines = root.findall('.//alto:TextLine', ALTO_NS)
    usable = [tl for tl in lines
              if tl.find('alto:String', ALTO_NS) is not None
              and (tl.find('alto:String', ALTO_NS).get('CONTENT') or '').strip()
              and tl.get('BASELINE')]
    return len(lines), len(usable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, help='JSONL: {"id","image_url","alto_url"}')
    parser.add_argument('--out', default=str(ROOT / 'data' / 'ehri-corpus'))
    parser.add_argument('--limit', type=int, default=0, help='Download at most N pages (0 = all)')
    args = parser.parse_args()
    out = Path(args.out)
    rows = [json.loads(line) for line in Path(args.index).read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate ids in index')
    done = 0
    for row in rows:
        image_path = out / f"{row['id']}.tif"
        alto_path = out / f"{row['id']}.xml"
        if image_path.exists() and alto_path.exists():
            done += 1
            continue
        fetch(row['image_url'], image_path)
        fetch(row['alto_url'], alto_path)
        total, usable = validate_alto(alto_path)
        if usable == 0:
            image_path.unlink()
            alto_path.unlink()
            print(f"{row['id']}: SKIPPED (ALTO has {total} lines, 0 usable)")
            continue
        done += 1
        print(f"{row['id']}: ok ({usable}/{total} lines)")
        if args.limit and done >= args.limit:
            break
    print(f'Done: {done}/{len(rows)} pages in {out}')


if __name__ == '__main__':
    main()
