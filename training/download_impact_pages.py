"""Download IMPACT-Polish pages (image + PAGE XML) from an impact_index.jsonl.

Images come from dLibra (expired TLS certificate - verification is disabled and
SHA-256 of every downloaded file is recorded in the output manifest). PAGE XML
files come from raw.githubusercontent.com (valid TLS).

Output layout (under --out):
  pages/images/{id}.jpg
  pages/pagexml/{id}.xml
  download_manifest.jsonl  - {id, image, pagexml, image_sha256, pagexml_sha256}

Existing files are skipped, so the download is resumable.

Usage:
  python -m training.download_impact_pages --index impact_index.jsonl --out data/impact-corpus
"""
import argparse
import hashlib
import json
import ssl
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 100_000_000
UA = {'User-Agent': 'OCR-engine-research/0.1'}
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE  # dLibra TLS cert expired


def _get(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https':
        raise ValueError(f'Only https downloads allowed: {url}')
    with urlopen(Request(url, headers=UA), timeout=180, context=_CTX) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f'File exceeds {MAX_BYTES} bytes: {url}')
    return data


def validate_pagexml(data):
    """PAGE XML must contain text, either line-level (TextLine/TextEquiv) or
    region-level (TextRegion/TextEquiv without line segmentation)."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(data)
    ns = {'pc': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

    def _texts(tag):
        if not ns:
            return [e for e in root.iter(tag) if (e.text or '').strip()]
        return [e for e in root.iter() if e.tag.endswith(tag) and (e.text or '').strip()]

    unicode_texts = _texts('Unicode')
    lines = root.findall('.//pc:TextLine', ns) or root.findall('.//TextLine')
    mode = 'line' if lines else ('region' if unicode_texts else 'empty')
    return len(lines), len(unicode_texts), mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--out', default=str(ROOT / 'data' / 'impact-corpus'))
    parser.add_argument('--limit', type=int, default=0, help='Download at most N pages (0 = all)')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / 'pages' / 'images'
    xml_dir = out / 'pages' / 'pagexml'
    manifest = out / 'download_manifest.jsonl'
    rows = [json.loads(line) for line in Path(args.index).read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate ids in index')
    done = [0]
    with manifest.open('a', encoding='utf-8') as fh:
        lock = threading.Lock()
        todo = [row for row in rows
                if not (img_dir / f"{row['id']}.jpg").exists()
                or not (xml_dir / f"{row['id']}.xml").exists()]
        print(f'To download: {len(todo)}/{len(rows)}')

        def download_one(row):
            try:
                return _download_one(row, img_dir, xml_dir, out, fh, lock, done, todo)
            except Exception as e:
                return f"{row['id']}: FAILED ({type(e).__name__}: {str(e)[:100]})"

        with ThreadPoolExecutor(args.workers) as pool:
            for msg in pool.map(download_one, todo):
                print(msg, flush=True)
    print(f'Done: {done[0]}/{len(rows)} pages in {out}')


def _download_one(row, img_dir, xml_dir, out, fh, lock, done, todo):
    img_path = img_dir / f"{row['id']}.jpg"
    xml_path = xml_dir / f"{row['id']}.xml"
    img_data = _get(row['image_url'])
    if len(img_data) < 10_000:
        raise ValueError(f"Image too small: {len(img_data)} bytes")
    xml_data = _get(row['page_xml_url'])
    total, usable, mode = validate_pagexml(xml_data)
    if mode == 'empty':
        return f"{row['id']}: SKIPPED (PAGE XML has no text)"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(img_data)
    xml_path.write_bytes(xml_data)
    with lock:
        fh.write(json.dumps({
            'id': row['id'], 'collection': row['collection'],
            'image': str(img_path.relative_to(out)),
            'pagexml': str(xml_path.relative_to(out)),
            'text_mode': mode,
            'image_sha256': hashlib.sha256(img_data).hexdigest(),
            'pagexml_sha256': hashlib.sha256(xml_data).hexdigest(),
        }, ensure_ascii=False) + '\n')
        fh.flush()
        done[0] += 1
        if done[0] % 10 == 0:
            print(f'  progress: {done[0]}/{len(todo)}', flush=True)
    return f"{row['id']}: ok ({mode}-level, {usable} texts, {len(img_data)//1024} KB)"


if __name__ == '__main__':
    main()
