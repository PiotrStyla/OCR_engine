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
    """PAGE XML must contain at least one TextLine with nonempty Unicode text."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(data)
    ns = {'pc': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
    lines = root.findall('.//pc:TextLine', ns) if ns else root.findall('.//TextLine')
    usable = 0
    for line in lines:
        for equiv in line.iter():
            if equiv.tag.endswith('Unicode') and (equiv.text or '').strip():
                usable += 1
                break
    return len(lines), usable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--out', default=str(ROOT / 'data' / 'impact-corpus'))
    args = parser.parse_args()
    out = Path(args.out)
    img_dir = out / 'pages' / 'images'
    xml_dir = out / 'pages' / 'pagexml'
    manifest = out / 'download_manifest.jsonl'
    rows = [json.loads(line) for line in Path(args.index).read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate ids in index')
    done = [0]
    with manifest.open('a', encoding='utf-8') as fh:
        for row in rows:
            img_path = img_dir / f"{row['id']}.jpg"
            xml_path = xml_dir / f"{row['id']}.xml"
            if img_path.exists() and xml_path.exists():
                done[0] += 1
                continue
            img_data = _get(row['image_url'])
            if len(img_data) < 10_000:
                raise ValueError(f"Image too small for {row['id']}: {len(img_data)} bytes")
            xml_data = _get(row['page_xml_url'])
            total, usable = validate_pagexml(xml_data)
            if usable == 0:
                print(f"{row['id']}: SKIPPED (PAGE XML has {total} lines, 0 with text)")
                continue
            img_path.parent.mkdir(parents=True, exist_ok=True)
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(img_data)
            xml_path.write_bytes(xml_data)
            fh.write(json.dumps({
                'id': row['id'], 'collection': row['collection'],
                'image': str(img_path.relative_to(out)),
                'pagexml': str(xml_path.relative_to(out)),
                'image_sha256': hashlib.sha256(img_data).hexdigest(),
                'pagexml_sha256': hashlib.sha256(xml_data).hexdigest(),
            }, ensure_ascii=False) + '\n')
            fh.flush()
            done[0] += 1
            print(f"{row['id']}: ok ({usable}/{total} lines, {len(img_data)//1024} KB)", flush=True)
    print(f'Done: {done[0]}/{len(rows)} pages in {out}')


if __name__ == '__main__':
    main()
