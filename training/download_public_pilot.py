"""Download the two fixed pilot originals; verify SHA256 before writing.

Usage: python -m training.download_public_pilot
No model, GPU, API key or private documents required.
"""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / 'benchmarks/public-pilot-v1/manifest.jsonl'
    for line in manifest.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        target = (manifest.parent / row['image']).resolve()
        if not target.is_relative_to(ROOT / 'data/public-pilot-v1'):
            raise ValueError('Image outside pilot directory')
        if target.exists():
            data = target.read_bytes()
        else:
            url = row['download_url']
            parsed = urlparse(url)
            if parsed.scheme != 'https' or parsed.hostname != 'upload.wikimedia.org':
                raise ValueError('Unexpected source host')
            with urlopen(Request(url, headers={'User-Agent': 'OCR-engine-research/0.1'}), timeout=60) as response:
                data = response.read(10_000_001)
            if len(data) > 10_000_000:
                raise ValueError('Image exceeds 10 MB limit')
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Image checksum mismatch: {row["id"]}')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        print(f'{row["id"]}: verified')


if __name__ == '__main__':
    main()
