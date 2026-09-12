"""Freeze pre-OCR visual references for two historical printed pages."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'data/print-pilot-v1'
    output = ROOT / 'benchmarks/print-pilot-v1'
    metadata_bytes = (source / 'source-metadata.json').read_bytes()
    metadata = json.loads(metadata_bytes.decode('utf-8-sig'))
    records = []
    for identifier, filename, match, group in [
        ('odezwa-12', 'odezwa-12.png', '012.png', 'krolicki-odezwa-do-matek'),
        ('torun-74', 'torun.jpg', '55498679', 'torunski-elementarz-1910'),
    ]:
        page = next(p for p in metadata['query']['pages'] if match in p['title'])
        info = page['imageinfo'][0]
        data = (source / filename).read_bytes()
        if hashlib.sha1(data).hexdigest() != info['sha1']:
            raise ValueError(f'Commons SHA1 mismatch: {identifier}')
        records.append(dict(id=identifier, image=f'../../data/print-pilot-v1/{filename}',
            sha256=hashlib.sha256(data).hexdigest(),
            text=(output/'references'/f'{identifier}.txt').read_text(encoding='utf-8').strip(),
            source_url=info['descriptionurl'], download_url=info['url'], source_sha1=info['sha1'],
            width=info['width'], height=info['height'],
            license_as_reported=info['extmetadata']['LicenseShortName']['value'],
            document_group=group, category='historical-print',
            reference_method='assistant visual transcription before OCR; no second reviewer',
            scope='All printed text in reading order, including page number; line-end hyphens retained; decorative rules excluded'))
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    (output/'source-metadata.json').write_bytes(metadata_bytes)
    print(f'Prepared {len(records)} references; Commons SHA1 verified')


if __name__ == '__main__':
    main()
