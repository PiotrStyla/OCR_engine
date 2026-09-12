"""Freeze visually transcribed references and source metadata for the tiny scan pilot.

Run after downloading the exact Commons originals into data/public-pilot-v1.
This does not use OCR predictions to construct references.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {
    'elementarz-15': ('104452362', '— 15 —\ntu motyl i tam\nmotyle-tam ryby\ny Y u U'),
    'elementarz-51': ('104452465', '— 51 —\nWanda rysowała. Potem\npisała to, co tu napisane:\nsowa i sówka,\nkoza i kózka,\ngóral, dwór, stróż,\nnóż i nożyk,\nLwów i Kraków.\n4*'),
}


def main():
    source = ROOT / 'data/public-pilot-v1'
    output = ROOT / 'benchmarks/public-pilot-v1'
    output.mkdir(parents=True, exist_ok=True)
    metadata_bytes = (source / 'source-metadata.json').read_bytes()
    metadata = json.loads(metadata_bytes.decode('utf-8-sig'))
    records = []
    for identifier, (source_id, reference) in REFERENCES.items():
        page = next(p for p in metadata['query']['pages'] if source_id in p['title'])
        info = page['imageinfo'][0]
        data = (source / (identifier + '.jpg')).read_bytes()
        if hashlib.sha1(data).hexdigest() != info['sha1']:
            raise ValueError(f'Commons SHA1 mismatch: {identifier}')
        records.append(dict(id=identifier, image=f'../../data/public-pilot-v1/{identifier}.jpg',
            sha256=hashlib.sha256(data).hexdigest(), text=reference,
            source_url=info['descriptionurl'], download_url=info['url'],
            source_sha1=info['sha1'], width=info['width'], height=info['height'],
            license_as_reported=info['extmetadata']['LicenseShortName']['value'],
            document_group='arnoldowa-elementarz-1930', category='printed-cursive',
            reference_method='assistant visual transcription before OCR; no second reviewer',
            scope='Reading order top-to-bottom; page number and footer included; illustrations and artist marks excluded'))
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    (output / 'source-metadata.json').write_bytes(metadata_bytes)
    print(f'Prepared {len(records)} references; source image SHA1 verified against Commons metadata')


if __name__ == '__main__':
    main()
