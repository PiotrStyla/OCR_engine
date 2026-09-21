"""Stage the pinned HF IMPACT test archive without changing frozen manifests."""
import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

REVISION = 'a2480fde6f15284701458ff370b81cce50dc5c2d'
ARCHIVE_SHA256 = '0a9ffa126029703726fc5883a8279ddccf15763c4fdd483ed9b8cc034bdb42f0'
SOURCE_URL = ('https://huggingface.co/datasets/PiotrSty/impact-print-v2/resolve/'
              + REVISION + '/impact-print-v2-test.tar.gz')


def stream_digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(chunk)
    return result.hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return stream_digest(stream)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').split('\n')
            if line.strip()]


def stage(archive, manifest, train_pool, output, expected_sha256=ARCHIVE_SHA256):
    archive, manifest, train_pool, output = map(Path, (archive, manifest, train_pool, output))
    archive_hash = digest(archive)
    if archive_hash != expected_sha256:
        raise ValueError('Archive checksum mismatch')
    if output.exists():
        raise FileExistsError('Use a new output directory; staged evidence is immutable')
    rows, train = read_rows(manifest), read_rows(train_pool)
    ids = [row['id'] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Test IDs must be nonempty and unique')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in ids):
        raise ValueError('Unsafe test ID')
    collections = {value.split('__')[0] for value in ids}
    if collections & {row['id'].split('__')[0] for row in train}:
        raise ValueError('Train/test collection overlap')
    if {row['sha256'] for row in rows} & {row['sha256'] for row in train}:
        raise ValueError('Train/test image hash overlap')

    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member names')
        by_name = {member.name: member for member in members}
        selected = []
        # Select exact known member names; never extract archive-controlled paths.
        for row in rows:
            name = f"impact-print-v2/../impact-corpus/pages/images/{row['id']}.jpg"
            member = by_name.get(name)
            if member is None or not member.isfile() or member.size > 100_000_000:
                raise ValueError(f"Missing or invalid image member: {row['id']}")
            with bundle.extractfile(member) as stream:
                if stream_digest(stream) != row['sha256']:
                    raise ValueError(f"Image checksum mismatch: {row['id']}")
            selected.append(member)

        output.mkdir(parents=True)
        (output / 'images').mkdir()
        portable = []
        for row, member in zip(rows, selected):
            image = f"images/{row['id']}.jpg"
            with bundle.extractfile(member) as stream:
                (output / image).write_bytes(stream.read())
            portable.append({**row, 'image': image})
    staged = output / 'manifest.jsonl'
    staged.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in portable),
                      encoding='utf-8', newline='\n')
    report = {
        'source_url': SOURCE_URL, 'source_revision': REVISION,
        'archive_sha256': archive_hash, 'frozen_manifest_sha256': digest(manifest),
        'train_pool_sha256': digest(train_pool), 'staged_manifest_sha256': digest(staged),
        'pages_verified': len(rows), 'train_regions': len(train),
        'test_collections': sorted(collections), 'collection_overlap': [],
        'image_hash_overlap': [], 'path_format': 'relative POSIX',
        'limitations': 'Exact hashes and collection IDs only; no near-duplicate audit.',
    }
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--manifest', default='benchmarks/polocrbench/history_testA_manifest.jsonl')
    parser.add_argument('--train-pool', default='benchmarks/polocrbench/history_train_pool.jsonl')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.archive, args.manifest, args.train_pool, args.output), indent=2))


if __name__ == '__main__':
    main()
