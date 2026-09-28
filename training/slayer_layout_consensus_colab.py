"""Safely combine three private teacher evidence ZIPs into layout consensus."""
from __future__ import annotations

import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from training.build_layout_consensus import build_consensus, digest


def safe_extract_zip(archive_path, output_dir):
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            path = PurePosixPath(member.filename.replace('\\', '/'))
            mode = member.external_attr >> 16
            if (path.is_absolute() or '..' in path.parts or
                    (mode and stat.S_ISLNK(mode))):
                raise ValueError(f'Unsafe ZIP member: {member.filename}')
        output_dir.mkdir(parents=True)
        archive.extractall(output_dir)
    return output_dir


def _one(root, name):
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f'Expected one {name}, found {len(matches)}')
    return matches[0]


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _verify_checksums(root):
    checksums_path = _one(root, 'checksums.json')
    checksums = json.loads(checksums_path.read_text(encoding='utf-8'))
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError('Invalid teacher checksums')
    for name, expected in checksums.items():
        if not isinstance(name, str) or '/' in name or '\\' in name:
            raise ValueError('Invalid teacher checksum path')
        path = _one(root, name)
        if digest(path) != expected:
            raise ValueError(f'Teacher checksum mismatch: {name}')


def combine_archives(archives, output_root, config):
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    expected_teachers = set(config['teachers'])
    if len(archives) != len(expected_teachers):
        raise ValueError(f'Expected {len(expected_teachers)} teacher archives')
    output_root.mkdir(parents=True)
    extracted_root = output_root / 'extracted'
    extracted_root.mkdir()
    input_evidence = []
    proposals = []
    page_sets = []
    seen_teachers = set()
    for index, archive in enumerate(map(Path, archives), 1):
        root = safe_extract_zip(archive, extracted_root / f'archive-{index}')
        _verify_checksums(root)
        run_path = _one(root, 'run.json')
        proposal_path = _one(root, 'teacher-proposals.jsonl')
        archived_config = json.loads(
            _one(root, 'experiment-config.json').read_text(encoding='utf-8'))
        if archived_config != config:
            raise ValueError('Teacher experiment config mismatch')
        run = json.loads(run_path.read_text(encoding='utf-8'))
        teacher_id = run.get('teacher_id')
        if teacher_id not in expected_teachers or teacher_id in seen_teachers:
            raise ValueError(f'Unexpected or duplicate teacher: {teacher_id}')
        if run.get('state') != 'completed':
            raise ValueError(f'Incomplete teacher run: {teacher_id}')
        if run.get('teacher', {}).get('revision') != config['teachers'][teacher_id]['revision']:
            raise ValueError(f'Teacher revision mismatch: {teacher_id}')
        rows = _jsonl(proposal_path)
        if len(rows) != run.get('pages_expected'):
            raise ValueError(f'Proposal count mismatch: {teacher_id}')
        if any(row.get('teacher', {}).get('id') != teacher_id for row in rows):
            raise ValueError(f'Proposal teacher mismatch: {teacher_id}')
        if any(row.get('teacher', {}).get('run_id') != run.get('run_id') for row in rows):
            raise ValueError(f'Proposal run mismatch: {teacher_id}')
        page_ids = {row.get('page_id') for row in rows}
        if len(page_ids) != len(rows):
            raise ValueError(f'Duplicate proposal page: {teacher_id}')
        seen_teachers.add(teacher_id)
        proposals.append(proposal_path)
        page_sets.append(page_ids)
        input_evidence.append({
            'teacher_id': teacher_id, 'archive_name': archive.name,
            'archive_sha256': digest(archive), 'run_id': run['run_id'],
            'pages_expected': run['pages_expected'], 'error_pages': run['error_pages'],
            'proposal_sha256': digest(proposal_path),
        })
    if seen_teachers != expected_teachers:
        raise ValueError('Teacher ensemble is incomplete')
    if any(pages != page_sets[0] for pages in page_sets[1:]):
        raise ValueError('Teacher page sets differ')

    consensus_dir = output_root / 'consensus'
    policy = config['consensus']
    report = build_consensus(
        proposals, consensus_dir, categories=policy['categories'],
        quorum=policy['quorum'], iou_threshold=policy['iou_threshold'],
        conflict_iou=policy['conflict_iou'], min_score=policy['min_score'])
    evidence = output_root / 'evidence'
    evidence.mkdir()
    shutil.copytree(consensus_dir, evidence / 'consensus')
    teacher_evidence = evidence / 'teachers'
    teacher_evidence.mkdir()
    for item, source_root in zip(input_evidence, sorted(extracted_root.iterdir())):
        target = teacher_evidence / item['teacher_id']
        target.mkdir()
        for source in source_root.rglob('*'):
            if source.is_file():
                shutil.copyfile(source, target / source.name)
    summary = {
        'schema': 'slayer-layout-teacher-consensus-run-v1',
        'state': 'completed', 'inputs': input_evidence,
        'pages': len(page_sets[0]), 'consensus': report,
        'images_or_references_included': False,
        'automatic_publication': False,
        'claim_boundary': config['claim_boundary'],
    }
    (evidence / 'run.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (evidence / 'experiment-config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    archive = Path(shutil.make_archive(
        str(output_root / 'slayer-layout-consensus-evidence'), 'zip', evidence))
    return archive, summary
