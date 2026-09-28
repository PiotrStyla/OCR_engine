"""Safely combine three private teacher evidence ZIPs into layout consensus."""
from __future__ import annotations

import hashlib
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


def combine_archives(archives, output_root, config, consensus_policy=None,
                     code_revision=None):
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    if code_revision is not None:
        if (not isinstance(code_revision, str) or len(code_revision) != 40 or
                any(character not in '0123456789abcdef' for character in code_revision)):
            raise ValueError('Invalid code revision')
    expected_teachers = set(config['teachers'])
    policy = dict(consensus_policy or config['consensus'])
    if policy.get('categories') != config['consensus']['categories']:
        raise ValueError('Consensus categories must match the frozen teacher ontology')
    allow_abstentions = policy.get('allow_teacher_abstentions', False)
    max_error_pages = policy.get('max_teacher_error_pages', 0)
    max_error_fraction = policy.get('max_teacher_error_fraction', 0.0)
    if (not isinstance(allow_abstentions, bool) or
            not isinstance(max_error_pages, int) or isinstance(max_error_pages, bool) or
            max_error_pages < 0 or
            not isinstance(max_error_fraction, (int, float)) or
            isinstance(max_error_fraction, bool) or
            not 0 <= max_error_fraction <= 1):
        raise ValueError('Invalid teacher abstention policy')
    if len(archives) != len(expected_teachers):
        raise ValueError(f'Expected {len(expected_teachers)} teacher archives')
    output_root.mkdir(parents=True)
    extracted_root = output_root / 'extracted'
    extracted_root.mkdir()
    input_evidence = []
    proposals = []
    page_sets = []
    seen_teachers = set()
    abstentions = []
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
        pages_expected = run.get('pages_expected')
        pages_completed = run.get('pages_completed')
        error_pages = run.get('error_pages')
        if (not isinstance(pages_expected, int) or pages_expected <= 0 or
                not isinstance(pages_completed, int) or pages_completed < 0 or
                not isinstance(error_pages, int) or error_pages < 0 or
                pages_completed + error_pages != pages_expected):
            raise ValueError(f'Invalid teacher page accounting: {teacher_id}')
        if error_pages and (not allow_abstentions or error_pages > max_error_pages or
                            error_pages / pages_expected > max_error_fraction):
            raise ValueError(f'Teacher run contains page errors: {teacher_id}')
        if run.get('teacher', {}).get('revision') != config['teachers'][teacher_id]['revision']:
            raise ValueError(f'Teacher revision mismatch: {teacher_id}')
        rows = _jsonl(proposal_path)
        if len(rows) != run.get('pages_expected'):
            raise ValueError(f'Proposal count mismatch: {teacher_id}')
        error_rows = []
        for row in rows:
            status, error = row.get('status'), row.get('error')
            if status == 'ok' and error is None:
                continue
            if (status == 'error' and isinstance(error, str) and error.strip() and
                    row.get('detections') == []):
                error_rows.append(row)
                continue
            raise ValueError(f'Invalid teacher proposal status: {teacher_id}')
        if len(error_rows) != error_pages:
            raise ValueError(f'Teacher proposal error count mismatch: {teacher_id}')
        for row in error_rows:
            abstentions.append({
                'schema': 'slayer-layout-teacher-abstention-v1',
                'teacher_id': teacher_id,
                'teacher_revision': run['teacher']['revision'],
                'run_id': run['run_id'],
                'page_id': row['page_id'],
                'image_sha256': row['image']['sha256'],
                'error': row['error'],
            })
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
    report = build_consensus(
        proposals, consensus_dir, categories=policy['categories'],
        quorum=policy['quorum'], iou_threshold=policy['iou_threshold'],
        conflict_iou=policy['conflict_iou'], min_score=policy['min_score'],
        containment_threshold=policy.get('containment_threshold', 0.9),
        granularity_ratio=policy.get('granularity_ratio', 2.0))
    if abstentions:
        abstentions.sort(key=lambda item: (item['page_id'], item['teacher_id']))
        (consensus_dir / 'teacher-abstentions.jsonl').write_text(
            ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in abstentions),
            encoding='utf-8', newline='\n')
        hard_path = consensus_dir / 'hard-examples.jsonl'
        hard_by_page = {
            item['page_id']: item for item in _jsonl(hard_path)
        }
        for item in abstentions:
            hard = hard_by_page.setdefault(item['page_id'], {
                'schema': 'slayer-hard-example-v1',
                'page_id': item['page_id'],
                'image_sha256': item['image_sha256'],
                'reasons': [],
                'review_object_ids': [],
                'source': 'teacher-consensus',
            })
            hard['reasons'] = sorted(set(hard['reasons']) | {'teacher-abstention'})
            hard['abstaining_teachers'] = sorted(
                set(hard.get('abstaining_teachers', [])) | {item['teacher_id']})
        hard_path.write_text(
            ''.join(json.dumps(hard_by_page[page_id], ensure_ascii=False) + '\n'
                    for page_id in sorted(hard_by_page)),
            encoding='utf-8', newline='\n')
        report['hard_example_pages'] = len(hard_by_page)
        report['teacher_abstentions'] = len(abstentions)
        report['teacher_abstention_pages'] = len({item['page_id'] for item in abstentions})
        report['limitations'].append(
            'Explicit teacher abstentions do not count as votes; affected pages require review.')
        (consensus_dir / 'report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        artifacts = sorted(
            path for path in consensus_dir.iterdir()
            if path.is_file() and path.name != 'checksums.sha256')
        (consensus_dir / 'checksums.sha256').write_text(
            ''.join(f'{digest(path)}  {path.name}\n' for path in artifacts),
            encoding='utf-8', newline='\n')
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
    policy_path = evidence / 'consensus-policy.json'
    policy_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = {
        'schema': 'slayer-layout-teacher-consensus-run-v1',
        'state': 'completed', 'inputs': input_evidence,
        'pages': len(page_sets[0]), 'consensus': report,
        'teacher_abstentions': abstentions,
        'code_revision': code_revision,
        'consensus_policy_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
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
