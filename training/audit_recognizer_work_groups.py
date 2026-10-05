"""Conservative work-family risk overlay; never rewrite human annotations."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.merge_recognizer_reviewed_pool import verified_package


def collect_sources(policy, output, *, opener=urlopen):
    """Record access evidence, not bibliographic identity or copied catalog text."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh source receipt')
    receipts = []
    for source in policy['sources']:
        url = source['url']
        if not url.startswith('https://'):
            raise ValueError('Bibliographic sources require HTTPS')
        row = {'source_id': source['id'], 'url': url,
               'checked_at': datetime.now(timezone.utc).isoformat(),
               'identity_verified': False, 'scope': source['scope']}
        try:
            request = Request(url, headers={'User-Agent': 'SLAYER-OCR bibliographic audit/1.0'})
            with opener(request, timeout=15) as response:
                body = response.read(5_000_001)
                if len(body) > 5_000_000:
                    raise ValueError('Source exceeds bounded metadata download')
                row.update(status='downloaded', http_status=response.status,
                           final_url=response.geturl(), bytes=len(body),
                           sha256=hashlib.sha256(body).hexdigest())
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            row.update(status='unavailable', error_type=type(exc).__name__, error=str(exc))
        receipts.append(row)
    write_json(output, {'schema': 'slayer-bibliographic-source-access-v1', 'sources': receipts})
    return receipts


def validate_groups(pages, groups):
    page_by_id = {r['id']: r for r in pages}
    if len(page_by_id) != len(pages):
        raise ValueError('Duplicate page ID')
    collections = {r['collection'] for r in pages}
    occupied, identifiers, results = set(), set(), []
    for group in groups:
        identifier = group['id']
        members = group['collections']
        if (not identifier or identifier in identifiers or len(members) < 2
                or len(set(members)) != len(members) or not set(members) <= collections
                or occupied & set(members)):
            raise ValueError('Invalid or overlapping work-family group')
        if (group.get('bibliographic_identity_verified') is not False
                or group.get('exact_scan_duplicate_claim') is not False
                or group.get('status') != 'protective-work-family-group'):
            raise ValueError('This protocol cannot certify identities or exact duplicates')
        evidence = group['visual_evidence_page_ids']
        if (len(evidence) < 2 or len(set(evidence)) != len(evidence)
                or any(p not in page_by_id for p in evidence)
                or {page_by_id[p]['collection'] for p in evidence} != set(members)):
            raise ValueError('Visual evidence must bind every group collection')
        rows = [r for r in pages if r['collection'] in members]
        splits = sorted({r['audit_split'] for r in rows})
        results.append({**group, 'pages': [r['id'] for r in rows], 'audit_splits': splits,
                        'cross_split_risk': len(splits) > 1,
                        'training_collections_to_quarantine': sorted({r['collection'] for r in rows
                            if r['audit_split'] == 'train' and len(splits) > 1}),
                        'visual_evidence': [{'page_id': p, 'image_sha256': page_by_id[p]['image_sha256'],
                            'source_page_url': page_by_id[p]['source_page_url']} for p in evidence]})
        identifiers.add(identifier)
        occupied.update(members)
    return results


def line_dispositions(rows, pages, groups, *, batch):
    page_by_id = {r['id']: r for r in pages}
    group_by_collection = {c: g for g in groups for c in g['collections']}
    seen, result = set(), []
    for row in rows:
        page = page_by_id.get(row['page_id'])
        if (row['id'] in seen or page is None or row['collection'] != page['collection']
                or row['dataset'] != pages[0]['dataset']
                or row['revision'] != pages[0]['revision'] or page['audit_split'] != 'train'
                or row['source_split'] != 'train'
                or row.get('eligible_for_evaluation') is not False
                or row.get('final_test') is not False):
            raise ValueError('Line/source identity or train-only boundary mismatch')
        group = group_by_collection.get(row['collection'])
        quarantine = bool(group and group['cross_split_risk'])
        result.append({'id': row['id'], 'batch': batch, 'page_id': row['page_id'],
                       'collection': row['collection'], 'crop_sha256': row['sha256'],
                       'root_line_id': row.get('root_line_id', row['id']),
                       'original_eligible_for_training': row['eligible_for_training'],
                       'work_family_group': group['id'] if group else None,
                       'quarantine_for_future_training': quarantine,
                       'reason': 'cross-split-work-family-risk' if quarantine else 'bibliographic-identity-unresolved',
                       'eligible_for_training': False, 'eligible_for_evaluation': False,
                       'human_decision_modified': False})
        seen.add(row['id'])
    return result


def audit(source_audit, policy_path, sources_path, reviewed_pool, expansion_review, output):
    source_audit, policy_path, sources_path, reviewed_pool, expansion_review, output = map(
        Path, (source_audit, policy_path, sources_path, reviewed_pool, expansion_review, output))
    if output.exists():
        raise FileExistsError('Use a fresh work-family audit directory')
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    if policy['schema'] != 'slayer-recognizer-work-family-policy-v1':
        raise ValueError('Unsupported work-family policy')
    for root in (source_audit, reviewed_pool, expansion_review):
        verified_package(root)
    files = {'page-signatures.jsonl': source_audit/'page-signatures.jsonl',
             'source-audit-report.json': source_audit/'audit-report.json',
             'reviewed-manifest.jsonl': reviewed_pool/'manifest.jsonl',
             'expansion-manifest.jsonl': expansion_review/'input/manifest.jsonl'}
    if any(digest(p) != policy['input_sha256'][n] for n, p in files.items()):
        raise ValueError('Frozen input mismatch')
    parent = json.loads(files['source-audit-report.json'].read_text(encoding='utf-8'))
    pages = read_rows(files['page-signatures.jsonl'])
    if (not pages or len(pages) != policy['pages'] or parent['pages'] != len(pages)
            or parent['dataset'] != policy['dataset'] or parent['revision'] != policy['revision']):
        raise ValueError('Page source mismatch')
    pages = [{**r, 'dataset': parent['dataset'], 'revision': parent['revision']} for r in pages]
    sources = json.loads(sources_path.read_text(encoding='utf-8'))
    expected = {s['id']: s for s in policy['sources']}
    receipts = sources['sources']
    if (sources['schema'] != 'slayer-bibliographic-source-access-v1'
            or len(receipts) != len(expected) or {s['source_id'] for s in receipts} != set(expected)):
        raise ValueError('Source receipt coverage mismatch')
    for receipt in receipts:
        source = expected[receipt['source_id']]
        if (receipt['url'] != source['url'] or receipt['scope'] != source['scope']
                or receipt['identity_verified'] is not False
                or receipt['status'] not in ('downloaded', 'unavailable')):
            raise ValueError('Source receipt identity mismatch')
        if receipt['status'] == 'downloaded' and (
                not re.fullmatch('[0-9a-f]{64}', receipt.get('sha256', ''))
                or type(receipt.get('bytes')) is not int or not 0 < receipt['bytes'] <= 5_000_000
                or receipt.get('http_status') != 200
                or not receipt.get('final_url', '').startswith('https://')):
            raise ValueError('Invalid downloaded source receipt')
        if receipt['status'] == 'unavailable' and not receipt.get('error_type'):
            raise ValueError('Invalid unavailable source receipt')
    if any(not set(g['source_ids']) <= set(expected) for g in policy['groups']):
        raise ValueError('Unknown group source')
    groups = validate_groups(pages, policy['groups'])
    blocked = sorted({c for g in groups for c in g['training_collections_to_quarantine']})
    batches = {'reviewed-pool': read_rows(files['reviewed-manifest.jsonl']),
               'expansion-review': read_rows(files['expansion-manifest.jsonl'])}
    dispositions = [r for batch, rows in batches.items()
                    for r in line_dispositions(rows, pages, groups, batch=batch)]
    if len({r['root_line_id'] for r in dispositions}) != len(dispositions):
        raise ValueError('Duplicate root across input batches')
    nontrain = {r['collection'] for r in pages if r['audit_split'] != 'train'}
    remaining = [r for r in pages if r['audit_split'] == 'train' and r['collection'] not in blocked]
    report = {'schema': 'slayer-recognizer-work-family-risk-audit-v1',
              'dataset': policy['dataset'], 'revision': policy['revision'], 'pages': len(pages),
              'input_sha256': {n: digest(p) for n, p in files.items()},
              'policy_sha256': digest(policy_path), 'source_receipts_sha256': digest(sources_path),
              'protective_groups': len(groups),
              'cross_split_groups': sum(g['cross_split_risk'] for g in groups),
              'training_collections_to_quarantine': blocked,
              'training_pages_to_quarantine': sum(r['audit_split'] == 'train' and r['collection'] in blocked for r in pages),
              'remaining_train_source_pages_not_cleared': len(remaining),
              'remaining_train_source_collections_not_cleared': len({r['collection'] for r in remaining}),
              'batch_impact': {b: {'lines': len(rows), 'quarantined': sum(r['batch'] == b and r['quarantine_for_future_training'] for r in dispositions)}
                               for b, rows in batches.items()},
              'source_access': dict(Counter(s['status'] for s in receipts)),
              'bibliographic_work_identities_verified': 0, 'human_decisions_modified': 0,
              'training_freeze_ready': False, 'sota_claim': False,
              'limitations': 'Protective families are not verified bibliographic edition IDs. Remaining collections are unresolved, not cleared. Old runs/labels remain unchanged. This does not audit upstream pretraining or the separate 36-page evaluation archive.'}
    output.mkdir(parents=True)
    for name, path in files.items():
        shutil.copyfile(path, output/name)
    shutil.copyfile(policy_path, output/'policy.json')
    shutil.copyfile(sources_path, output/'source-access.json')
    write_rows(output/'work-family-groups.jsonl', groups)
    write_rows(output/'line-dispositions.jsonl', dispositions)
    write_json(output/'future-training-selection-policy.json', {
        'dataset': policy['dataset'], 'revision': policy['revision'],
        'forbidden_collections': sorted(nontrain | set(blocked)),
        'training_freeze_ready': False, 'requires_resolved_bibliographic_groups': True,
        'scope': 'New future runs only; no retroactive split or annotation edits'})
    write_json(output/'audit-report.json', report)
    write_json(output/'checksums.json', {p.name: digest(p) for p in output.iterdir() if p.is_file()})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--sources', required=True)
    parser.add_argument('--collect-sources', action='store_true')
    for name in ('source-audit', 'reviewed-pool', 'expansion-review', 'output'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if args.collect_sources:
        collect_sources(json.loads(Path(args.policy).read_text(encoding='utf-8')), args.sources)
    else:
        if any(getattr(args, name) is None for name in ('source_audit', 'reviewed_pool', 'expansion_review', 'output')):
            parser.error('All audit inputs and output are required')
        print(json.dumps(audit(args.source_audit, args.policy, args.sources, args.reviewed_pool,
                               args.expansion_review, args.output), indent=2))
