"""Archive a replayed human-review import with a separate, closed training gate."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from training.audit_recognizer_work_groups import line_dispositions, validate_groups
from training.adjudicate_reviews import text_hash
from training.build_annotation_review import issues
from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.merge_recognizer_reviewed_pool import replay_package, verified_package


def validate_confirmation(rows, packet, replayed, import_checksums_sha256):
    if (packet.get('schema') != 'slayer-line-proposal-visual-confirmation-v1'
            or packet.get('source_kind') != 'direct-user-message'
            or packet.get('source_actor_role') != 'user'
            or packet.get('scope') != 'all-latest-proposed-complete-line-transcriptions'
            or packet.get('resolved_answer') != 'visually-verified-against-line-crop'
            or not isinstance(packet.get('question'), str) or not packet['question'].strip()
            or not isinstance(packet.get('answer'), str) or not packet['answer'].strip()
            or packet.get('review_export_sha256') != replayed['review_export_sha256']
            or packet.get('review_manifest_sha256') != replayed['review_manifest_sha256']
            or packet.get('base_import_checksums_sha256') != import_checksums_sha256):
        raise ValueError('Visual confirmation provenance or package binding mismatch')
    entries = packet.get('confirmations')
    if not isinstance(entries, list) or not entries or any(not isinstance(e, dict) for e in entries):
        raise ValueError('Expected explicit nonempty visual confirmations')
    by_id = {e.get('id'): e for e in entries}
    expected = {r['id']: r for r in rows if r['import_status'] == 'pending'
                and r['review_status'] == 'proposed' and r['geometry_decision'] == 'complete-line'}
    if (len(by_id) != len(entries) or set(by_id) != set(expected)
            or packet.get('confirmation_count') != len(entries)):
        raise ValueError('Visual confirmation must cover exactly the proposed complete-line scope')
    for key, entry in by_id.items():
        row = expected[key]
        if (entry.get('review_event_id') != row['review_event_id']
                or entry.get('crop_sha256') != row['sha256']
                or entry.get('context_sha256') != row['context']['sha256']
                or entry.get('text_sha256') != text_hash(row['text'])
                or row['reviewed_text_sha256'] != text_hash(row['text'])
                or row['eligible_for_training'] is not False or row['gold'] is not False
                or not row['text'].strip() or '\n' in row['text'] or '\r' in row['text']
                or issues(row['text'])):
            raise ValueError('Visual confirmation event, text or crop binding mismatch')
    return by_id


def gate_rows(rows, pages, groups, confirmations=None):
    confirmations = confirmations or {}
    dispositions = line_dispositions(rows, pages, groups, batch='expansion-review-import')
    results = []
    for row, disposition in zip(rows, dispositions):
        raw_accepted = row['import_status'] == 'single-review-candidate'
        confirmed = row['id'] in confirmations
        accepted = raw_accepted or confirmed
        if row['eligible_for_training'] is not raw_accepted:
            raise ValueError('Import candidate status mismatch')
        if raw_accepted and (row['review_status'] != 'verified'
                         or row['geometry_decision'] != 'complete-line' or row['gold'] is not False):
            raise ValueError('Accepted annotation lacks verified single-line evidence')
        if confirmed and (row['import_status'] != 'pending' or row['review_status'] != 'proposed'
                          or row['geometry_decision'] != 'complete-line' or row['gold'] is not False):
            raise ValueError('Only pending proposed complete-line text can be confirmed')
        result = deepcopy(row)
        result.update(annotation_verified=accepted, eligible_for_training=False,
                      eligible_for_evaluation=False, final_test=False,
                      work_family_group=disposition['work_family_group'],
                      quarantine_for_future_training=disposition['quarantine_for_future_training'],
                      training_gate_reason=disposition['reason'],
                      pre_work_gate_eligible_for_training=row['eligible_for_training'],
                      normalization_applied='none')
        result['image'] = 'history/import/'+result['image']
        result['context']['image'] = 'history/import/'+result['context']['image']
        if confirmations:
            result['annotation_status'] = 'verified' if accepted else 'not-verified'
            result['annotation_verification_source'] = ('direct-user-visual-confirmation' if confirmed
                else 'original-latest-review-event' if raw_accepted else None)
            result['visual_confirmation'] = deepcopy(confirmations.get(row['id']))
        results.append(result)
    return results


def gate(import_directory, work_audit, config, metadata_directory, output, *, confirmation=None):
    import_directory, work_audit, config, metadata_directory, output = map(
        Path, (import_directory, work_audit, config, metadata_directory, output))
    archive = output.with_suffix('.zip')
    if output.exists() or archive.exists():
        raise FileExistsError('Use a fresh gated annotation package')
    import_sums = verified_package(import_directory)
    work_sums = verified_package(work_audit)
    work_report = json.loads((work_audit/'audit-report.json').read_text(encoding='utf-8'))
    policy = json.loads((work_audit/'policy.json').read_text(encoding='utf-8'))
    if (work_report['schema'] != 'slayer-recognizer-work-family-risk-audit-v1'
            or work_report['training_freeze_ready'] is not False
            or work_report['bibliographic_work_identities_verified'] != 0
            or digest(work_audit/'policy.json') != work_report['policy_sha256']
            or digest(import_directory/'original-manifest.jsonl') != work_report['input_sha256']['expansion-manifest.jsonl']
            or digest(work_audit/'page-signatures.jsonl') != work_report['input_sha256']['page-signatures.jsonl']):
        raise ValueError('Work audit or expansion binding mismatch')
    pages = [{**r, 'dataset': work_report['dataset'], 'revision': work_report['revision']}
             for r in read_rows(work_audit/'page-signatures.jsonl')]
    groups = validate_groups(pages, policy['groups'])
    if groups != read_rows(work_audit/'work-family-groups.jsonl'):
        raise ValueError('Work-family groups do not reproduce')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gated-review-', dir=output.parent) as temporary:
        scratch = Path(temporary)
        replayed = replay_package(import_directory, scratch/'replay', config, metadata_directory)
        original = read_rows(import_directory/'reviewed-lines.jsonl')
        packet = json.loads(Path(confirmation).read_text(encoding='utf-8')) if confirmation else None
        confirmations = validate_confirmation(original, packet, replayed,
            digest(import_directory/'checksums.json')) if packet is not None else {}
        rows = gate_rows(original, pages, groups, confirmations)
        stage = scratch/'package'
        (stage/'history/import').mkdir(parents=True)
        for name in sorted(import_sums):
            target = stage/'history/import'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(import_directory/name, target)
        shutil.copyfile(import_directory/'checksums.json', stage/'history/import/checksums.json')
        if confirmation:
            shutil.copyfile(confirmation, stage/'visual-confirmation.json')
        (stage/'work-audit').mkdir()
        for name in ('audit-report.json', 'policy.json', 'work-family-groups.jsonl',
                     'future-training-selection-policy.json', 'source-access.json'):
            shutil.copyfile(work_audit/name, stage/'work-audit'/name)
        write_json(stage/'work-audit/source-package-binding.json', {
            'checksums_sha256': digest(work_audit/'checksums.json'),
            'copied_files_sha256': {n: work_sums[n] for n in ('audit-report.json', 'policy.json',
                'work-family-groups.jsonl', 'future-training-selection-policy.json', 'source-access.json')},
            'scope': 'Subset of separate frozen work audit; not its complete source package'})
        accepted = [r for r in rows if r['annotation_verified']]
        write_rows(stage/'manifest.jsonl', accepted)
        write_rows(stage/'review-ledger.jsonl', rows)
        write_rows(stage/'followup-queue.jsonl', [r for r in rows if not r['annotation_verified']])
        report = {'schema': 'slayer-recognizer-gated-review-import-v1',
            'lines': len(rows), 'events': replayed['events'], 'reviewed_lines': replayed['reviewed_lines'],
            'source_statuses': dict(Counter(r['import_status'] for r in rows)),
            'annotation_accepted': len(accepted), 'followup_lines': len(rows)-len(accepted),
            'accepted_work_family_quarantined': sum(r['quarantine_for_future_training'] for r in accepted),
            'accepted_bibliographic_identity_unresolved': sum(not r['quarantine_for_future_training'] for r in accepted),
            'review_export_sha256': replayed['review_export_sha256'],
            'review_manifest_sha256': replayed['review_manifest_sha256'],
            'manifest_sha256': digest(stage/'manifest.jsonl'),
            'raw_import_checksums_sha256': digest(import_directory/'checksums.json'),
            'work_audit_checksums_sha256': digest(work_audit/'checksums.json'),
            'raw_import_replayed': True, 'human_decisions_modified': 0,
            'normalization_applied': 'none; retain exact UTF-8 text and raw event history',
            'eligible_training_examples': 0, 'gold_labels_created': 0,
            'training_freeze_ready': False, 'model_training_performed': False, 'sota_claim': False,
            'raw_history_has_pre_work_gate_candidate_flags': True,
            'policy': 'Only latest verified + complete-line annotations enter manifest. '
                      'Top-level manifests close training/evaluation gates for all rows. '
                      'Raw historical flags are archival, not a runnable training selection.',
            'limitations': 'Single-human review, not independent gold. Proposed/rejected/unreviewed '
                          'decisions are never promoted. Remaining bibliographic identities unresolved.'}
        if confirmation:
            report.update(schema='slayer-recognizer-gated-review-import-v2',
                visual_confirmation_sha256=digest(confirmation),
                direct_user_confirmed_proposals=len(confirmations),
                original_review_decisions_modified=0,
                policy='Latest verified + complete-line annotations or exactly bound direct-user '
                       'visual confirmations enter the manifest. Original review fields/events '
                       'remain unchanged. All training/evaluation gates remain closed.',
                limitations='Single-human review with explicit follow-up confirmation, not independent '
                            'gold. Rejected crops, unchecked geometries and unreviewed text remain '
                            'excluded. Bibliographic identities remain unresolved.')
        write_json(stage/'gate-report.json', report)
        write_json(stage/'checksums.json', {p.relative_to(stage).as_posix(): digest(p)
            for p in sorted(stage.rglob('*')) if p.is_file()})
        stage.rename(output)
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                stream.write(p, p.relative_to(output).as_posix())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('import-directory', 'work-audit', 'config', 'metadata-directory', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--confirmation', help='Explicit, hash-bound direct-user visual confirmation')
    print(json.dumps(gate(**vars(parser.parse_args())), indent=2))
