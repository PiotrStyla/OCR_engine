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
from training.full_page_pilot import digest, read_rows, write_json, write_rows
from training.merge_recognizer_reviewed_pool import replay_package, verified_package


def gate_rows(rows, pages, groups):
    dispositions = line_dispositions(rows, pages, groups, batch='expansion-review-import')
    results = []
    for row, disposition in zip(rows, dispositions):
        accepted = row['import_status'] == 'single-review-candidate'
        if row['eligible_for_training'] is not accepted:
            raise ValueError('Import candidate status mismatch')
        if accepted and (row['review_status'] != 'verified'
                         or row['geometry_decision'] != 'complete-line' or row['gold'] is not False):
            raise ValueError('Accepted annotation lacks verified single-line evidence')
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
        results.append(result)
    return results


def gate(import_directory, work_audit, config, metadata_directory, output):
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
        rows = gate_rows(original, pages, groups)
        stage = scratch/'package'
        (stage/'history/import').mkdir(parents=True)
        for name in sorted(import_sums):
            target = stage/'history/import'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(import_directory/name, target)
        shutil.copyfile(import_directory/'checksums.json', stage/'history/import/checksums.json')
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
    print(json.dumps(gate(**vars(parser.parse_args())), indent=2))
