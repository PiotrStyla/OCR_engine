"""Verify and archive a code-free Colab pilot ZIP, retaining failed predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from training.benchmark_pages import normalize
from training.full_page_pilot import digest, markdown_text, safe_relative, write_json


def verify_archive(archive):
    with zipfile.ZipFile(archive) as stream:
        infos = stream.infolist()
        if len(infos) != len({item.filename for item in infos}):
            raise ValueError('Duplicate archive members')
        if sum(item.file_size for item in infos) > 32 * 1024 * 1024:
            raise ValueError('Evidence ZIP exceeds the bounded audit size')
        payloads = {}
        for item in infos:
            name = safe_relative(item.filename)
            if item.is_dir() or Path(name).suffix not in ('.json', '.jsonl', '.log'):
                raise ValueError('Expected code-free JSON/JSONL/log evidence')
            payloads[name] = stream.read(item)
    checksums = json.loads(payloads['checksums.json'])
    if set(checksums) != set(payloads) - {'checksums.json'}:
        raise ValueError('Checksum coverage mismatch')
    if any(hashlib.sha256(payloads[name]).hexdigest() != value for name, value in checksums.items()):
        raise ValueError('Evidence checksum mismatch')
    return payloads


def audit(archive, output, expected_config):
    from jiwer import cer, wer
    archive, output = Path(archive), Path(output)
    payloads = verify_archive(archive)
    config = json.loads(payloads['config.json'])
    if config != json.loads(Path(expected_config).read_text(encoding='utf-8')):
        raise ValueError('Archived configuration differs from the frozen experiment')
    rows = [json.loads(line) for line in payloads['dataset/manifest.jsonl'].decode().splitlines()]
    ids = {row['id'] for row in rows}
    if len(ids) != len(rows) or not rows:
        raise ValueError('Expected unique nonempty page IDs')
    reports = json.loads(payloads['scores/metrics.json'])['reports']
    summary = {}
    for variant, report in reports.items():
        engine = 'ovis-ocr2' if variant.startswith('ovis-') else 'mixed-v3'
        records = [json.loads(line) for line in payloads[f'predictions/{engine}/{variant}.jsonl'].decode().splitlines()]
        if len(records) != len({row['id'] for row in records}) or {row['id'] for row in records} != ids:
            raise ValueError('Prediction page identity mismatch')
        identity = json.loads(payloads[f'predictions/{engine}/identity.json'])
        if (identity['spec'] != config['models'][engine]
                or identity['input_sha256'] != hashlib.sha256(payloads['dataset/inference-inputs.jsonl']).hexdigest()
                or identity['runner_sha256'] != json.loads(payloads['code-provenance.json'])['embedded_runner_sha256']):
            raise ValueError('Prediction provenance mismatch')
        by_id = {row['id']: row for row in records}
        references = [normalize(row['text']) for row in rows]
        hypotheses = []
        for row in rows:
            record = by_id[row['id']]
            if record['status'] not in ('ok', 'error'):
                raise ValueError('Unexpected prediction status')
            text = record['text'] if record['status'] == 'ok' else ''
            hypotheses.append(normalize(markdown_text(text) if record.get('format') == 'markdown' else text))
        recalculated = {'cer_micro':cer(references,hypotheses), 'wer_micro':wer(references,hypotheses)}
        if any(abs(recalculated[key] - report[key]) > 1e-12 for key in recalculated):
            raise ValueError('Recomputed metrics do not match the archive')
        errors = [row for row in records if row['status'] == 'error']
        truncated = sum(bool(row.get('token_limit_reached')) for row in records)
        if report['errors_or_missing'] != len(errors) or report['token_limit_pages'] != truncated:
            raise ValueError('Execution/truncation counts differ from archived metrics')
        for record in records:
            if engine == 'ovis-ocr2' and record['status'] == 'ok':
                geometry = record.get('input_geometry')
                spec = config['models'][engine]
                if (not geometry or geometry['processed_pixels'] > spec['max_pixels']
                        or geometry['requested_max_pixels'] != spec['max_pixels']
                        or geometry['visual_tokens'] > spec.get('max_visual_tokens', geometry['visual_tokens'])
                        or record['generated_tokens'] > spec['max_new_tokens']):
                    raise ValueError('Ovis input/output budget metadata mismatch')
        summary[variant] = {**recalculated, 'pages':len(rows), 'successful_pages':len(rows)-len(errors),
                            'failed_pages':len(errors), 'errors':[row.get('error') for row in errors],
                            'token_limit_pages':truncated, 'execution_complete':not errors,
                            'complete_generation_pages':sum(row['status']=='ok' and not row.get('token_limit_reached') for row in records),
                            'quality_comparison_available':not errors and not truncated,
                            'annotation_assisted':variant.endswith('source-regions')}
    if output.exists():
        raise FileExistsError('Use a new audit directory; never overwrite evidence')
    output.mkdir(parents=True)
    for name, data in payloads.items():
        path = output/'evidence'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    shutil.copyfile(archive, output/'source-evidence.zip')
    result = {'schema':'slayer-full-page-pilot-archive-audit-v1',
              'source_archive_sha256':digest(archive), 'members_verified':len(payloads),
              'metrics_recomputed':True, 'page_ids':[row['id'] for row in rows],
              'reports':summary, 'reference_status':'source-unreviewed',
              'head_to_head_quality_comparison_available':all(row['quality_comparison_available'] for row in summary.values()),
              'sota_claim':False, 'model_promotion':False}
    write_json(output/'audit.json', result)
    write_json(output/'checksums.json', {path.relative_to(output).as_posix():digest(path)
                                      for path in output.rglob('*') if path.is_file()})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive,args.output,args.config), ensure_ascii=False, indent=2))
