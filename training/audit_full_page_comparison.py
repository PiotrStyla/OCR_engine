"""Validate a returned v5 evidence ZIP and recompute metrics without GPU inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import zipfile

from training.full_page_comparison import compare, repetition_flags
from training.full_page_pilot import digest, read_rows, safe_relative, write_json


ROOT = Path(__file__).resolve().parents[1]


def audit(archive_path, dataset, config_path, output, code_revision):
    archive_path, dataset, config_path, output = map(Path, (archive_path, dataset, config_path, output))
    if output.exists():
        raise FileExistsError('Use a new audit directory')
    if len(code_revision) != 40 or any(char not in '0123456789abcdef' for char in code_revision):
        raise ValueError('Expected a full pinned Git revision')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    expected_code = {'code_revision': code_revision}
    for key,name in [('runner_sha256','full_page_pilot.py'), ('comparison_sha256','full_page_comparison.py')]:
        expected_code[key] = hashlib.sha256(subprocess.check_output(
            ['git','show',f'{code_revision}:training/{name}'], cwd=ROOT)).hexdigest()
    with zipfile.ZipFile(archive_path) as stream:
        members = stream.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or sum(item.file_size for item in members) > 100_000_000:
            raise ValueError('Duplicate or oversized evidence members')
        for item in members:
            name = safe_relative(item.filename)
            if (item.is_dir() or stat.S_ISLNK(item.external_attr >> 16)
                    or (Path(name).suffix not in ('.json','.jsonl','.csv','.log')
                        and name != 'cer-comparison.png')):
                raise ValueError('Unexpected evidence member')
        checksums = json.loads(stream.read('checksums.json'))
        if set(checksums) != set(names)-{'checksums.json'}:
            raise ValueError('Evidence checksum coverage mismatch')
        payload = {name:stream.read(name) for name in names}
        if any(hashlib.sha256(payload[name]).hexdigest() != sha for name,sha in checksums.items()):
            raise ValueError('Evidence member checksum mismatch')
    decode = lambda name: json.loads(payload[name])
    if decode('config.json') != config:
        raise ValueError('Frozen configuration mismatch')
    provenance = decode('code-provenance.json')
    if any(provenance.get(key) != value for key,value in expected_code.items()):
        raise ValueError('Pinned code provenance mismatch')
    manifest_hash = hashlib.sha256(payload['dataset/manifest.jsonl']).hexdigest()
    baseline_hash = hashlib.sha256(payload['dataset/retained-projected-predictions.jsonl']).hexdigest()
    if (manifest_hash != config['dataset']['manifest_sha256']
            or manifest_hash != digest(dataset/'manifest.jsonl')
            or baseline_hash != config['dataset']['baseline_sha256']
            or baseline_hash != digest(dataset/config['baseline']['source'])):
        raise ValueError('Frozen reference/baseline mismatch')
    rows = read_rows(dataset/'manifest.jsonl')
    if (len(rows) != config['dataset']['pages']
            or any(row['split'] != 'validation' or row.get('final_test') is not False for row in rows)):
        raise ValueError('Unexpected page scope')
    inputs = [json.loads(line) for line in payload['dataset/inference-inputs.jsonl'].splitlines()]
    expected_inputs = [{key:row[key] for key in ('id','image','sha256','width','height')}
                       | {'source_regions':[]} for row in rows]
    if inputs != expected_inputs:
        raise ValueError('Model inputs differ or contain reference/geometry leakage')
    identity = decode('predictions/qwen3-vl-4b/identity.json')
    input_hash = hashlib.sha256(payload['dataset/inference-inputs.jsonl']).hexdigest()
    if (identity != {'engine':'qwen3-vl-4b', 'spec':config['models']['qwen3-vl-4b'],
                     'input_sha256':input_hash, 'runner_sha256':expected_code['runner_sha256']}):
        raise ValueError('Worker identity mismatch')
    staging = decode('dataset/staging-provenance.json')
    if (staging['bundle_sha256'] != config['dataset']['archive_sha256']
            or staging['inputs_sha256'] != input_hash
            or staging['reference_text_sent_to_model'] is not False
            or staging['source_geometry_sent_to_model'] is not False):
        raise ValueError('Staging provenance mismatch')
    environment = decode('predictions/qwen3-vl-4b/environment.json')
    packages = {key.lower():value for key,value in environment['packages'].items()}
    for package in config['models']['qwen3-vl-4b']['packages']:
        name,version = package.split('==')
        if name != 'sentencepiece' and packages.get(name.lower()) != version:
            raise ValueError('Worker package version mismatch')
    if environment['reference_text_sent_to_model'] is not False:
        raise ValueError('Worker reference leakage attestation mismatch')
    predictions = [json.loads(line) for line in payload['predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl'].splitlines()]
    if (len(predictions) != len(rows) or len({row['id'] for row in predictions}) != len(rows)
            or {row['id'] for row in predictions} != {row['id'] for row in rows}):
        raise ValueError('Incomplete or duplicate candidate coverage')
    for prediction in predictions:
        if prediction['status'] == 'ok':
            trace = prediction['generation_trace']
            tokens = trace['generated_token_ids']
            eos = bool(tokens and tokens[-1] in trace['eos_token_ids'])
            capped = len(tokens) >= config['models']['qwen3-vl-4b']['max_new_tokens'] and not eos
            reason = 'eos' if eos else ('length' if capped else 'unknown')
            if (prediction['generated_tokens'] != len(tokens)
                    or prediction['token_limit_reached'] != capped or prediction['finish_reason'] != reason):
                raise ValueError('Generation token/termination mismatch')
    output.mkdir(parents=True)
    extracted = output/'evidence'
    for name,data in payload.items():
        target = extracted/safe_relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    recomputed = compare(config, dataset,
        extracted/'predictions/qwen3-vl-4b/qwen3-vl-4b-full-page.jsonl', output/'recomputed')
    if recomputed != decode('scores/metrics.json'):
        raise ValueError('Recomputed metrics mismatch')
    if ((output/'recomputed/qwen3-vl-4b-projected.jsonl').read_bytes()
            != payload['scores/qwen3-vl-4b-projected.jsonl']):
        raise ValueError('Saved scored projection mismatch')
    report = {'schema':'slayer-full-page-comparison-v5-audit', 'source_archive_sha256':digest(archive_path),
        'members_verified':len(names), 'metrics_recomputed':True, 'pages':len(rows),
        'code_provenance':expected_code, 'runtime_reported':environment,
        'process_result':decode('process-result.json'), 'reports':recomputed['reports'],
        'glyph_counts':{char:{'reference':sum(row['text'].count(char) for row in rows),
                             'candidate':sum(row['text'].count(char) for row in predictions)}
                        for char in ('\u017f','\u00e1','\u0247')},
        'glyph_count_is_recall':False,
        'candidate_inference_seconds':sum(row['elapsed_seconds'] for row in predictions),
        'peak_allocated_bytes':max(row['peak_allocated_bytes'] for row in predictions),
        'capped_pages':[row['id'] for row in predictions if row.get('token_limit_reached')],
        'repetition_flagged_pages':[row['id'] for row in predictions
                                   if repetition_flags(row['text'])['suspicious_repetition']],
        'gold_pages':0, 'production_promotion':False, 'automatic_teacher_promotion':False,
        'sota_claim':False, 'claim_boundary':recomputed['claim_boundary']}
    shutil.copyfile(archive_path, output/'source-evidence.zip')
    write_json(output/'audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive','dataset','config','output','code-revision'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    report = audit(args.archive,args.dataset,args.config,args.output,args.code_revision)
    print(json.dumps({key:report[key] for key in ('source_archive_sha256','members_verified','pages',
        'metrics_recomputed','candidate_inference_seconds','peak_allocated_bytes','capped_pages','glyph_counts')},
        ensure_ascii=False, indent=2))
