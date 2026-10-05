"""Audit training-source teacher evidence without treating weak labels as gold."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import zipfile

from training import recognizer_data_pilot as pilot
from training.full_page_pilot import digest, read_rows, safe_relative, write_json

ROOT = Path(__file__).resolve().parents[1]
EOS = {'qwen3-vl-4b': [151645, 151643], 'trocr-mixed-v3': [2]}


def validate_generation(row, reference, spec, engine):
    if row.get('status') == 'error':
        if row.get('text') != '' or not isinstance(row.get('error'), str) or not row['error']:
            raise ValueError('Error prediction must retain error and empty text')
        return
    if row.get('status') != 'ok' or not isinstance(row.get('text'), str):
        raise ValueError('Invalid prediction status/text')
    trace = row['generation_trace']
    tokens = trace['generated_token_ids']
    if (not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens)
            or trace['eos_token_ids'] != EOS[engine] or len(tokens) > spec['max_new_tokens']):
        raise ValueError('Invalid generation token trace')
    stops = [i for i, token in enumerate(tokens) if token in EOS[engine]]
    ended = bool(stops)
    if ended and (len(stops) != 1 or any(t != 1 for t in tokens[stops[0]+1:])
                  or (engine == 'qwen3-vl-4b' and stops[0] != len(tokens)-1)):
        raise ValueError('Unexpected tokens after EOS')
    capped = len(tokens) >= spec['max_new_tokens'] and not ended
    if (row.get('generated_tokens') != len(tokens) or row.get('token_limit_reached') is not capped
            or row.get('finish_reason') != ('eos' if ended else ('length' if capped else 'unknown'))):
        raise ValueError('Generation termination mismatch')
    geometry = row['input_geometry']
    if (geometry.get('original_width') != reference['width']
            or geometry.get('original_height') != reference['height']):
        raise ValueError('Original crop dimensions mismatch')
    if engine == 'trocr-mixed-v3':
        if geometry.get('processor_tensor_shape') != [1, 3, 384, 384]:
            raise ValueError('Unexpected TrOCR processor tensor shape')
    else:
        grid = geometry['image_grid_thw']
        if (len(grid) != 1 or len(grid[0]) != 3 or grid[0][0] != 1
                or any(type(n) is not int or n <= 0 for n in grid[0])
                or geometry['patch_size'] != 16 or geometry['merge_size'] != 2
                or grid[0][1] % 2 or grid[0][2] % 2):
            raise ValueError('Unexpected Qwen visual grid')
        _, h, w = grid[0]
        expected = {'processed_width': w*16, 'processed_height': h*16,
            'processed_pixels': h*w*256, 'visual_tokens': h*w//4,
            'min_pixels': spec['min_pixels'], 'max_pixels': spec['max_pixels']}
        if (any(geometry.get(key) != value for key, value in expected.items())
                or not spec['min_pixels'] <= expected['processed_pixels'] <= spec['max_pixels']):
            raise ValueError('Qwen processed geometry mismatch')


def audit(archive, input_bundle, config_path, output, code_revision):
    archive, input_bundle, config_path, output = map(Path, (archive, input_bundle, config_path, output))
    if output.exists():
        raise FileExistsError('Use a new audit directory')
    if not re.fullmatch('[0-9a-f]{40}', code_revision):
        raise ValueError('Full published inference code revision required')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    code = {'code_revision': code_revision, 'reference_text_sent_to_models': False,
            'gpu_execution_validated_locally': False}
    hashes = {}
    for engine, key, name in [('trocr-mixed-v3', 'runner_sha256', 'recognizer_data_pilot.py'),
                             ('qwen3-vl-4b', 'qwen_runner_sha256', 'full_page_pilot.py')]:
        # Git blobs are canonical LF bytes, unlike a Windows worktree checkout.
        code[key] = hashlib.sha256(subprocess.check_output(
            ['git', 'show', f'{code_revision}:training/{name}'], cwd=ROOT)).hexdigest()
        hashes[engine] = code[key]
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or sum(i.file_size for i in members) > 20_000_000:
            raise ValueError('Duplicate or oversized evidence members')
        for item in members:
            if (safe_relative(item.filename) != item.filename or item.is_dir()
                    or stat.S_ISLNK(item.external_attr >> 16)
                    or Path(item.filename).suffix not in ('.json', '.jsonl', '.log', '.csv')):
                raise ValueError('Unsafe evidence member')
        payload = {name: stream.read(name) for name in names}
    checksums = json.loads(payload['checksums.json'])
    if set(checksums) != set(names)-{'checksums.json'} or any(
            hashlib.sha256(payload[name]).hexdigest() != sha for name, sha in checksums.items()):
        raise ValueError('Evidence checksum coverage mismatch')
    decode = lambda name: json.loads(payload[name])
    rows = lambda name: [json.loads(line) for line in payload[name].splitlines() if line.strip()]
    if decode('config.json') != config or decode('code-provenance.json') != code:
        raise ValueError('Frozen configuration/code provenance mismatch')
    with zipfile.ZipFile(input_bundle) as stream:
        for name in ('manifest.jsonl', 'inference-inputs.jsonl', 'report.json', 'excluded.jsonl'):
            if payload['dataset/'+name] != stream.read(name):
                raise ValueError('Evidence dataset differs from frozen input')
    references = rows('dataset/manifest.jsonl')
    input_hash = hashlib.sha256(payload['dataset/inference-inputs.jsonl']).hexdigest()
    staging = {'lines': len(references), 'labels_sent_to_models': False,
               'training_examples_created': 0, 'input_sha256': input_hash}
    if decode('staging-report.json') != staging:
        raise ValueError('Staging report mismatch')
    probes = [json.loads(line) for line in payload['preflight.log'].decode().splitlines() if line.startswith('{')]
    packages = dict(item.split('==') for item in config['packages'])
    if len(probes) != 1 or probes[0].get('packages') != packages:
        raise ValueError('Package preflight mismatch')
    bootstrap = decode('bootstrap.json')
    if bootstrap.get('status') != 'ok' or bootstrap.get('gpu') != probes[0].get('gpu'):
        raise ValueError('Environment bootstrap mismatch')
    exits, environments, predictions = decode('worker-exits.json'), {}, {}
    by_id = {row['id']: row for row in references}
    for engine in pilot.ENGINES:
        prefix = 'predictions/'+engine
        spec = config['models'][engine]
        if decode(prefix+'/identity.json') != {'engine': engine, 'spec': spec,
                'input_sha256': input_hash, 'runner_sha256': hashes[engine]}:
            raise ValueError('Worker identity mismatch')
        if exits.get(engine) != {'exit_code': 0, 'status': 'finished'}:
            raise ValueError('Worker did not finish successfully; audit as partial evidence instead')
        env = decode(prefix+'/environment.json')
        actual = {key.lower(): value for key, value in env['packages'].items()}
        required = {key: value for key, value in packages.items()
                    if key not in ('sentencepiece',) and (engine == 'qwen3-vl-4b' or key not in ('bitsandbytes', 'accelerate'))}
        if (any(actual.get(key.lower()) != value for key, value in required.items())
                or actual.get('torch') != probes[0].get('torch') or env.get('gpu') != bootstrap['gpu']
                or env.get('reference_text_sent_to_model') is not False or env.get('load_error') is not None):
            raise ValueError('Worker environment mismatch')
        raw = rows(prefix+'/'+pilot.FILES[engine])
        if len(raw) != len(references) or {row['id'] for row in raw} != set(by_id):
            raise ValueError('Incomplete/duplicate teacher coverage')
        for row in raw:
            validate_generation(row, by_id[row['id']], spec, engine)
        environments[engine], predictions[engine] = env, raw
    output.mkdir(parents=True)
    pilot.stage(config, input_bundle, output/'dataset')
    extracted = output/'evidence'
    for name, data in payload.items():
        path = extracted/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    recomputed = pilot.combine(config, output/'dataset', extracted/'predictions', output/'recomputed', runner_hashes=hashes)
    if (recomputed != decode('combined/report.json')
            or read_rows(output/'recomputed/proposals.jsonl') != rows('combined/proposals.jsonl')):
        raise ValueError('Recomputed teacher proposals differ from returned evidence')
    diagnostics = []
    for row in read_rows(output/'recomputed/proposals.jsonl'):
        diagnostics.append({'id': row['id'], 'status': row['status'],
            'source_characters': len(row['source_text']), 'source_glyph_counts': {c: row['source_text'].count(c) for c in '\u017f\u00e1\u0247'},
            'teachers': {engine: {'characters': len(vote['text']), 'glyph_counts': {c: vote['text'].count(c) for c in '\u017f\u00e1\u0247'}}
                         for engine, vote in row['teachers'].items()}})
    write_json(output/'glyph-diagnostics.json', diagnostics)
    report = {'schema': 'slayer-recognizer-data-v3-teacher-audit-v1',
        'source_archive_sha256': digest(archive), 'input_archive_sha256': digest(input_bundle),
        'members_verified': len(names), 'lines': len(references), 'statuses': recomputed['statuses'],
        'source_agreement_proposals': recomputed['source_agreement_proposals'],
        'teacher_health': recomputed['teacher_health'], 'code_provenance': code,
        'worker_environments': environments, 'generation_traces_verified': True,
        'proposals_and_report_recomputed': True, 'remote_GPU_execution_evidence_verified': True,
        'recomputed_proposals_sha256': digest(output/'recomputed/proposals.jsonl'),
        'recomputed_report_sha256': digest(output/'recomputed/report.json'),
        'glyph_counts_are_recall_or_accuracy': False,
        'glyph_counts': {engine: {c: sum(r['text'].count(c) for r in predictions[engine]) for c in '\u017f\u00e1\u0247'} for engine in pilot.ENGINES},
        'training_examples_created': 0, 'gold_labels_created': 0, 'sota_claim': False,
        'next_gate': 'Review crop geometry and diplomatic text, with source-region context. Do not train on count-matched source labels or teacher agreement alone.'}
    write_json(output/'audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'input-bundle', 'config', 'output', 'code-revision'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.input_bundle, args.config, args.output, args.code_revision), ensure_ascii=False, indent=2))
