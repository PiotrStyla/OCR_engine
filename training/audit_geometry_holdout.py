"""Verify a geometry holdout evidence ZIP and quantify paired uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import zipfile

from training.geometry_holdout_runner import normalize, score, VARIANTS


def digest(data):
    return hashlib.sha256(data).hexdigest()


def percentile(sorted_values, probability):
    return sorted_values[min(len(sorted_values) - 1, int(probability * len(sorted_values)))]


def paired_bootstrap(rows, rectangle_edits, band_edits, samples=20_000, seed=20260924):
    ids = [r['id'] for r in rows]
    characters = {r['id']: len(normalize(r['text'])) for r in rows}
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        selected = [rng.choice(ids) for _ in ids]
        deltas.append(sum(band_edits[i] - rectangle_edits[i] for i in selected) /
                      sum(characters[i] for i in selected))
    deltas.sort()
    observed = sum(band_edits.values()) / sum(characters.values()) - (
        sum(rectangle_edits.values()) / sum(characters.values()))
    return {'unit': 'region', 'samples': samples, 'seed': seed,
            'cer_delta_line_band_minus_rectangle': observed,
            'percentile_95_interval': [percentile(deltas, .025), percentile(deltas, .975)],
            'bootstrap_fraction_line_band_better': sum(x < 0 for x in deltas) / samples}


def audit(evidence_path, frozen_manifest_path):
    evidence_path, frozen_manifest_path = Path(evidence_path), Path(frozen_manifest_path)
    with zipfile.ZipFile(evidence_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        checksums = json.loads(archive.read('checksums.json'))
        if set(names) != set(checksums) | {'checksums.json'}:
            raise ValueError('Unlisted ZIP payload')
        if any(digest(archive.read(name)) != value for name, value in checksums.items()):
            raise ValueError('Evidence checksum mismatch')
        manifest = json.loads(archive.read('input-manifest.json'))
        report = json.loads(archive.read('report.json'))
        provenance = json.loads(archive.read('provenance.json'))
        predictions = {model: json.loads(archive.read(model.replace('/', '--') + '.json'))
                       for model in report['models']}
    frozen_data = frozen_manifest_path.read_bytes()
    if digest(frozen_data) != provenance['manifest_sha256']:
        raise ValueError('Frozen manifest checksum mismatch')
    if json.loads(frozen_data) != manifest:
        raise ValueError('Evidence manifest differs from pre-OCR freeze')
    rows = manifest['regions']
    results = {}
    for model, by_variant in predictions.items():
        recomputed = {variant: score(rows, by_variant[variant]) for variant in VARIANTS}
        if any(recomputed[v] != report['results'][model][v] for v in VARIANTS):
            raise ValueError('Stored metrics mismatch: ' + model)
        if any(p['status'] != 'ok' or not p['text'].strip()
               for variant in VARIANTS for p in by_variant[variant]):
            raise ValueError('Incomplete inference: ' + model)
        truncated = {variant: sum(line['possibly_truncated'] for p in by_variant[variant]
                                  for line in p['lines']) for variant in VARIANTS}
        bootstrap = paired_bootstrap(rows,
                                     recomputed['rectangle']['per_region_character_edits'],
                                     recomputed['line_band']['per_region_character_edits'])
        results[model] = {'metrics': recomputed,
                          'comparison': report['results'][model]['comparison'],
                          'possibly_truncated_lines': truncated,
                          'paired_bootstrap': bootstrap}
    return {'scope': report['scope'], 'evidence_zip_sha256': digest(evidence_path.read_bytes()),
            'manifest_sha256': provenance['manifest_sha256'], 'environment': report['environment'],
            'regions': len(rows), 'collections': len({r['collection'] for r in rows}),
            'reference_lines': sum(len(r['text'].splitlines()) for r in rows),
            'reference_private_use_characters': sum(r['reference_private_use_count'] for r in rows),
            'selection_frozen_before_ocr': provenance['selection_frozen_before_ocr'],
            'all_regions_retained': provenance['all_regions_retained'], 'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = audit(args.evidence, args.manifest)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with Path(args.output).open('x', encoding='utf-8') as stream:
            stream.write(rendered + '\n')
    print(rendered)
