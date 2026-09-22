"""PolOCRBench composite score over submitted subtasks.

Deterministic, no LLM judges, stdlib only. Schema ``polocrbench-composite-v1``.

Normalized subtask scores: A = 1 - CER_micro (negative when CER > 1),
B = TEDS mean, C = field-level F1 micro. The composite is the arithmetic mean
over the subtasks actually submitted (participants may enter any subset).
Reports must come from ``training.transcription_eval``, ``training.table_eval``
or ``training.kie_eval``; protocol versions are checked by prefix.

When report entries carry ``elapsed_seconds``, the output includes inference
timing per record (cost-per-page reporting for baselines).

Usage: python -m training.composite_score --report-a A.json --report-b B.json --output C.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOL_VERSION = 'polocrbench-composite-v1'
_SPECS = {'A': ('polocrbench-transcription-', 'pages'),
          'B': ('polocrbench-table-teds-', 'tables'),
          'C': ('polocrbench-kie-', 'documents')}
_METRICS = {'A': 'cer_micro', 'B': 'teds_mean', 'C': 'f1_micro'}


def subtask_score(subtask, report):
    if subtask == 'A':
        return 1.0 - report['cer_micro']
    return report[_METRICS[subtask]]


def composite(reports):
    """Arithmetic mean of (1 - CER_micro, TEDS_mean, F1_micro) over submitted subtasks."""
    if not reports or set(reports) - set(_SPECS):
        raise ValueError('Reports must cover at least one of subtasks A, B, C')
    scores, components, timing = {}, {}, {}
    for subtask in sorted(reports):
        report = reports[subtask]
        prefix, unit = _SPECS[subtask]
        version = str(report.get('protocol_version', ''))
        if not version.startswith(prefix):
            raise ValueError(f'Wrong protocol version for subtask {subtask}: {version}')
        scores[subtask] = subtask_score(subtask, report)
        components[subtask] = {'protocol_version': version,
                               'records': report[unit],
                               'errors_or_missing': report['errors_or_missing'],
                               'metric': _METRICS[subtask],
                               'score': scores[subtask]}
        timings = [entry['elapsed_seconds'] for entry in report.get('results', [])
                   if isinstance(entry.get('elapsed_seconds'), (int, float))
                   and not isinstance(entry.get('elapsed_seconds'), bool)]
        if timings:
            timing[subtask] = {'records': report[unit], 'timed_records': len(timings),
                               'mean_elapsed_seconds': sum(timings) / len(timings),
                               'max_elapsed_seconds': max(timings)}
    result = {'schema': PROTOCOL_VERSION,
              'subtasks': sorted(reports),
              'scores': scores,
              'composite': sum(scores.values()) / len(scores),
              'formula': 'mean over submitted subtasks of (1 - CER_micro, TEDS_mean, F1_micro)',
              'components': components}
    if timing:
        result['timing'] = timing
    return result


def load_report(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for subtask in 'ABC':
        parser.add_argument(f'--report-{subtask.lower()}')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reports = {}
    for subtask in 'ABC':
        path = getattr(args, f'report_{subtask.lower()}')
        if path:
            reports[subtask] = load_report(path)
    result = composite(reports)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('subtasks', 'scores', 'composite')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
