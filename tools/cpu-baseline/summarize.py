"""Compute separate page/crop metrics from a completed local baseline run."""
import json
from pathlib import Path
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.probe_correction import distance


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def main():
    directory = Path(sys.argv[1])
    rows = [json.loads(line) for line in (directory/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    result = {'normalization': 'NFC + whitespace collapse; case and diacritics retained'}
    for kind in ('page', 'line'):
        group = [row for row in rows if row['kind'] == kind]
        refs = [normalize(row['reference']) for row in group]
        hyps = [normalize(row['text']) if row['status']=='ok' else '' for row in group]
        chars = sum(map(len,refs))
        words = sum(len(ref.split()) for ref in refs)
        edits = sum(distance(ref,hyp) for ref,hyp in zip(refs,hyps))
        word_edits = sum(distance(ref.split(),hyp.split()) for ref,hyp in zip(refs,hyps))
        result[kind] = {'samples':len(group),'exact_matches':sum(r==h for r,h in zip(refs,hyps)),
            'character_edits':edits,'reference_chars':chars,'cer':edits/chars,
            'word_edits':word_edits,'reference_words':words,'wer':word_edits/words,
            'inference_seconds':sum(row['elapsed_seconds'] for row in group)}
    (directory/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
