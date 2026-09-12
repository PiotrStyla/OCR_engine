"""Offline scoring of full-page predictions, including missing/error outputs.

Predictions JSONL: {id, text, status: ok|error, elapsed_seconds?}.
This evaluator never loads a model or calls a service.
"""
import argparse
import hashlib
import json
import unicodedata
from pathlib import Path


def normalize(text):
    return " ".join(unicodedata.normalize("NFC",text).split())


def evaluate(manifest, predictions):
    from jiwer import cer, wer
    manifest=Path(manifest)
    records=[json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError("Manifest must contain unique nonempty records")
    rows=[json.loads(line) for line in Path(predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError("Duplicate predictions")
    supplied={r['id']:r for r in rows}
    if set(supplied)-{r['id'] for r in records}:
        raise ValueError("Predictions contain unknown IDs")
    references=[]
    hypotheses=[]
    output=[]
    for row in records:
        data=(manifest.parent/row['image']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f"Image checksum mismatch: {row['id']}")
        reference=normalize(row['text'])
        prediction=supplied.get(row['id'])
        if prediction and prediction.get('status') not in ('ok','error'):
            raise ValueError("Prediction status must be ok or error")
        status=prediction['status'] if prediction else 'missing'
        hypothesis=normalize(prediction['text']) if status=='ok' else ''
        references.append(reference)
        hypotheses.append(hypothesis)
        output.append({'id':row['id'],'status':status,'cer':cer(reference,hypothesis),
                       'wer':wer(reference,hypothesis),'exact_match':reference==hypothesis})
    return {'pages':len(records),'errors_or_missing':sum(r['status']!='ok' for r in output),
            'cer_micro':cer(references,hypotheses),'wer_micro':wer(references,hypotheses),
            'normalization':'Unicode NFC + whitespace only; case and diacritics preserved',
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'predictions_sha256':hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
            'results':output}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=evaluate(args.manifest,args.predictions)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
