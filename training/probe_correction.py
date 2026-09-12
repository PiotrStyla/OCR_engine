"""Bounded sequential correction experiment. No models or images are uploaded.

Defaults to dry-run; --execute uses FABRYKA_API_KEY from the environment.
Errors are recorded by type/status only, without response bodies or credentials.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nxt = [i]
        for j, cb in enumerate(b, 1):
            nxt.append(min(nxt[-1]+1, row[j]+1, row[j-1]+(ca != cb)))
        row = nxt
    return row[-1]


def production_prompt():
    tree = ast.parse((ROOT/'ocr/postprocess.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_SYSTEM_PROMPT' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Production prompt not found')


def score(record):
    raw, ref, corrected = record['input'], record['reference'], record.get('output')
    accepted = record['status']=='ok' and isinstance(corrected,str) and bool(corrected.strip())
    if accepted and len(corrected.strip().split('\n')) != len(raw.split('\n')):
        accepted = False
    effective = corrected.strip() if accepted else raw
    return {'accepted':accepted, 'raw_edits':distance(ref,raw),
            'effective_edits':distance(ref,effective),'reference_chars':len(ref),
            'changed_clean':raw==ref and effective!=ref,
            'raw_output_edits':distance(ref,corrected) if isinstance(corrected,str) else None}


def summarize(records):
    result={}
    for model in dict.fromkeys(r['requested_model'] for r in records):
        rows=[r for r in records if r['requested_model']==model]
        measures=[score(r) for r in rows]
        chars=sum(s['reference_chars'] for s in measures)
        result[model]={'requests':len(rows),'completed':sum(r['status']=='ok' for r in rows),
            'baseline_cer':sum(s['raw_edits'] for s in measures)/chars,
            'effective_cer':sum(s['effective_edits'] for s in measures)/chars,
            'clean_controls':sum(r['category']=='preserve_clean' for r in rows),
            'clean_controls_changed':sum(s['changed_clean'] for s in measures),
            'tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in rows),
            'cases':[{'id':r['id'],**s} for r,s in zip(rows,measures)]}
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    path=ROOT/'benchmarks/correction-v1/cases.jsonl'
    cases=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    models=['bielik-11b-v3','qwen3.8-27b']
    prompt=production_prompt()
    meta={'time_utc':datetime.now(timezone.utc).isoformat(),
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'cases_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'system_prompt':prompt,'endpoint':'https://fabryka.ai/v1/chat/completions',
          'models':models,'max_tokens_per_request':256,'requests_planned':len(cases)*len(models),
          'scope':'Constructed correction diagnostic, not an OCR benchmark or SOTA evaluation.',
          'temperature':0,'retries':0,'timeout_seconds':45}
    if not args.execute:
        print(json.dumps(meta,ensure_ascii=False,indent=2))
        return
    key=os.environ.get('FABRYKA_API_KEY')
    if not key:
        raise SystemExit('Set FABRYKA_API_KEY in process environment')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    opener=urllib.request.build_opener(NoRedirect())
    records=[]
    stop=False
    for model in models:
        for case in cases:
            request={'model':model,'messages':[{'role':'system','content':prompt},
                        {'role':'user','content':case['input']}],'temperature':0,'max_tokens':256}
            row={**case,'requested_model':model}
            start=time.perf_counter()
            try:
                req=urllib.request.Request(meta['endpoint'],data=json.dumps(request).encode(),
                    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                with opener.open(req,timeout=45) as response:
                    obj=json.load(response)
                choice=obj['choices'][0]
                row.update(status='ok' if choice['finish_reason']=='stop' else 'incomplete',
                    output=choice['message'].get('content'),finish_reason=choice['finish_reason'],
                    returned_model=obj.get('model'),usage=obj.get('usage'),
                    system_fingerprint=obj.get('system_fingerprint'),response_id=obj.get('id'))
            except Exception as exc:
                row.update(status='error',error_type=type(exc).__name__)
                if isinstance(exc,urllib.error.HTTPError):
                    row['http_status']=exc.code
                    stop = exc.code in (401,402,403,429)
            row['elapsed_seconds']=round(time.perf_counter()-start,3)
            records.append(row)
            with (args.output/'responses.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(f"{len(records)}/{meta['requests_planned']} {model} {case['id']}: {row['status']}",flush=True)
            if stop:
                break
        if stop:
            break
    report={'complete':len(records)==meta['requests_planned'],'models':summarize(records)}
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
