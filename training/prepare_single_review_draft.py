"""Materialize one reviewer's proposals without promoting them to gold labels."""
from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
import shutil
import unicodedata
import zipfile

from training.adjudicate_reviews import reviewer_key, text_hash, validate
from training.build_annotation_review import issues
from training.full_page_pilot import digest, read_rows, safe_relative, write_json, write_rows


def glyph_proposals(events):
    observed = []
    for event in events:
        for tag, a, b, c, d in SequenceMatcher(None,event['before'],event['after'],autojunk=False).get_opcodes():
            before, after = event['before'][a:b], event['after'][c:d]
            if (tag == 'replace' and len(before) == 1 and unicodedata.category(before) == 'Co'
                    and 1 <= len(after) <= 4 and all(char.isalpha() for char in after)):
                observed.append({'codepoint':f'U+{ord(before):04X}', 'replacement':after,
                                 'page_id':event['page_id'], 'event_id':event['id']})
    counts = Counter((row['codepoint'],row['replacement']) for row in observed)
    return {'schema':'polocrbench-glyph-review-proposals-v1',
            'status':'observed-single-review-proposals-not-global-mapping',
            'automatic_application_allowed':False,
            'mappings':[{'codepoint':key[0],'replacement':key[1],'observations':count}
                        for key,count in sorted(counts.items())], 'evidence':observed}


def prepare(manifest, review, output, predictions=None):
    manifest, review, output = map(Path,(manifest,review,output))
    if output.exists():
        raise FileExistsError('Use a new draft directory')
    rows = read_rows(manifest)
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Expected unique nonempty source pages')
    packet = json.loads(review.read_text(encoding='utf-8-sig'))
    events = validate(packet,rows,digest(manifest))
    if not events or len({reviewer_key(event['reviewer']) for event in events}) != 1:
        raise ValueError('A single-review draft requires exactly one reviewer with events')
    latest = {event['page_id']:event for event in events}
    for row in rows:
        source = manifest.parent/safe_relative(row['image'])
        if digest(source) != row['sha256']:
            raise ValueError('Source image checksum mismatch')
        if row.get('pagexml_local'):
            xml = manifest.parent/safe_relative(row['pagexml_local'])
            if xml.suffix != '.xml' or digest(xml) != row['pagexml_sha256']:
                raise ValueError('Source PAGE XML checksum mismatch')
    if predictions is not None:
        supplied = read_rows(predictions)
        if (len(supplied) != len({row['id'] for row in supplied})
                or {row['id'] for row in supplied} != {row['id'] for row in rows}):
            raise ValueError('Retain every prediction, including failed outputs')
    output.mkdir(parents=True)
    (output/'images').mkdir()
    candidates, changes = [], []
    for index,row in enumerate(rows):
        event = latest.get(row['id'])
        text = event['after'] if event else row['text']
        image = f'images/{index:04d}{Path(row["image"]).suffix}'
        shutil.copyfile(manifest.parent/row['image'],output/image)
        candidate = {**row, 'image':image, 'text':text,
            'source_reference_text':row['text'], 'source_reference_text_sha256':text_hash(row['text']),
            'reference_status':'single-review-draft-not-gold', 'completeness_verified':False,
            'eligible_for_evaluation':False, 'eligible_for_training':False,
            'review_status':event['decision'] if event else 'unreviewed',
            'review_event_id':event['id'] if event else None,
            'reviewer':event['reviewer'] if event else None,
            'annotation_notes':event['note'] if event else '',
            'content_scope_status':'not-adjudicated'}
        if row.get('pagexml_local'):
            target_xml = output/'pagexml'/f'{index:04d}.xml'
            target_xml.parent.mkdir(exist_ok=True)
            shutil.copyfile(manifest.parent/row['pagexml_local'],target_xml)
            candidate.update(pagexml_local=target_xml.relative_to(output).as_posix(),
                             pagexml_reference_status='source-unreviewed-unchanged')
        candidates.append(candidate)
        changes.append({'id':row['id'], 'changed':text != row['text'],
            'decision':event['decision'] if event else 'unreviewed',
            'event_id':event['id'] if event else None,
            'source_text_sha256':text_hash(row['text']), 'draft_text_sha256':text_hash(text),
            'source_unicode_issues':len(issues(row['text'])), 'draft_unicode_issues':len(issues(text)),
            'notes_present':bool(event and event['note'].strip())})
    target = output/'manifest.jsonl'
    write_rows(target,candidates)
    shutil.copyfile(manifest,output/'original-manifest.jsonl')
    shutil.copyfile(review,output/'original-review.json')
    write_json(output/'glyph-proposals.json',glyph_proposals(events))
    report = {'schema':'polocrbench-single-review-draft-v1', 'status':'single-review-draft-not-gold',
        'source_manifest_sha256':digest(manifest), 'review_sha256':digest(review),
        'draft_manifest_sha256':digest(target), 'pages':len(rows), 'events':len(events),
        'reviewed_pages':len(latest), 'changed_pages':sum(row['changed'] for row in changes),
        'decisions':dict(Counter(row['decision'] for row in changes)),
        'source_unicode_issues':sum(row['source_unicode_issues'] for row in changes),
        'draft_unicode_issues':sum(row['draft_unicode_issues'] for row in changes),
        'gold_pages':0, 'sota_claim':False, 'model_promotion':False,
        'policy':'Use after verbatim, preserve original text and notes separately. No verified events synthesized. No global glyph replacement or spelling modernization. One reviewer is not independent adjudication.',
        'results':changes}
    if predictions is not None:
        from training.benchmark_pages import evaluate
        source_report = evaluate(manifest,predictions)
        shutil.copyfile(predictions,output/'retained-projected-predictions.jsonl')
        draft_report = evaluate(target,output/'retained-projected-predictions.jsonl')
        comparison = {'schema':'polocrbench-reference-intervention-diagnostic-v1',
            'predictions_unchanged':digest(predictions)==digest(output/'retained-projected-predictions.jsonl'),
            'source':source_report, 'draft':draft_report,
            'reference_status':'single-review-draft-not-gold', 'content_scope_verified':False,
            'generation_health_unchanged':{
                'runtime_errors':sum(row['status']!='ok' for row in supplied),
                'token_limit_pages':sum(bool(row.get('token_limit_reached')) for row in supplied),
                'eos_pages':sum(row.get('finish_reason')=='eos' for row in supplied)},
            'model_improved':False, 'sota_claim':False,
            'claim_boundary':'Same saved predictions; score changes come only from draft reference edits, including possible content-scope changes. No independent OCR accuracy claim.'}
        write_json(output/'diagnostic-comparison.json',comparison)
        report['diagnostic_cer_micro'] = {'source':source_report['cer_micro'],'draft':draft_report['cer_micro']}
        report['diagnostic_wer_micro'] = {'source':source_report['wer_micro'],'draft':draft_report['wer_micro']}
    write_json(output/'draft-report.json',report)
    files = [path for path in output.rglob('*') if path.is_file()]
    write_json(output/'checksums.json',{path.relative_to(output).as_posix():digest(path) for path in files})
    archive = output/'full-page-validation-v4-single-review-draft.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as stream:
        for path in files+[output/'checksums.json']:
            stream.write(path,path.relative_to(output).as_posix())
    return {**report, 'archive':str(archive), 'archive_sha256':digest(archive)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--review',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--predictions')
    args = parser.parse_args()
    result = prepare(args.manifest,args.review,args.output,args.predictions)
    print(json.dumps({key:value for key,value in result.items() if key!='results'},indent=2))
