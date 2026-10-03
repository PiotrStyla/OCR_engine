"""Prepare source-preserving review inputs after a verified full-page pilot."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import unicodedata
import zipfile

from training.build_annotation_review import build as build_review
from training.full_page_pilot import digest, markdown_text, read_rows, safe_relative, write_json, write_rows
from training.benchmark_pages import normalize


def summarize(rows, predictions, reference_audit):
    from jiwer import cer, wer
    if (len({row['id'] for row in rows}) != len(rows)
            or len({row['id'] for row in predictions}) != len(predictions)
            or {row['id'] for row in rows} != {row['id'] for row in predictions}):
        raise ValueError('Review requires every prediction, including failures')
    by_id = {row['id']:row for row in predictions}
    references = {row['id']:row for row in reference_audit['results']}
    results = []
    for row in rows:
        prediction = by_id[row['id']]
        flags = list(references[row['id']]['flags'])
        if prediction['status'] != 'ok':
            flags.append('inference-error')
        if prediction.get('token_limit_reached'):
            flags.append('token-limit')
        if prediction.get('finish_reason') != 'eos':
            flags.append('non-eos-output')
        if '<img ' in prediction['text']:
            flags.append('generated-image-placeholder')
        if any('CYRILLIC' in unicodedata.name(char, '') for char in prediction['text']):
            flags.append('cyrillic-output-review')
        hypothesis = (markdown_text(prediction['text']) if prediction.get('format') == 'markdown'
                      else prediction['text']) if prediction['status']=='ok' else ''
        results.append({'id':row['id'], 'collection':row['collection'], 'flags':flags,
                        'status':prediction['status'], 'finish_reason':prediction.get('finish_reason'),
                        'reference':normalize(row['text']), 'hypothesis':normalize(hypothesis),
                        'cer':cer(normalize(row['text']),normalize(hypothesis)),
                        'wer':wer(normalize(row['text']),normalize(hypothesis)),
                        'generated_tokens':prediction.get('generated_tokens'),
                        'reference_status':'source-unreviewed'})
    partitions = {}
    for name, selected in [('all_pages',results), ('eos_only_diagnostic',
            [row for row in results if row['status']=='ok' and row['finish_reason']=='eos'])]:
        if selected:
            partitions[name] = {'pages':len(selected),
                'cer_micro':cer([row['reference'] for row in selected],[row['hypothesis'] for row in selected]),
                'wer_micro':wer([row['reference'] for row in selected],[row['hypothesis'] for row in selected]),
                'selection_biased':name!='all_pages'}
    results.sort(key=lambda row:(not ('token-limit' in row['flags']),
        not ('reading-order-incomplete' in row['flags']), -row['cer'], row['id']))
    return {'schema':'slayer-full-page-triage-v1', 'pages':len(rows), 'gold_pages':0,
            'reference_status':'source-unreviewed', 'sota_claim':False, 'model_promotion':False,
            'partitions':partitions, 'flag_counts':dict(Counter(flag for row in results for flag in row['flags'])),
            'claim_boundary':'EOS-only is a selected diagnostic, never the primary score. Flags require visual review. No raw outputs or source text are corrected.',
            'results':results}


def prepare(audited, image_pool, output):
    from PIL import Image
    audited, image_pool, output = map(Path, (audited,image_pool,output))
    if output.exists():
        raise FileExistsError('Use a new review directory')
    audit = json.loads((audited/'audit.json').read_text(encoding='utf-8'))
    for name, expected in json.loads((audited/'checksums.json').read_text()).items():
        if digest(audited/safe_relative(name)) != expected:
            raise ValueError('Audited evidence changed')
    evidence = audited/'evidence'
    rows = read_rows(evidence/'dataset/manifest.jsonl')
    predictions = read_rows(evidence/'predictions/ovis-ocr2/ovis-ocr2-full-page.jsonl')
    reference_audit = json.loads((evidence/'dataset/reference-audit.json').read_text())
    triage = summarize(rows,predictions,reference_audit)
    originals = {row['id']:row for row in rows}
    local = {row['id']:row for row in read_rows(image_pool/'manifest.jsonl')}
    output.mkdir(parents=True)
    (output/'images').mkdir()
    (output/'pagexml').mkdir()
    staged = []
    for index, item in enumerate(triage['results']):
        original, cached = originals[item['id']], local[item['id']]
        if any(original[key] != cached[key] for key in ('text','image_sha256','pagexml_sha256','width','height')):
            raise ValueError('Review image/source identity mismatch')
        png = image_pool/safe_relative(cached['image'])
        jpeg = image_pool/'sources'/f'{png.stem}.jpg'
        xml = image_pool/safe_relative(cached['pagexml_local'])
        if (digest(png)!=cached['sha256'] or digest(jpeg)!=original['image_sha256']
                or digest(xml)!=original['pagexml_sha256']):
            raise ValueError('Cached review image/XML checksum mismatch')
        with Image.open(png) as image, Image.open(jpeg) as source:
            if (image.size != (original['width'],original['height']) or image.mode != source.mode
                    or image.tobytes() != source.tobytes()):
                raise ValueError('Review PNG differs from source JPEG decoded pixels')
        destination = output/'images'/f'{index:04d}.png'
        shutil.copyfile(png,destination)
        target_xml = output/'pagexml'/f'{index:04d}.xml'
        shutil.copyfile(xml,target_xml)
        staged.append({**original, 'inference_png_sha256':original['sha256'],
                       'image':destination.relative_to(output).as_posix(), 'sha256':digest(destination),
                       'pagexml_local':target_xml.relative_to(output).as_posix(),
                       'review_status':'pending', 'completeness_verified':False})
    write_rows(output/'manifest.jsonl',staged)
    write_rows(output/'raw-predictions.jsonl',predictions)
    write_json(output/'triage.json',triage)
    write_json(output/'reference-audit.json',reference_audit)
    write_json(output/'lineage.json',{'source_archive_sha256':audit['source_archive_sha256'],
        'inference_manifest_sha256':digest(evidence/'dataset/manifest.jsonl'),
        'source_dataset':json.loads((evidence/'config.json').read_text())['dataset'],
        'review_png_policy':'Native-size source JPEG to PNG; decoded pixels checked. Encoded SHA may differ from Colab; both hashes retained.',
        'reference_text_unchanged':True, 'predictions_unchanged':True, 'gold_pages':0,
        'code_files_in_dataset_zip':False, 'review_builder_sha256':digest(Path(__file__))})
    dataset_files = [path for path in output.rglob('*') if path.is_file()]
    write_json(output/'checksums.json',{path.relative_to(output).as_posix():digest(path) for path in dataset_files})
    archive = output/'full-page-validation-v4-review-inputs.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as stream:
        for path in dataset_files+[output/'checksums.json']:
            stream.write(path,path.relative_to(output).as_posix())
    review = build_review(output/'manifest.jsonl',output/'review')
    return {'pages':len(staged),'gold_pages':0,'review':review,'archive':str(archive),
            'archive_sha256':digest(archive),'partitions':triage['partitions']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audited',required=True)
    parser.add_argument('--image-pool',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.audited,args.image_pool,args.output),indent=2))
