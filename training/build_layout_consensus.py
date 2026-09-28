"""Build traceable layout consensus and an RF-DETR-compatible COCO dataset.

Teacher proposals are evidence, not ground truth. Only detections supported by
the configured number of distinct teacher IDs are accepted. Everything else is
preserved in a review queue and as a hard-example signal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median


SCHEMA = 'slayer-layout-teacher-proposal-v1'
OUTPUT_SCHEMA = 'slayer-layout-consensus-v1'
TEACHER_ID = re.compile(r'[a-z0-9][a-z0-9._/-]{0,127}\Z')
SCORE_KINDS = {'model-confidence', 'neutral-unavailable'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def containment(first, second):
    """Return intersection over the smaller box area."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    smaller = min(_area(first), _area(second))
    return intersection / smaller if smaller else 0.0


def _required_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Invalid {field}')
    return value


def _sha256(value, field):
    value = _required_string(value, field)
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise ValueError(f'Invalid {field}')
    return value


def _validate_bbox(value, width, height):
    if (not isinstance(value, list) or len(value) != 4 or
            any(not isinstance(item, (int, float)) or not math.isfinite(item) for item in value)):
        raise ValueError('Invalid bbox_xyxy')
    box = [float(item) for item in value]
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError('bbox_xyxy outside image or empty')
    return box


def load_proposals(paths, categories=None):
    """Validate all JSONL inputs before any output directory is created."""
    category_set = set(categories or [])
    pages = {}
    seen_teacher_pages = set()
    teacher_runs = {}
    input_evidence = []
    for path in map(Path, paths):
        rows = 0
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'Invalid JSON at {path}:{line_number}') from error
                if not isinstance(row, dict) or row.get('schema') != SCHEMA:
                    raise ValueError(f'Invalid proposal schema at {path}:{line_number}')
                page_id = _required_string(row.get('page_id'), 'page_id')
                image = row.get('image')
                teacher = row.get('teacher')
                detections = row.get('detections')
                if not isinstance(image, dict) or not isinstance(teacher, dict) or not isinstance(detections, list):
                    raise ValueError(f'Invalid proposal fields at {path}:{line_number}')
                width, height = image.get('width'), image.get('height')
                if (not isinstance(width, int) or isinstance(width, bool) or width <= 0 or
                        not isinstance(height, int) or isinstance(height, bool) or height <= 0):
                    raise ValueError('Invalid image dimensions')
                image_identity = {
                    'file_name': _required_string(image.get('file_name'), 'image.file_name'),
                    'sha256': _sha256(image.get('sha256'), 'image.sha256'),
                    'width': width,
                    'height': height,
                }
                teacher_id = _required_string(teacher.get('id'), 'teacher.id')
                if not TEACHER_ID.fullmatch(teacher_id):
                    raise ValueError('teacher.id must be a lowercase model-family slug')
                teacher_identity = {
                    'id': teacher_id,
                    'revision': _required_string(teacher.get('revision'), 'teacher.revision'),
                    'run_id': _required_string(teacher.get('run_id'), 'teacher.run_id'),
                    'prompt_sha256': _sha256(teacher.get('prompt_sha256'), 'teacher.prompt_sha256'),
                }
                previous_run = teacher_runs.setdefault(teacher_id, teacher_identity)
                if previous_run != teacher_identity:
                    raise ValueError(f'Inconsistent teacher run identity: {teacher_id}')
                key = (page_id, teacher_id)
                if key in seen_teacher_pages:
                    raise ValueError(f'Duplicate teacher vote for page: {page_id}/{teacher_id}')
                seen_teacher_pages.add(key)
                page = pages.setdefault(page_id, {'image': image_identity, 'detections': []})
                if page['image'] != image_identity:
                    raise ValueError(f'Image identity mismatch for page: {page_id}')
                seen_detection_ids = set()
                for detection in detections:
                    if not isinstance(detection, dict):
                        raise ValueError('Invalid detection')
                    detection_id = _required_string(detection.get('id'), 'detection.id')
                    if detection_id in seen_detection_ids:
                        raise ValueError(f'Duplicate detection ID within proposal: {detection_id}')
                    seen_detection_ids.add(detection_id)
                    label = _required_string(detection.get('label'), 'detection.label')
                    if category_set and label not in category_set:
                        raise ValueError(f'Unknown category: {label}')
                    score = detection.get('score')
                    if (not isinstance(score, (int, float)) or isinstance(score, bool) or
                            not math.isfinite(score) or not 0 <= score <= 1):
                        raise ValueError('Detection score must be in [0, 1]')
                    score_kind = detection.get('score_kind', 'model-confidence')
                    if score_kind not in SCORE_KINDS:
                        raise ValueError('Invalid detection score_kind')
                    page['detections'].append({
                        'id': detection_id,
                        'teacher': teacher_identity,
                        'label': label,
                        'bbox_xyxy': _validate_bbox(detection.get('bbox_xyxy'), width, height),
                        'score': float(score),
                        'score_kind': score_kind,
                        'source': {'path': str(path), 'line': line_number},
                    })
        input_evidence.append({'path': str(path), 'sha256': digest(path), 'rows': rows})
    if not pages:
        raise ValueError('No teacher proposals')
    return pages, input_evidence


def _fused_box(detections):
    return [median(detection['bbox_xyxy'][axis] for detection in detections) for axis in range(4)]


def _cluster_page(detections, iou_threshold, min_score):
    eligible = [item for item in detections if item['score'] >= min_score]
    low_score = [item for item in detections if item['score'] < min_score]
    clusters = []
    for detection in sorted(eligible, key=lambda item: (item['label'], item['teacher']['id'], item['id'])):
        matches = []
        for index, cluster in enumerate(clusters):
            if cluster[0]['label'] != detection['label']:
                continue
            if detection['teacher']['id'] in {item['teacher']['id'] for item in cluster}:
                continue
            overlaps = [iou(detection['bbox_xyxy'], item['bbox_xyxy']) for item in cluster]
            if overlaps and min(overlaps) >= iou_threshold:
                matches.append((min(overlaps), index))
        if matches:
            clusters[max(matches, key=lambda item: (item[0], -item[1]))[1]].append(detection)
        else:
            clusters.append([detection])
    clusters.extend([[item] for item in low_score])
    return clusters


def build_consensus(paths, output, *, categories=None, quorum=2,
                    iou_threshold=0.5, conflict_iou=0.5, min_score=0.0,
                    containment_threshold=0.9, granularity_ratio=2.0):
    if quorum < 2:
        raise ValueError('quorum must be at least 2')
    if any(not 0 <= value <= 1 for value in (
            iou_threshold, conflict_iou, min_score, containment_threshold)):
        raise ValueError('thresholds must be in [0, 1]')
    if (not isinstance(granularity_ratio, (int, float)) or
            not math.isfinite(granularity_ratio) or granularity_ratio <= 1):
        raise ValueError('granularity_ratio must be finite and greater than 1')
    if categories is not None and (not categories or len(set(categories)) != len(categories)):
        raise ValueError('categories must be unique and nonempty')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new consensus output directory')
    pages, input_evidence = load_proposals(paths, categories)

    page_results = []
    review_queue = []
    all_labels = set(categories or [])
    for page_id in sorted(pages):
        page = pages[page_id]
        all_labels.update(item['label'] for item in page['detections'])
        clusters = []
        for index, members in enumerate(_cluster_page(page['detections'], iou_threshold, min_score), 1):
            clusters.append({
                'id': f'{page_id}:cluster-{index:04d}',
                'page_id': page_id,
                'label': members[0]['label'],
                'bbox_xyxy': _fused_box(members),
                'teachers': sorted(item['teacher']['id'] for item in members),
                'mean_score': sum(item['score'] for item in members) / len(members),
                'score_kinds': sorted({item['score_kind'] for item in members}),
                'proposals': members,
                'reasons': [],
            })
        for cluster in clusters:
            if len(cluster['teachers']) < quorum:
                cluster['reasons'].append('below-quorum')
            if any(item['score'] < min_score for item in cluster['proposals']):
                cluster['reasons'].append('below-score-threshold')
        for left_index, left in enumerate(clusters):
            for right in clusters[left_index + 1:]:
                if left['label'] != right['label'] and iou(left['bbox_xyxy'], right['bbox_xyxy']) >= conflict_iou:
                    left['reasons'].append('label-conflict')
                    right['reasons'].append('label-conflict')
                if left['label'] != right['label']:
                    continue
                left_area, right_area = _area(left['bbox_xyxy']), _area(right['bbox_xyxy'])
                ratio = max(left_area, right_area) / min(left_area, right_area)
                if (ratio < granularity_ratio or
                        containment(left['bbox_xyxy'], right['bbox_xyxy']) < containment_threshold):
                    continue
                left_votes, right_votes = len(left['teachers']), len(right['teachers'])
                if left_votes <= right_votes:
                    left['reasons'].append('granularity-conflict')
                if right_votes <= left_votes:
                    right['reasons'].append('granularity-conflict')
        accepted = []
        for cluster in clusters:
            cluster['reasons'] = sorted(set(cluster['reasons']))
            if cluster['reasons']:
                review_queue.append(cluster)
            else:
                accepted.append({key: cluster[key] for key in (
                    'id', 'label', 'bbox_xyxy', 'teachers', 'mean_score', 'score_kinds')})
        page_results.append({'schema': OUTPUT_SCHEMA, 'page_id': page_id,
                             'image': page['image'], 'objects': accepted})

    labels = list(categories) if categories else sorted(all_labels)
    category_ids = {label: index + 1 for index, label in enumerate(labels)}
    coco_images, coco_annotations = [], []
    annotation_id = 1
    for image_id, page in enumerate(page_results, 1):
        image = page['image']
        coco_images.append({'id': image_id, 'file_name': image['file_name'],
                            'width': image['width'], 'height': image['height'],
                            'sha256': image['sha256'], 'page_id': page['page_id']})
        for item in page['objects']:
            x1, y1, x2, y2 = item['bbox_xyxy']
            coco_annotations.append({
                'id': annotation_id, 'image_id': image_id,
                'category_id': category_ids[item['label']],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0,
                'consensus_id': item['id'],
            })
            annotation_id += 1
    hard_examples = []
    for page in page_results:
        page_reviews = [item for item in review_queue if item['page_id'] == page['page_id']]
        if page_reviews:
            hard_examples.append({
                'schema': 'slayer-hard-example-v1', 'page_id': page['page_id'],
                'image_sha256': page['image']['sha256'],
                'reasons': sorted({reason for item in page_reviews for reason in item['reasons']}),
                'review_object_ids': [item['id'] for item in page_reviews],
                'source': 'teacher-consensus',
            })

    output.mkdir(parents=True)
    files = {
        'consensus.jsonl': ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in page_results),
        'review-queue.jsonl': ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in review_queue),
        'hard-examples.jsonl': ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in hard_examples),
        'annotations.coco.json': json.dumps({
            'info': {'description': 'SLAYER-OCR layout consensus; accepted teacher agreement only',
                     'schema': OUTPUT_SCHEMA},
            'images': coco_images,
            'annotations': coco_annotations,
            'categories': [{'id': category_ids[label], 'name': label} for label in labels],
        }, ensure_ascii=False, indent=2) + '\n',
    }
    for name, content in files.items():
        (output / name).write_text(content, encoding='utf-8', newline='\n')
    report = {
        'schema': 'slayer-layout-consensus-report-v1',
        'status': 'candidate-not-reviewed',
        'inputs': input_evidence,
        'policy': {'quorum': quorum, 'iou_threshold': iou_threshold,
                   'conflict_iou': conflict_iou, 'min_score': min_score,
                   'containment_threshold': containment_threshold,
                   'granularity_ratio': granularity_ratio,
                   'teacher_identity': 'teacher.id; prompts/runs from one teacher do not add votes'},
        'pages': len(page_results),
        'teacher_rows': sum(item['rows'] for item in input_evidence),
        'teacher_detections': sum(len(page['detections']) for page in pages.values()),
        'accepted_objects': len(coco_annotations),
        'review_objects': len(review_queue),
        'hard_example_pages': len(hard_examples),
        'categories': labels,
        'limitations': [
            'Claimed image SHA-256 values are cross-checked between teachers but image bytes are not copied or re-hashed.',
            'Consensus is weak supervision, not human ground truth or SOTA evidence.',
            'Thresholds must be tuned on development data only; final test data must stay sealed.',
        ],
    }
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8', newline='\n')
    artifacts = [output / name for name in files] + [report_path]
    (output / 'checksums.sha256').write_text(
        ''.join(f'{digest(path)}  {path.name}\n' for path in artifacts),
        encoding='utf-8', newline='\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposals', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', help='JSON containing categories and consensus thresholds')
    args = parser.parse_args()
    config = {}
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        config = config.get('consensus', config)
    report = build_consensus(
        args.proposals, args.output,
        categories=config.get('categories'),
        quorum=config.get('quorum', 2),
        iou_threshold=config.get('iou_threshold', 0.5),
        conflict_iou=config.get('conflict_iou', 0.5),
        min_score=config.get('min_score', 0.0),
        containment_threshold=config.get('containment_threshold', 0.9),
        granularity_ratio=config.get('granularity_ratio', 2.0),
    )
    print(json.dumps({key: report[key] for key in (
        'status', 'pages', 'teacher_rows', 'teacher_detections', 'accepted_objects',
        'review_objects', 'hard_example_pages')}, indent=2))


if __name__ == '__main__':
    main()
