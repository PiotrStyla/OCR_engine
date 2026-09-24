"""Experimental image-only component grouping for single-column paragraph crops."""
from statistics import median


def detect_lines(image):
    """Return proposed boxes without accepting reference text, counts or line IDs."""
    import cv2
    import numpy as np

    gray = np.asarray(image.convert('L'))
    height, width = gray.shape
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = [tuple(map(int, s)) for s in stats[1:]]
    candidates = [h for x, y, w, h, area in components
                  if 8 <= h <= height * .45 and 2 <= w <= width * .08 and area >= 16]
    if not candidates:
        return {'boxes': [], 'foreign_ink_fraction': [], 'character_height': None, 'rejected_edge_groups': 0}
    ch = float(np.percentile(candidates, 65))
    anchors = [c for c in components if .5 * ch <= c[3] <= 1.8 * ch
               and c[4] >= .035 * ch * ch and c[2] <= width * .35]
    groups = []
    for c in sorted(anchors, key=lambda c: (c[0], c[1])):
        cy = c[1] + c[3] / 2
        cx = c[0] + c[2] / 2
        choices = []
        for i, group in enumerate(groups):
            xs = [p[0] + p[2] / 2 for p in group['parts']]
            if len(xs) >= 4 and max(xs) - min(xs) >= 3 * ch:
                slope, _ = np.polyfit(xs, group['centers'], 1)
                predicted = float(np.clip(slope, -.2, .2)) * (cx - median(xs)) + median(group['centers'])
            else:
                predicted = median(group['centers'])
            choices.append((abs(cy - predicted), i))
        if choices and min(choices)[0] <= .65 * ch:
            group = groups[min(choices)[1]]
            group['parts'].append(c)
            group['centers'].append(cy)
        else:
            groups.append({'parts': [c], 'centers': [cy]})

    def bounds(parts):
        return [min(c[0] for c in parts), min(c[1] for c in parts),
                max(c[0] + c[2] for c in parts), max(c[1] + c[3] for c in parts)]

    kept, rejected = [], 0
    for group in groups:
        parts = group['parts']
        box = bounds(parts)
        if len(parts) < 2 or box[2] - box[0] < 1.5 * ch:
            continue
        touches = sum(c[1] <= 1 or c[1] + c[3] >= height - 1 for c in parts)
        if touches / len(parts) >= .5:
            rejected += 1
            continue
        kept.append({'parts': parts[:], 'anchor_box': box})
    anchor_set = set(anchors)
    for c in components:
        x, y, w, h, area = c
        if c in anchor_set or area < max(2, .0015 * ch * ch) or h > .65 * ch:
            continue
        if y <= 0 or y + h >= height:
            continue
        choices = []
        for i, group in enumerate(kept):
            a, b, d, e = group['anchor_box']
            if x + w < a - .25 * ch or x > d + .25 * ch:
                continue
            gap = max(b - (y + h), y - e, 0)
            limit = .2 * ch if y >= e else .55 * ch
            if gap <= limit:
                distance = abs(y + h / 2 - (b + e) / 2)
                choices.append((gap, distance, i))
        if choices:
            kept[min(choices)[2]]['parts'].append(c)
    padding = max(2, round(.1 * ch))
    proposals = []
    component_labels = {c: i + 1 for i, c in enumerate(components)}
    for group in kept:
        x1, y1, x2, y2 = bounds(group['parts'])
        box = [max(0, x1 - padding), max(0, y1 - padding),
               min(width, x2 + padding), min(height, y2 + padding)]
        a, b, d, e = box
        roi = labels[b:e, a:d]
        ink = int(np.count_nonzero(roi))
        own = int(np.count_nonzero(np.isin(roi, [component_labels[c] for c in group['parts']])))
        proposals.append((box, (ink - own) / ink if ink else 1.0))
    proposals.sort(key=lambda p: ((p[0][1] + p[0][3]) / 2, p[0][0]))
    return {'boxes': [p[0] for p in proposals], 'foreign_ink_fraction': [p[1] for p in proposals],
            'character_height': ch, 'rejected_edge_groups': rejected}
