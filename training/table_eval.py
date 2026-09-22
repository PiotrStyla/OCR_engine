"""PolOCRBench subtask B metric: table extraction to HTML (TEDS).

Deterministic, no LLM judges, stdlib only. Protocol ``polocrbench-table-teds-v1``.

Canonicalization applied to reference and hypothesis before comparison:

- only ``table``, ``tr``, ``td``, ``th`` are structure; ``thead``/``tbody``/
  ``tfoot`` wrappers are dropped and their rows promoted, so ``<table><tr>``
  equals ``<table><tbody><tr>``;
- any other element is transparent (its text stays in the enclosing cell);
  ``<br>`` contributes one space; text outside cells is ignored;
- attributes other than ``rowspan``/``colspan`` are ignored, spans default to 1;
- cell text is normalized like subtask A (Unicode NFC, typographic quotes ->
  straight, whitespace collapse), case and diacritics preserved.

TEDS = 1 - TED / max(nodes_ref, nodes_hyp) over the ordered tree edit distance
(Zhang-Shasha) with unit insert/delete and update cost: 0 for identical labels,
1 for different tag/span labels, and for leaf cells with matching spans
1 - levenshtein(cell texts) / max(len). A side without any recognized element
scores 0.0 unless both sides have none (then 1.0). TEDS-struct ignores cell
text. Aggregation is a macro mean over all manifest tables; missing predictions
and explicit errors score zero and stay in the denominator.

Manifest JSONL: {id, image, sha256, html, table_index?}.
Predictions JSONL: {id, status: ok|error, html: str, elapsed_seconds?}.
Usage: python -m training.table_eval --manifest M.jsonl --predictions P.jsonl --output R.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

from training.transcription_eval import normalize
from training.validate_submission import load_jsonl

PROTOCOL_VERSION = 'polocrbench-table-teds-v1'
ROOT_TAG = 'polocrbench-tables'
_CELL_TAGS = frozenset({'td', 'th'})
_TREE_TAGS = ('table', 'tr', 'td', 'th')


class _Node:
    __slots__ = ('tag', 'rowspan', 'colspan', 'parts', 'children', 'text')

    def __init__(self, tag, rowspan=1, colspan=1):
        self.tag = tag
        self.rowspan = rowspan
        self.colspan = colspan
        self.parts = []
        self.children = []
        self.text = ''


def _span(value):
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 1


class _TableParser(HTMLParser):
    """Tolerant HTML -> canonical node forest of kept tags."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.forest = []
        self._stack = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == 'br':
            self._write(' ')
        if tag in _TREE_TAGS:
            node = _Node(tag)
            if tag in _CELL_TAGS:
                values = dict(attrs)
                node.rowspan = _span(values.get('rowspan'))
                node.colspan = _span(values.get('colspan'))
            parent = self._container()
            (parent.children if parent is not None else self.forest).append(node)
            self._stack.append((tag, node))
        else:
            self._stack.append((tag, None))

    def handle_startendtag(self, tag, attrs):
        if tag.lower() == 'br':
            self._write(' ')
            return
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                del self._stack[index:]
                return

    def handle_data(self, data):
        self._write(data)

    def _write(self, text):
        for _, node in reversed(self._stack):
            if node is not None and node.tag in _CELL_TAGS:
                node.parts.append(text)
                return

    def _container(self):
        for _, node in reversed(self._stack):
            if node is not None:
                return node
        return None


def _finalize(node):
    node.text = normalize(''.join(node.parts)) if node.tag in _CELL_TAGS else ''
    node.parts = []
    for child in node.children:
        _finalize(child)


def parse_table_html(html):
    """Canonical tree: virtual root over every top-level recognized element."""
    parser = _TableParser()
    parser.feed(html or '')
    parser.close()
    root = _Node(ROOT_TAG)
    root.children = parser.forest
    _finalize(root)
    return root


def _levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def _char_sim(a, b):
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return max(0.0, 1.0 - _levenshtein(a, b) / max(len(a), len(b)))


def _update_cost(a, b, structure_only=False):
    if a.tag != b.tag:
        return 1.0
    if a.tag in _CELL_TAGS:
        if (a.rowspan, a.colspan) != (b.rowspan, b.colspan):
            return 1.0
        if not a.children and not b.children:
            return 0.0 if structure_only else 1.0 - _char_sim(a.text, b.text)
    return 0.0


class _Tree:
    """Postorder-indexed tree for Zhang-Shasha (nodes numbered 1..size)."""

    __slots__ = ('nodes', 'children', 'lmd', 'keyroots', 'size')

    def __init__(self, root):
        order, stack = [], [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            else:
                stack.append((node, True))
                stack.extend((child, False) for child in reversed(node.children))
        index = {id(node): position for position, node in enumerate(order, 1)}
        self.size = len(order)
        self.nodes = [None] + order
        self.children = [[] for _ in range(self.size + 1)]
        for position, node in enumerate(order, 1):
            self.children[position] = [index[id(child)] for child in node.children]
        self.lmd = [0] * (self.size + 1)
        for position in range(1, self.size + 1):
            children = self.children[position]
            self.lmd[position] = self.lmd[children[0]] if children else position
        keyroots = {}
        for position in range(1, self.size + 1):
            keyroots[self.lmd[position]] = position
        self.keyroots = sorted(keyroots.values())


def tree_edit_distance(root_a, root_b, update_cost):
    """Zhang-Shasha ordered tree edit distance; unit insert/delete."""
    tree_a, tree_b = _Tree(root_a), _Tree(root_b)
    n, m = tree_a.size, tree_b.size
    if n == 0 or m == 0:
        return float(n + m)
    forest = [[0.0] * (m + 1) for _ in range(n + 1)]
    treedist = [[0.0] * (m + 1) for _ in range(n + 1)]
    lmd_a, lmd_b = tree_a.lmd, tree_b.lmd
    for i in tree_a.keyroots:
        for j in tree_b.keyroots:
            low_a, low_b = lmd_a[i], lmd_b[j]
            forest[low_a - 1][low_b - 1] = 0.0
            for di in range(low_a, i + 1):
                forest[di][low_b - 1] = forest[di - 1][low_b - 1] + 1.0
            for dj in range(low_b, j + 1):
                forest[low_a - 1][dj] = forest[low_a - 1][dj - 1] + 1.0
            for di in range(low_a, i + 1):
                for dj in range(low_b, j + 1):
                    if lmd_a[di] == low_a and lmd_b[dj] == low_b:
                        best = min(forest[di - 1][dj] + 1.0,
                                   forest[di][dj - 1] + 1.0)
                        best = min(best, forest[di - 1][dj - 1] +
                                   update_cost(tree_a.nodes[di], tree_b.nodes[dj]))
                        forest[di][dj] = best
                        treedist[di][dj] = best
                    else:
                        forest[di][dj] = min(
                            forest[di - 1][dj] + 1.0,
                            forest[di][dj - 1] + 1.0,
                            forest[lmd_a[di] - 1][lmd_b[dj] - 1] + treedist[di][dj])
    return treedist[n][m]


def _tree_size(node):
    size, stack = 0, [node]
    while stack:
        current = stack.pop()
        size += 1
        stack.extend(current.children)
    return size


def _teds_trees(ref, hyp, structure_only):
    if bool(ref.children) != bool(hyp.children):
        return 0.0
    if not ref.children:
        return 1.0

    def update(a, b):
        return _update_cost(a, b, structure_only)

    distance = tree_edit_distance(ref, hyp, update)
    return max(0.0, 1.0 - distance / max(_tree_size(ref), _tree_size(hyp)))


def teds(html_ref, html_hyp, structure_only=False):
    """TEDS in [0, 1]; structure_only ignores cell text (TEDS-struct)."""
    return _teds_trees(parse_table_html(html_ref), parse_table_html(html_hyp),
                       structure_only)


def _score_case(case):
    ref, hyp = parse_table_html(case[0]), parse_table_html(case[1])
    return {'teds': _teds_trees(ref, hyp, False),
            'teds_struct': _teds_trees(ref, hyp, True),
            'nodes': {'ref': _tree_size(ref), 'hyp': _tree_size(hyp)}}


def evaluate(manifest, predictions, jobs=1):
    manifest = Path(manifest)
    records = load_jsonl(manifest)
    if not records or len({row['id'] for row in records}) != len(records):
        raise ValueError('Manifest must contain unique nonempty records')
    rows = load_jsonl(Path(predictions))
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate predictions')
    supplied = {row['id']: row for row in rows}
    if set(supplied) - {row['id'] for row in records}:
        raise ValueError('Predictions contain unknown IDs')
    cases, order = [], []
    for row in records:
        data = (manifest.parent / row['image']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Image checksum mismatch: {row["id"]}')
        if not isinstance(row.get('html'), str):
            raise ValueError(f'Manifest table HTML missing: {row["id"]}')
        prediction = supplied.get(row['id'])
        if prediction and prediction.get('status') not in ('ok', 'error'):
            raise ValueError('Prediction status must be ok or error')
        status = prediction['status'] if prediction else 'missing'
        hypothesis = prediction.get('html') if status == 'ok' else ''
        if not isinstance(hypothesis, str):
            raise ValueError(f'Prediction HTML must be a string: {row["id"]}')
        cases.append((row['html'], hypothesis))
        order.append((row, prediction, status))
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            scores = list(pool.map(_score_case, cases))
    else:
        scores = [_score_case(case) for case in cases]
    output = []
    for (row, prediction, status), score in zip(order, scores):
        entry = {'id': row['id'], 'status': status}
        entry.update(score)
        if status != 'ok':
            # A failed request is not a successful empty-table prediction.
            entry['teds'] = 0.0
            entry['teds_struct'] = 0.0
        if prediction and 'elapsed_seconds' in prediction:
            entry['elapsed_seconds'] = prediction['elapsed_seconds']
        output.append(entry)
    count = len(output)
    return {'protocol_version': PROTOCOL_VERSION,
            'tables': count,
            'errors_or_missing': sum(entry['status'] != 'ok' for entry in output),
            'teds_mean': sum(entry['teds'] for entry in output) / count,
            'teds_struct_mean': sum(entry['teds_struct'] for entry in output) / count,
            'aggregation': 'table macro mean; error/missing tables score zero',
            'normalization': 'NFC + typographic quotes -> straight + whitespace; '
                             'the/tbody/tfoot wrappers dropped; spans part of labels',
            'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'predictions_sha256': hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
            'results': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    result = evaluate(args.manifest, args.predictions, jobs=max(1, args.jobs))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {key: result[key] for key in
               ('tables', 'errors_or_missing', 'teds_mean', 'teds_struct_mean')}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
