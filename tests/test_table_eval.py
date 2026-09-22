import hashlib
import json
import random

import pytest

from training.table_eval import (
    _Node,
    _update_cost,
    evaluate,
    parse_table_html,
    teds,
    tree_edit_distance,
)


# --- independent reference: ordered TED over exhaustive edit mappings ---------


def intervals(tree):
    """Preorder labels with [tin, tout) ancestry intervals."""
    labels, tin, tout, clock = [], {}, {}, [0]

    def visit(node):
        label, children = node
        index = len(labels)
        labels.append(label)
        tin[index] = clock[0]
        clock[0] += 1
        for child in children:
            visit(child)
        tout[index] = clock[0]

    visit(tree)
    return labels, tin, tout


def relation(tin, tout, left, right):
    if left == right:
        return 'same'
    if tin[left] <= tin[right] < tout[left]:
        return 'ancestor'
    if tin[right] <= tin[left] < tout[right]:
        return 'descendant'
    return 'left' if tin[left] < tin[right] else 'right'


def compatible(rel_a, rel_b, u, v, i, j):
    return (rel_a(u, i) == rel_b(v, j)) and (rel_a(i, u) == rel_b(j, v))


def brute_ted(tree_a, tree_b, cost):
    """Min edit cost over every order-preserving node mapping."""
    labels_a, tin_a, tout_a = intervals(tree_a)
    labels_b, tin_b, tout_b = intervals(tree_b)
    n, m = len(labels_a), len(labels_b)
    rel_a = lambda u, i: relation(tin_a, tout_a, u, i)  # noqa: E731
    rel_b = lambda v, j: relation(tin_b, tout_b, v, j)  # noqa: E731
    best = [float(n + m)]

    def extend(i, mapping):
        if i == n:
            k = len(mapping)
            total = (n - k) + (m - k) + sum(cost(labels_a[u], labels_b[v])
                                           for u, v in mapping)
            best[0] = min(best[0], total)
            return
        extend(i + 1, mapping)
        for j in range(m):
            if any(v == j for _, v in mapping):
                continue
            if all(compatible(rel_a, rel_b, u, v, i, j) for u, v in mapping):
                extend(i + 1, mapping + [(i, j)])

    extend(0, [])
    return best[0]


def node_tree(tree):
    def build(item):
        label, children = item
        node = _Node(label)
        node.children = [build(child) for child in children]
        return node

    return build(tree)


def zs_ted(tree_a, tree_b):
    cost = lambda a, b: 0.0 if a.tag == b.tag else 1.0  # noqa: E731
    return tree_edit_distance(node_tree(tree_a), node_tree(tree_b), cost)


def random_tree(rng, size):
    children = []
    remaining = size - 1
    while remaining > 0:
        take = rng.randint(1, remaining)
        children.append(random_tree(rng, take))
        remaining -= take
    return (rng.choice('ab'), tuple(children))


def test_hand_computed_distances():
    assert zs_ted(('r', ()), ('r', ())) == 0
    assert zs_ted(('r', ()), ('q', ())) == 1
    assert zs_ted(('r', (('a', ()),)), ('r', (('a', ()), ('b', ())))) == 1
    assert zs_ted(('r', (('a', ()), ('b', ()))),
                  ('r', (('b', ()), ('a', ())))) == 2
    assert zs_ted(('r', (('a', (('c', ()),)), ('b', ()))),
                  ('r', (('a', ()), ('b', (('c', ()),))))) == 2


@pytest.mark.parametrize('seed', range(40))
def test_zhang_shasha_matches_exhaustive_mapping_reference(seed):
    rng = random.Random(20260922 + seed)
    tree_a = random_tree(rng, rng.randint(1, 5))
    tree_b = random_tree(rng, rng.randint(1, 5))
    assert zs_ted(tree_a, tree_b) == brute_ted(tree_a, tree_b,
                                               lambda x, y: 0.0 if x == y else 1.0)


def test_distance_is_symmetric_on_random_trees():
    rng = random.Random(7)
    for _ in range(20):
        tree_a = random_tree(rng, rng.randint(1, 5))
        tree_b = random_tree(rng, rng.randint(1, 5))
        assert zs_ted(tree_a, tree_b) == zs_ted(tree_b, tree_a)


TABLE = '<table><tr><td>{}</td></tr></table>'


def test_identical_table_scores_one():
    assert teds(TABLE.format('Ala ma kota'), TABLE.format('Ala ma kota')) == 1.0


def test_cell_text_distance_and_struct_variant():
    # virtual root + table + tr + td = 4 nodes; one relabel of 'Ala'/'Ola'
    # costs 1 - 1/3, so TEDS = 1 - (1/3)/4.
    assert teds(TABLE.format('Ala'), TABLE.format('Ola')) == pytest.approx(11 / 12)
    assert teds(TABLE.format('Ala'), TABLE.format('Ola'), structure_only=True) == 1.0


def test_missing_side_scores_zero_and_both_empty_score_one():
    assert teds(TABLE.format('x'), '') == 0.0
    assert teds('', TABLE.format('x')) == 0.0
    assert teds('', '') == 1.0


def test_extra_row_costs_two_insertions():
    hyp = '<table><tr><td>x</td></tr><tr><td>y</td></tr></table>'
    assert teds(TABLE.format('x'), hyp) == pytest.approx(2 / 3)


def test_spans_are_part_of_the_label():
    assert teds(TABLE.format('x').replace('td>', 'td colspan="2">', 1),
                TABLE.format('x')) == pytest.approx(0.75)
    assert teds(TABLE.format('x'), TABLE.format('x').replace('td', 'th', 1)) == pytest.approx(0.75)


def test_canonicalization_drops_wrappers_and_transparent_tags():
    assert teds('<table><tbody><tr><td>x</td></tr></tbody></table>',
                TABLE.format('x')) == 1.0
    assert teds(TABLE.format('foo bar'),
                '<table><tr><td>foo<br>bar</td></tr></table>') == 1.0
    assert teds(TABLE.format('tekst'),
                '<table><tr><td><p class="x">tekst</p></td></tr></table>') == 1.0


def test_normalization_unifies_quotes_and_whitespace():
    assert teds('<table><tr><td>\u201eAla\u201d  ma\t kota</td></tr></table>',
                TABLE.format('"Ala" ma kota')) == 1.0


def test_nested_table_inside_cell_is_structural():
    # Hypothesis wraps the cell in a nested table: 7 nodes vs 4, three extra
    # nodes must be deleted/inserted whatever the mapping.
    hyp = '<table><tr><td><table><tr><td>x</td></tr></table></td></tr></table>'
    assert teds(TABLE.format('x'), hyp) == pytest.approx(1 - 3 / 7)


def fixture(tmp_path, html_rows):
    image = tmp_path / 'page.png'
    image.write_bytes(b'fixture')
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps({
        'id': str(index), 'image': image.name, 'sha256': digest, 'html': html,
    }) for index, html in enumerate(html_rows)), encoding='utf-8')
    return manifest


def predictions(tmp_path, rows):
    path = tmp_path / 'predictions.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    return path


def test_failures_stay_in_denominator(tmp_path):
    manifest = fixture(tmp_path, [TABLE.format('x')] * 2)
    paths = (manifest, predictions(tmp_path, [
        {'id': '0', 'status': 'ok', 'html': TABLE.format('x'), 'elapsed_seconds': 1.5},
        {'id': '1', 'status': 'error', 'html': ''},
    ]))
    result = evaluate(*paths)
    assert result['tables'] == 2
    assert result['errors_or_missing'] == 1
    assert result['teds_mean'] == 0.5
    assert result['results'][0]['elapsed_seconds'] == 1.5
    assert result['results'][1]['teds'] == 0.0
    assert result['protocol_version'] == 'polocrbench-table-teds-v1'


def test_missing_predictions_score_zero(tmp_path):
    result = evaluate(fixture(tmp_path, [TABLE.format('x')]), predictions(tmp_path, []))
    assert result['teds_mean'] == 0.0
    assert result['results'][0]['status'] == 'missing'


def test_error_is_not_a_successful_empty_table(tmp_path):
    manifest = fixture(tmp_path, ['', ''])
    result = evaluate(manifest, predictions(tmp_path, [
        {'id': '0', 'status': 'ok', 'html': ''},
        {'id': '1', 'status': 'error', 'html': ''},
    ]))
    assert result['teds_mean'] == 0.5


def test_invalid_inputs_raise(tmp_path):
    manifest = fixture(tmp_path, [TABLE.format('x')])
    with pytest.raises(ValueError, match='unknown'):
        evaluate(manifest, predictions(tmp_path, [{'id': '9', 'status': 'ok', 'html': ''}]))
    with pytest.raises(ValueError, match='Duplicate'):
        evaluate(manifest, predictions(tmp_path, [
            {'id': '0', 'status': 'ok', 'html': ''}, {'id': '0', 'status': 'ok', 'html': ''}]))
    with pytest.raises(ValueError, match='HTML must be a string'):
        evaluate(manifest, predictions(tmp_path, [{'id': '0', 'status': 'ok', 'html': 7}]))
    image = tmp_path / 'page.png'
    bad = tmp_path / 'bad.jsonl'
    bad.write_text(json.dumps({'id': '0', 'image': image.name,
                               'sha256': hashlib.sha256(image.read_bytes()).hexdigest()}),
                   encoding='utf-8')
    with pytest.raises(ValueError, match='HTML missing'):
        evaluate(bad, predictions(tmp_path, []))


def test_parallel_jobs_are_deterministic(tmp_path):
    manifest = fixture(tmp_path, [TABLE.format('Ala'), TABLE.format('Ola')])
    paths = (manifest, predictions(tmp_path, [
        {'id': '0', 'status': 'ok', 'html': TABLE.format('Ola')},
        {'id': '1', 'status': 'ok', 'html': TABLE.format('')},
    ]))
    assert evaluate(*paths, jobs=1) == evaluate(*paths, jobs=2)
