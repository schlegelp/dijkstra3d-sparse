"""Grouped connected components and label adjacency.

Both primitives exist so a caller can contract the voxel graph (level sets
into "rings", rings into a skeleton graph) without ever materializing an
edge list. The tests here use an explicit edge list *as the oracle* — the
thing the primitives let the caller avoid — and check the coordinate-native
answer matches it.
"""

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as scipy_cc

import dijkstra3d_sparse as ds
from helpers import offsets, random_cloud


def edge_pairs(voxels, connectivity=26):
    """Explicit undirected edge list (the oracle these primitives replace)."""
    lookup = {tuple(v): i for i, v in enumerate(np.asarray(voxels))}
    pairs = []
    for i, v in enumerate(np.asarray(voxels)):
        for off, _ in offsets(connectivity):
            j = lookup.get(tuple(v + off))
            if j is not None and i < j:
                pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def reference_grouped_cc(voxels, group, connectivity=26):
    """(n_components, labels) via scipy over the same-group edge list."""
    n = len(voxels)
    e = edge_pairs(voxels, connectivity)
    if len(e):
        e = e[np.asarray(group)[e[:, 0]] == np.asarray(group)[e[:, 1]]]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return scipy_cc(g, directed=False)


def reference_adjacency(voxels, labels, connectivity=26):
    """Sorted distinct inter-label pairs via the explicit edge list."""
    labels = np.asarray(labels, dtype=np.int64)
    e = edge_pairs(voxels, connectivity)
    if not len(e):
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.sort(labels[e], axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(pairs, axis=0).reshape(-1, 2)


def same_partition(a, b):
    """Label arrays describe the same partition (up to relabelling)."""
    a, b = np.asarray(a), np.asarray(b)
    return len(np.unique(np.stack([a, b], axis=1), axis=0)) == len(np.unique(a)) == len(np.unique(b))


LINE = np.array([[x, 0, 0] for x in range(6)], dtype=np.int32)


# --------------------------------------------------------------------------
# grouped connected components
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sorted", "hash"])
@pytest.mark.parametrize("connectivity", [6, 18, 26])
def test_group_none_and_uniform_match_ungrouped(kind, connectivity):
    rng = np.random.default_rng(0)
    vox = np.vstack([random_cloud(rng, 300, box=8), random_cloud(rng, 300, box=8, offset=(50, 0, 0))])
    want = ds.connected_components(vox, connectivity=connectivity, index_kind=kind)
    for group in (None, np.zeros(len(vox), dtype=np.int64), np.full(len(vox), -7)):
        got_n, got_labels = ds.connected_components(
            vox, group=group, connectivity=connectivity, index_kind=kind
        )
        assert got_n == want[0]
        np.testing.assert_array_equal(got_labels, want[1])


def test_group_splits_a_run_at_the_boundary():
    # coordinate-adjacent across the boundary, but the groups differ, so the
    # union must not happen
    n, labels = ds.connected_components(LINE, group=[0, 0, 0, 1, 1, 1])
    assert n == 2
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])


def test_group_constrains_unions_but_never_merges_by_value():
    # The value 0 appears in two spatially separate runs: two components,
    # not one. This is the property that matters most.
    n, labels = ds.connected_components(LINE, group=[0, 0, 1, 1, 0, 0])
    assert n == 3
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2])

    # ...and the same in a U shape, where the two arms share a group value
    # but only touch through the (differently grouped) bend.
    u = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0], [2, 2, 0], [2, 1, 0], [2, 0, 0]],
        dtype=np.int32,
    )
    n, labels = ds.connected_components(u, group=[5, 5, 9, 9, 9, 5, 5], connectivity=6)
    assert n == 3
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 1, 2, 2])


def test_distinct_group_per_voxel_gives_singletons():
    n_vox = len(LINE)
    n, labels = ds.connected_components(LINE, group=np.arange(n_vox))
    assert n == n_vox
    np.testing.assert_array_equal(labels, np.arange(n_vox))


@pytest.mark.parametrize("kind", ["sorted", "hash"])
@pytest.mark.parametrize("connectivity", [6, 26])
def test_grouped_cc_parity_with_scipy(kind, connectivity):
    rng = np.random.default_rng(1)
    vox = random_cloud(rng, 400, box=7)
    # a level-set-like grouping: the caller's actual use is floor(dist/step)
    group = (np.linalg.norm(vox.astype(float), axis=1) / 2.0).astype(np.int64)
    got_n, got_labels = ds.connected_components(
        vox, group=group, connectivity=connectivity, index_kind=kind
    )
    want_n, want_labels = reference_grouped_cc(vox, group, connectivity)
    assert got_n == want_n
    assert same_partition(got_labels, want_labels)
    # every component is group-pure, and labels are dense in first-appearance order
    assert len(np.unique(got_labels)) == got_n
    for lab in range(got_n):
        assert len(np.unique(group[got_labels == lab])) == 1
    np.testing.assert_array_equal(np.unique(got_labels[np.argsort(got_labels, kind="stable")]), np.arange(got_n))


def test_grouped_cc_is_deterministic_and_graph_parity():
    rng = np.random.default_rng(2)
    vox = random_cloud(rng, 300, box=6)
    group = rng.integers(0, 4, len(vox))
    first = ds.connected_components(vox, group=group)
    for _ in range(3):
        again = ds.connected_components(vox, group=group)
        assert again[0] == first[0]
        np.testing.assert_array_equal(again[1], first[1])
    # Graph handle must give exactly the free-function answer
    for kind in ("sorted", "hash"):
        g = ds.Graph(vox, index_kind=kind)
        got_n, got_labels = g.connected_components(group=group)
        assert got_n == first[0]
        np.testing.assert_array_equal(got_labels, first[1])
        # ungrouped call on the same handle is unaffected by the grouped one
        np.testing.assert_array_equal(
            g.connected_components()[1], ds.connected_components(vox)[1]
        )


def test_group_accepts_any_integer_dtype_and_rejects_the_rest():
    want = ds.connected_components(LINE, group=[0, 0, 0, 1, 1, 1])
    for dtype in (np.int8, np.uint8, np.int32, np.int64):
        got = ds.connected_components(LINE, group=np.array([0, 0, 0, 1, 1, 1], dtype=dtype))
        assert got[0] == want[0]
        np.testing.assert_array_equal(got[1], want[1])

    with pytest.raises(ValueError, match="group must be an integer array"):
        ds.connected_components(LINE, group=np.zeros(6, dtype=np.float64))
    with pytest.raises(ValueError, match=r"group must have shape \(N,\)"):
        ds.connected_components(LINE, group=[0, 1])
    with pytest.raises(ValueError, match="connectivity"):
        ds.connected_components(LINE, group=np.zeros(6, dtype=np.int64), connectivity=7)


def test_grouped_cc_empty_and_single_voxel():
    empty = np.empty((0, 3), dtype=np.int32)
    n, labels = ds.connected_components(empty, group=np.empty(0, dtype=np.int64))
    assert n == 0 and labels.shape == (0,) and labels.dtype == np.int32
    n, labels = ds.connected_components([[3, -4, 5]], group=[42])
    assert n == 1 and labels.tolist() == [0]


# --------------------------------------------------------------------------
# label adjacency
# --------------------------------------------------------------------------


def test_adjacent_pair_emitted_once_same_label_empty():
    two = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    np.testing.assert_array_equal(ds.label_adjacency(two, [7, 3]), [[3, 7]])

    out = ds.label_adjacency(two, [5, 5])
    assert out.shape == (0, 2) and out.dtype == np.int64


def test_labels_may_be_negative_and_non_dense():
    tri = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int32)
    # three mutually adjacent labels (26-conn: all three voxels pairwise
    # adjacent only for the first/second and second/third at distance 1)
    np.testing.assert_array_equal(ds.label_adjacency(tri, [-9, 1000, 4]), [[-9, 1000], [4, 1000]])

    # a genuinely mutually-adjacent triple
    clump = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32)
    np.testing.assert_array_equal(
        ds.label_adjacency(clump, [-1, 5, 2]), [[-1, 2], [-1, 5], [2, 5]]
    )


def test_alternating_run_dedups_to_one_pair():
    alt = np.array([[x, 0, 0] for x in range(9)], dtype=np.int32)
    labels = np.array([0, 1] * 4 + [0])
    np.testing.assert_array_equal(ds.label_adjacency(alt, labels), [[0, 1]])


@pytest.mark.parametrize("connectivity,expected", [(6, 0), (18, 1), (26, 1)])
def test_connectivity_sensitivity(connectivity, expected):
    diag = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.int32)
    out = ds.label_adjacency(diag, [0, 1], connectivity=connectivity)
    assert out.shape == (expected, 2)


@pytest.mark.parametrize("kind", ["sorted", "hash"])
@pytest.mark.parametrize("connectivity", [6, 18, 26])
def test_label_adjacency_parity_with_edge_list(kind, connectivity):
    # the cheapest way to be sure the half-offset probe covers every
    # undirected pair exactly once
    rng = np.random.default_rng(3)
    vox = random_cloud(rng, 500, box=8)
    labels = rng.integers(-5, 20, len(vox))
    got = ds.label_adjacency(vox, labels, connectivity=connectivity, index_kind=kind)
    want = reference_adjacency(vox, labels, connectivity)
    np.testing.assert_array_equal(got, want)
    assert np.all(got[:, 0] < got[:, 1])
    # lexicographically sorted
    assert np.array_equal(got, got[np.lexsort((got[:, 1], got[:, 0]))])


def test_label_adjacency_graph_parity_and_reuse():
    rng = np.random.default_rng(4)
    vox = random_cloud(rng, 300, box=6)
    labels_a = rng.integers(0, 8, len(vox))
    labels_b = rng.integers(0, 3, len(vox))
    for kind in ("sorted", "hash"):
        g = ds.Graph(vox, index_kind=kind)
        for labels in (labels_a, labels_b, labels_a):
            np.testing.assert_array_equal(
                g.label_adjacency(labels),
                ds.label_adjacency(vox, labels, index_kind=kind),
            )


def test_label_adjacency_edge_cases_and_validation():
    empty = np.empty((0, 3), dtype=np.int32)
    out = ds.label_adjacency(empty, np.empty(0, dtype=np.int64))
    assert out.shape == (0, 2) and out.dtype == np.int64

    single = ds.label_adjacency([[1, 2, 3]], [4])
    assert single.shape == (0, 2)

    # isolated voxels never touch, whatever their labels
    far = np.array([[0, 0, 0], [50, 0, 0]], dtype=np.int32)
    assert ds.label_adjacency(far, [0, 1]).shape == (0, 2)

    with pytest.raises(ValueError, match="labels must be an integer array"):
        ds.label_adjacency(LINE, np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError, match=r"labels must have shape \(N,\)"):
        ds.label_adjacency(LINE, [0, 1])
    with pytest.raises(ValueError, match="connectivity"):
        ds.label_adjacency(LINE, np.zeros(6, dtype=np.int64), connectivity=7)


# --------------------------------------------------------------------------
# the two together: the wavefront pipeline, edge-list-free
# --------------------------------------------------------------------------


def test_wavefront_pipeline_matches_edge_list_pipeline():
    """The motivating caller: components -> geodesic field -> rings ->
    contract onto the ring graph, with no edge list anywhere."""
    rng = np.random.default_rng(5)
    vox = np.vstack([random_cloud(rng, 400, box=7), random_cloud(rng, 400, box=7, offset=(40, 0, 0))])
    g = ds.Graph(vox)

    n_comp, comp = g.connected_components()
    seeds = [int(np.flatnonzero(comp == c)[0]) for c in range(n_comp)]
    dist, _ = g.dijkstra_field(seeds, cost_mode="geometric")
    lvl = np.floor(np.where(np.isfinite(dist), dist, 0.0) / 2.0).astype(np.int64)

    n_rings, rings = g.connected_components(group=lvl)
    sk_edges = g.label_adjacency(rings)

    want_n, want_rings = reference_grouped_cc(vox, lvl)
    assert n_rings == want_n
    assert same_partition(rings, want_rings)
    np.testing.assert_array_equal(sk_edges, reference_adjacency(vox, rings))
    # rings only ever touch rings of an adjacent (or equal) level
    assert np.all(np.abs(lvl[[np.flatnonzero(rings == a)[0] for a, _ in sk_edges]]
                         - lvl[[np.flatnonzero(rings == b)[0] for _, b in sk_edges]]) <= 1)
