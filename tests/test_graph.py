"""Reusable ``Graph`` handle (handoff_graph_handle.md).

Parity basis: a ``Graph`` method must return *exactly* what the free
function returns for the same arguments — the algorithms are unchanged,
only where the spatial index is built moves. Every comparison here is
``np.testing.assert_array_equal`` (bit-identical), never ``allclose``.
"""

import numpy as np
import pytest

import dijkstra3d_sparse as ds
from helpers import random_cloud


def _case(seed, n_pts=500, box=10, n_stops=10):
    rng = np.random.default_rng(seed)
    vox = random_cloud(rng, n_pts, box=box)
    n = len(vox)
    node_cost = rng.uniform(0.1, 3.0, n).astype(np.float32)
    free = np.zeros(n, dtype=bool)
    free[rng.choice(n, size=n // 10, replace=False)] = True
    stop = np.zeros(n, dtype=bool)
    stop[rng.choice(np.arange(1, n), size=n_stops, replace=False)] = True
    return vox, node_cost, free, stop


@pytest.mark.parametrize("kind", ["sorted", "hash"])
@pytest.mark.parametrize("cost_mode", ["vertex", "additive", "geometric"])
@pytest.mark.parametrize("min_only", [True, False])
def test_dijkstra_field_parity(kind, cost_mode, min_only):
    vox, node_cost, free, stop = _case(1)
    kw = dict(
        node_cost=node_cost,
        connectivity=18,
        anisotropy=(1.0, 2.0, 3.5),
        cost_mode=cost_mode,
        free_mask=free,
        free_eps=1e-5,
        min_only=min_only,
        stop_mask=stop,
        stop_count=2,
    )
    g = ds.Graph(vox, index_kind=kind)
    for sources in (0, [0, 5, 17]):
        want_d, want_p = ds.dijkstra_field(vox, sources, index_kind=kind, **kw)
        got_d, got_p = g.dijkstra_field(sources, **kw)
        np.testing.assert_array_equal(got_d, want_d)
        np.testing.assert_array_equal(got_p, want_p)


@pytest.mark.parametrize("kind", ["sorted", "hash"])
def test_shortest_path_parity(kind):
    vox, node_cost, free, stop = _case(2)
    g = ds.Graph(vox, index_kind=kind)
    kw = dict(node_cost=node_cost, cost_mode="vertex", anisotropy=(1.0, 1.5, 2.0))

    want_p, want_c = ds.shortest_path(vox, 0, 42, index_kind=kind, **kw)
    got_p, got_c = g.shortest_path(0, 42, **kw)
    np.testing.assert_array_equal(got_p, want_p)
    assert got_c == want_c

    want = ds.shortest_path_to_set(vox, 0, stop, free_mask=free, index_kind=kind, **kw)
    got = g.shortest_path_to_set(0, stop, free_mask=free, **kw)
    np.testing.assert_array_equal(got[0], want[0])
    assert (got[1], got[2]) == (want[1], want[2])


def test_shortest_path_to_set_unreachable_parity():
    vox = np.array([[0, 0, 0], [1, 0, 0], [9, 9, 9]], dtype=np.int32)
    mask = np.zeros(3, dtype=bool)
    mask[2] = True
    g = ds.Graph(vox)
    p, hit, cost = g.shortest_path_to_set(0, mask)
    assert p.shape == (0, 3) and p.dtype == np.int32
    assert hit == -1 and cost == float("inf")
    p, cost = g.shortest_path(0, 2)
    assert p.shape == (0, 3) and cost == float("inf")


@pytest.mark.parametrize("kind", ["sorted", "hash"])
@pytest.mark.parametrize("connectivity", [6, 18, 26])
def test_connected_components_parity(kind, connectivity):
    rng = np.random.default_rng(3)
    a = random_cloud(rng, 300, box=8)
    b = random_cloud(rng, 300, box=8, offset=(100, 0, 0))
    vox = np.vstack([a, b])
    g = ds.Graph(vox, index_kind=kind)
    want_n, want_labels = ds.connected_components(vox, connectivity=connectivity, index_kind=kind)
    got_n, got_labels = g.connected_components(connectivity=connectivity)
    assert got_n == want_n
    np.testing.assert_array_equal(got_labels, want_labels)


def test_index_of_parity_and_strict():
    vox, *_ = _case(4)
    g = ds.Graph(vox)
    assert g.index_of(vox[7]) == 7
    queries = np.vstack([vox[[3, 1, 4]], [[99, 99, 99]]]).astype(np.int32)
    out = g.index_of(queries, strict=False)
    np.testing.assert_array_equal(out, ds.index_of(vox, queries, strict=False))
    assert out.tolist() == [3, 1, 4, -1]
    with pytest.raises(KeyError, match="not present"):
        g.index_of(queries)


def test_reuse_no_state_leak():
    # several *different* queries on one handle, each checked against a
    # freshly built stateless call — including a repeat of the first query
    # at the end, which must be unaffected by everything in between
    vox, node_cost, free, stop = _case(5)
    g = ds.Graph(vox)
    queries = [
        dict(sources=0),
        dict(sources=[3, 9], cost_mode="geometric", anisotropy=(2.0, 1.0, 1.0)),
        dict(sources=1, node_cost=node_cost, cost_mode="additive", free_mask=free),
        dict(sources=[0], node_cost=node_cost, cost_mode="vertex", stop_mask=stop, stop_count=3),
        dict(sources=2, min_only=False),
        dict(sources=0),
    ]
    for kw in queries:
        kw = dict(kw)
        sources = kw.pop("sources")
        want_d, want_p = ds.dijkstra_field(vox, sources, **kw)
        got_d, got_p = g.dijkstra_field(sources, **kw)
        np.testing.assert_array_equal(got_d, want_d)
        np.testing.assert_array_equal(got_p, want_p)


def test_grafting_loop_matches_free_functions():
    # the motivating consumer pattern: one handle per component, repeated
    # shortest_path_to_set against a growing anchor set + index_of on the
    # returned coordinates
    vox, node_cost, _, _ = _case(6)
    n = len(vox)
    g = ds.Graph(vox)
    anchor = np.zeros(n, dtype=bool)
    anchor[0] = True
    rng = np.random.default_rng(6)
    for q in rng.integers(0, n, 8):
        got = g.shortest_path_to_set(int(q), anchor, node_cost=node_cost, cost_mode="additive")
        want = ds.shortest_path_to_set(
            vox, int(q), anchor, node_cost=node_cost, cost_mode="additive"
        )
        np.testing.assert_array_equal(got[0], want[0])
        assert (got[1], got[2]) == (want[1], want[2])
        if got[1] >= 0:
            rows = g.index_of(got[0])
            np.testing.assert_array_equal(rows, ds.index_of(vox, got[0]))
            anchor[rows] = True


def test_construction_errors_match_free_functions():
    dup = np.array([[0, 0, 0], [1, 2, 3], [0, 0, 0]], dtype=np.int32)
    for kind in ("sorted", "hash"):
        with pytest.raises(ValueError, match="duplicate"):
            ds.Graph(dup, index_kind=kind)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        ds.Graph(np.zeros((2, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="integer"):
        ds.Graph(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="index_kind"):
        ds.Graph([[0, 0, 0], [1, 0, 0]], index_kind="btree")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(connectivity=7), "connectivity"),
        (dict(cost_mode="euclidean"), "cost_mode"),
        (dict(anisotropy=(1.0, 0.0, 1.0)), "anisotropy"),
        (dict(anisotropy=(1.0, -2.0, 1.0)), "anisotropy"),
        (dict(free_eps=0.0), "free_eps"),
        (dict(free_eps=float("nan")), "free_eps"),
        (dict(node_cost=[1.0, -0.5]), "finite and non-negative"),
        (dict(free_mask=[True]), r"shape \(N,\)"),
        (dict(stop_mask=[True]), r"shape \(N,\)"),
        (dict(stop_mask=[True, False], stop_count=-1), "stop_count"),
    ],
)
def test_method_argument_validation(kwargs, match):
    g = ds.Graph([[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match=match):
        g.dijkstra_field(0, **kwargs)
    with pytest.raises(ValueError, match="out of range"):
        g.dijkstra_field(5)
    with pytest.raises(ValueError, match="out of range"):
        g.shortest_path(0, 5)


def test_index_kind_is_fixed_at_construction():
    g = ds.Graph([[0, 0, 0], [1, 0, 0]])
    with pytest.raises(TypeError):
        g.dijkstra_field(0, index_kind="sorted")
    with pytest.raises(TypeError):
        g.connected_components(index_kind="sorted")


def test_handle_introspection():
    vox = np.array([[0, 0, 0], [2, 1, 0], [1, 1, 1]], dtype=np.int32)
    g = ds.Graph(vox, index_kind="sorted")
    assert g.n == 3
    assert g.index_kind == "sorted"
    np.testing.assert_array_equal(g.voxels, vox)
    assert g.voxels.dtype == np.int32
    assert not g.voxels.flags.writeable
    assert repr(g) == "Graph(n=3, index_kind='sorted')"


def test_handle_snapshot_survives_input_mutation():
    vox = np.array([[x, 0, 0] for x in range(5)], dtype=np.int32)
    g = ds.Graph(vox)
    vox[:] = 99  # the native handle owns its own copy of the coordinates
    dist, _ = g.dijkstra_field(0, connectivity=6)
    np.testing.assert_array_equal(dist, np.arange(5, dtype=float))
    assert g.index_of([3, 0, 0]) == 3


def test_graph_path_convenience():
    vox = np.array([[x, 0, 0] for x in range(5)], dtype=np.int32)
    g = ds.Graph(vox)
    dist, pred = g.dijkstra_field(0)
    np.testing.assert_array_equal(g.path(pred, 4), ds.path(vox, pred, 4))
    np.testing.assert_array_equal(g.path(pred, 4, dist=dist), ds.path(vox, pred, 4, dist=dist))


def test_empty_and_single_voxel_handles():
    g = ds.Graph(np.empty((0, 3), dtype=np.int32))
    assert g.n == 0
    dist, pred = g.dijkstra_field([])
    assert dist.shape == (0,) and dist.dtype == np.float64
    assert pred.shape == (0,) and pred.dtype == np.int64
    n_comp, labels = g.connected_components()
    assert n_comp == 0 and labels.shape == (0,)

    g1 = ds.Graph([[3, -4, 5]])
    dist, pred = g1.dijkstra_field(0)
    assert dist.tolist() == [0.0] and pred.tolist() == [-1]
    assert g1.connected_components() == (1, np.array([0], dtype=np.int32))
