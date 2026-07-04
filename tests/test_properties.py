"""Structural invariants of the returned fields and paths (spec §7.3)."""

import numpy as np
import pytest

import dijkstra3d_sparse as ds
from helpers import build_csr, path_cost, random_cloud

CASES = [
    # (connectivity, anisotropy, cost_mode, with_node_cost, with_free_mask)
    (6, (1, 1, 1), "geometric", False, False),
    (18, (2, 1, 0.5), "geometric", False, False),
    (26, (1, 1, 1), "vertex", True, False),
    (26, (0.7, 1.3, 2.1), "additive", True, False),
    (26, (1, 1, 1), "vertex", True, True),
]


@pytest.mark.parametrize("connectivity,anisotropy,cost_mode,with_nc,with_fm", CASES)
def test_field_and_path_invariants(connectivity, anisotropy, cost_mode, with_nc, with_fm):
    rng = np.random.default_rng(17)
    vox = random_cloud(rng, 450, box=10)
    n = len(vox)
    node_cost = rng.uniform(0.05, 3.0, n).astype(np.float32) if with_nc else None
    free_mask = (rng.random(n) < 0.25) if with_fm else None
    free_eps = 1e-5
    sources = [0, n // 3]

    dist, pred = ds.dijkstra_field(
        vox, sources, node_cost=node_cost, connectivity=connectivity,
        anisotropy=anisotropy, cost_mode=cost_mode,
        free_mask=free_mask, free_eps=free_eps,
    )

    # sources at distance 0 with no predecessor
    for s in sources:
        assert dist[s] == 0.0
        assert pred[s] == -1

    # unreached <=> dist inf <=> pred -1 (except sources)
    unreached = ~np.isfinite(dist)
    assert np.all(pred[unreached] == -1)
    reached_non_source = np.isfinite(dist)
    reached_non_source[sources] = False
    assert np.all(pred[reached_non_source] >= 0)

    # triangle inequality along every edge of the equivalent explicit graph
    csr = build_csr(vox, connectivity, anisotropy, cost_mode, node_cost, free_mask, free_eps).tocoo()
    finite = np.isfinite(dist)
    lhs = dist[csr.col]
    rhs = dist[csr.row] + csr.data
    mask = finite[csr.row]
    assert np.all(lhs[mask] <= rhs[mask] * (1 + 1e-9) + 1e-12)

    # path reconstruction: endpoints, adjacency, and cost == dist[target]
    targets = np.flatnonzero(np.isfinite(dist))[::37]
    max_manhattan = {6: 1, 18: 2, 26: 3}[connectivity]
    for t in targets:
        p = ds.path(vox, pred, t, dist=dist)
        assert tuple(p[-1]) == tuple(vox[t])
        assert tuple(p[0]) in {tuple(vox[s]) for s in sources} or len(p) == 1
        steps = np.diff(p.astype(np.int64), axis=0)
        assert np.all(np.abs(steps) <= 1)
        assert np.all(np.abs(steps).sum(axis=1) >= 1)
        assert np.all(np.abs(steps).sum(axis=1) <= max_manhattan)

        rows = [int(ds.index_of(vox, c)) for c in p]
        cost = path_cost(vox, rows, connectivity, anisotropy, cost_mode, node_cost, free_mask, free_eps)
        np.testing.assert_allclose(cost, dist[t], rtol=1e-9, atol=1e-12)


def test_free_mask_makes_masked_nodes_cheap():
    # straight line; masking the interior makes the far end ~free to reach
    vox = np.array([[x, 0, 0] for x in range(10)], dtype=np.int32)
    free_mask = np.ones(10, dtype=bool)
    free_mask[[0, 9]] = False
    dist, _ = ds.dijkstra_field(
        vox, 0, connectivity=6, cost_mode="geometric",
        free_mask=free_mask, free_eps=1e-6,
    )
    # 8 free hops + 1 real step into the unmasked end
    np.testing.assert_allclose(dist[9], 8 * 1e-6 + 1.0, rtol=1e-9)
    np.testing.assert_allclose(dist[8], 8 * 1e-6, rtol=1e-9)


def test_unreachable_cluster():
    vox = np.array([[0, 0, 0], [1, 0, 0], [10, 10, 10], [11, 10, 10]], dtype=np.int32)
    dist, pred = ds.dijkstra_field(vox, 0)
    assert np.isfinite(dist[:2]).all()
    assert np.isinf(dist[2:]).all()
    assert pred[2] == -1 and pred[3] == -1
    with pytest.raises(ValueError, match="not reached"):
        ds.path(vox, pred, 2, dist=dist)
    # without dist the degenerate single-voxel path is documented behaviour
    assert ds.path(vox, pred, 2).tolist() == [[10, 10, 10]]


def test_vertex_mode_matches_dijkstra3d_semantics():
    # cost into a voxel = its node_cost * step length; path around an
    # expensive voxel must win
    vox = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]],
                   dtype=np.int32)
    node_cost = np.array([1, 100, 1, 1, 1, 1], dtype=np.float32)
    dist, pred = ds.dijkstra_field(vox, 0, node_cost=node_cost, connectivity=6,
                                   cost_mode="vertex")
    # 0 -> (0,1,0) -> (1,1,0) -> (2,1,0) -> (2,0,0): cost 4, beats 100 + 1
    np.testing.assert_allclose(dist[2], 4.0)
    p = ds.path(vox, pred, 2)
    assert [1, 0, 0] not in p.tolist()


def test_additive_mode_semantics():
    vox = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int32)
    node_cost = np.array([0.0, 0.5, 0.25], dtype=np.float32)
    dist, _ = ds.dijkstra_field(vox, 0, node_cost=node_cost, connectivity=6,
                                cost_mode="additive")
    np.testing.assert_allclose(dist, [0.0, 1.5, 2.75])


def test_min_only_false_rows_match_individual_runs():
    rng = np.random.default_rng(23)
    vox = random_cloud(rng, 300, box=8)
    sources = [1, 42, 7]
    dist2d, pred2d = ds.dijkstra_field(vox, sources, min_only=False)
    for i, s in enumerate(sources):
        d, p = ds.dijkstra_field(vox, [s])
        np.testing.assert_array_equal(dist2d[i], d)
        np.testing.assert_array_equal(pred2d[i], p)
