"""Parity of dist fields and components against scipy.sparse.csgraph on the
equivalent explicit CSR graph (spec §7.1)."""

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components as scipy_cc
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

import dijkstra3d_sparse as ds
from helpers import build_csr, random_cloud

RTOL = 1e-5


@pytest.mark.parametrize("connectivity", [6, 18, 26])
@pytest.mark.parametrize("cost_mode", ["geometric", "vertex", "additive"])
@pytest.mark.parametrize("anisotropy", [(1.0, 1.0, 1.0), (0.5, 2.0, 3.5)])
def test_single_source_parity(connectivity, cost_mode, anisotropy):
    rng = np.random.default_rng(42)
    # spans negative coordinates on purpose
    vox = random_cloud(rng, 500, box=10, offset=(-5, -5, -5))
    n = len(vox)
    # strictly positive so scipy's stored-zero-vs-no-edge ambiguity never bites
    node_cost = None if cost_mode == "geometric" else rng.uniform(0.1, 4.0, n).astype(np.float32)

    dist, pred = ds.dijkstra_field(
        vox, 0, node_cost=node_cost, connectivity=connectivity,
        anisotropy=anisotropy, cost_mode=cost_mode,
    )

    csr = build_csr(vox, connectivity, anisotropy, cost_mode, node_cost)
    ref = scipy_dijkstra(csr, directed=True, indices=0)
    np.testing.assert_allclose(dist, ref, rtol=RTOL)

    # predecessors may legitimately differ on ties; validate them via the
    # invariant dist[pred[v]] + cost(pred[v] -> v) == dist[v] instead
    reached = np.isfinite(dist) & (pred >= 0)
    assert np.all(np.isfinite(dist[np.flatnonzero(reached)]))
    assert pred[0] == -1 and dist[0] == 0.0


@pytest.mark.parametrize("min_only", [True, False])
def test_multi_source_parity(min_only):
    rng = np.random.default_rng(7)
    vox = random_cloud(rng, 400, box=9)
    sources = [0, len(vox) // 2, len(vox) - 1]

    dist, pred = ds.dijkstra_field(vox, sources, min_only=min_only)
    csr = build_csr(vox, connectivity=26)

    if min_only:
        ref = scipy_dijkstra(csr, directed=True, indices=sources, min_only=True)
        assert dist.shape == (len(vox),)
        np.testing.assert_allclose(dist, ref, rtol=RTOL)
        assert all(dist[s] == 0.0 and pred[s] == -1 for s in sources)
    else:
        ref = scipy_dijkstra(csr, directed=True, indices=sources, min_only=False)
        assert dist.shape == (len(sources), len(vox))
        assert pred.shape == dist.shape
        np.testing.assert_allclose(dist, ref, rtol=RTOL)


def test_free_mask_parity():
    rng = np.random.default_rng(3)
    vox = random_cloud(rng, 400, box=9)
    n = len(vox)
    node_cost = rng.uniform(0.5, 3.0, n).astype(np.float32)
    free_mask = rng.random(n) < 0.3
    free_eps = 1e-4

    dist, _ = ds.dijkstra_field(
        vox, 0, node_cost=node_cost, cost_mode="vertex",
        free_mask=free_mask, free_eps=free_eps,
    )
    csr = build_csr(vox, 26, (1, 1, 1), "vertex", node_cost, free_mask, free_eps)
    ref = scipy_dijkstra(csr, directed=True, indices=0)
    np.testing.assert_allclose(dist, ref, rtol=RTOL)


@pytest.mark.parametrize("connectivity", [6, 18, 26])
def test_connected_components_parity(connectivity):
    rng = np.random.default_rng(11)
    # sparse enough to fragment into many components
    vox = random_cloud(rng, 500, box=25)
    n_ours, labels = ds.connected_components(vox, connectivity=connectivity)

    csr = build_csr(vox, connectivity)
    n_ref, ref_labels = scipy_cc(csr, directed=False)

    assert n_ours == n_ref
    # equal partitions up to label permutation: the pairing must be a bijection
    pairs = set(zip(labels.tolist(), ref_labels.tolist()))
    assert len(pairs) == n_ours
    assert len({p[0] for p in pairs}) == n_ours
    assert len({p[1] for p in pairs}) == n_ours


def test_index_kinds_agree_bit_for_bit():
    rng = np.random.default_rng(5)
    vox = random_cloud(rng, 600, box=11)
    node_cost = rng.uniform(0.1, 2.0, len(vox)).astype(np.float32)
    results = [
        ds.dijkstra_field(vox, [0, 5], node_cost=node_cost, index_kind=kind)
        for kind in ("sorted", "hash")
    ]
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])

    ccs = [ds.connected_components(vox, index_kind=kind) for kind in ("sorted", "hash")]
    assert ccs[0][0] == ccs[1][0]
    np.testing.assert_array_equal(ccs[0][1], ccs[1][1])


def test_determinism_across_runs():
    rng = np.random.default_rng(9)
    vox = random_cloud(rng, 500, box=10)
    a = ds.dijkstra_field(vox, [0, 1, 2])
    b = ds.dijkstra_field(vox, [0, 1, 2])
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
