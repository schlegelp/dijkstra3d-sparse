"""Early termination & search-to-a-set (extension spec §8).

Correctness basis: Dijkstra settles nodes in non-decreasing distance order,
so stopping when the first anchor is popped yields exact values on
everything settled so far — the partial field must be *identical* to the
full field on every node at least as close as the hit.
"""

import numpy as np
import pytest

import dijkstra3d_sparse as ds
from helpers import random_cloud

# strictly positive node costs: no zero-cost edges, so every node with
# dist_full <= dist_full[hit] is settled (and therefore final) at the break
COST_KW = dict(cost_mode="vertex", connectivity=26)


def _random_case(seed, n_anchors=12):
    rng = np.random.default_rng(seed)
    vox = random_cloud(rng, 500, box=10)
    n = len(vox)
    node_cost = rng.uniform(0.1, 3.0, n).astype(np.float32)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(np.arange(1, n), size=n_anchors, replace=False)] = True
    return vox, node_cost, mask


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_early_exit_prefix_parity(seed):
    vox, node_cost, mask = _random_case(seed)
    full_d, full_p = ds.dijkstra_field(vox, 0, node_cost=node_cost, **COST_KW)
    part_d, part_p = ds.dijkstra_field(
        vox, 0, node_cost=node_cost, stop_mask=mask, stop_count=1, **COST_KW
    )

    reachable_anchors = np.flatnonzero(mask & np.isfinite(full_d))
    assert reachable_anchors.size > 0, "test setup: want reachable anchors"
    hit_dist = full_d[reachable_anchors].min()

    # identical settle prefix: everything at least as close as the hit is final
    prefix = full_d <= hit_dist
    np.testing.assert_array_equal(part_d[prefix], full_d[prefix])
    np.testing.assert_array_equal(part_p[prefix], full_p[prefix])

    # the search actually stopped early: strictly fewer voxels touched
    assert np.isfinite(part_d).sum() < np.isfinite(full_d).sum()

    # nothing outside the touched region got a value
    assert np.all(part_p[~np.isfinite(part_d)] == -1)


@pytest.mark.parametrize("seed", [4, 5])
def test_to_set_matches_full_field(seed):
    vox, node_cost, mask = _random_case(seed)
    full_d, _ = ds.dijkstra_field(vox, 0, node_cost=node_cost, **COST_KW)

    p, hit, cost = ds.shortest_path_to_set(vox, 0, mask, node_cost=node_cost, **COST_KW)

    # hit is the nearest reachable anchor (ties: compare distance, not index)
    reachable = np.flatnonzero(mask & np.isfinite(full_d))
    assert mask[hit]
    np.testing.assert_allclose(cost, full_d[reachable].min(), rtol=1e-12)
    np.testing.assert_allclose(cost, full_d[hit], rtol=1e-12)

    # path endpoints + adjacency
    assert tuple(p[0]) == tuple(vox[0])
    assert tuple(p[-1]) == tuple(vox[hit])
    steps = np.abs(np.diff(p.astype(np.int64), axis=0))
    assert np.all(steps <= 1) and np.all(steps.sum(axis=1) >= 1)


def test_stop_count_k_settles_k_nearest_anchors():
    vox, node_cost, mask = _random_case(6)
    full_d, _ = ds.dijkstra_field(vox, 0, node_cost=node_cost, **COST_KW)
    k = 4
    part_d, _ = ds.dijkstra_field(
        vox, 0, node_cost=node_cost, stop_mask=mask, stop_count=k, **COST_KW
    )
    # the k nearest anchors (by full-field distance) are settled and final
    anchor_rows = np.flatnonzero(mask)
    order = anchor_rows[np.argsort(full_d[anchor_rows])]
    np.testing.assert_array_equal(part_d[order[:k]], full_d[order[:k]])
    # ...and anchors strictly beyond the k-th were never settled
    kth = full_d[order[k - 1]]
    beyond = order[full_d[order] > kth]
    assert np.all(~np.isfinite(part_d[beyond]) | (part_d[beyond] >= kth))


def test_stop_count_zero_and_empty_mask_are_full_field():
    vox, node_cost, mask = _random_case(7)
    full = ds.dijkstra_field(vox, 0, node_cost=node_cost, **COST_KW)
    zero = ds.dijkstra_field(
        vox, 0, node_cost=node_cost, stop_mask=mask, stop_count=0, **COST_KW
    )
    empty = ds.dijkstra_field(
        vox, 0, node_cost=node_cost, stop_mask=np.zeros(len(vox), bool), **COST_KW
    )
    for got in (zero, empty):
        np.testing.assert_array_equal(got[0], full[0])
        np.testing.assert_array_equal(got[1], full[1])


def test_stop_count_exceeds_reachable_anchors():
    # anchors: one reachable, plus mask over an unreachable far cluster
    vox = np.array(
        [[x, 0, 0] for x in range(5)] + [[50, 50, 50], [51, 50, 50]], dtype=np.int32
    )
    mask = np.zeros(len(vox), bool)
    mask[[3, 5, 6]] = True  # row 3 reachable, rows 5-6 not
    dist, pred = ds.dijkstra_field(vox, 0, stop_mask=mask, stop_count=10)
    # loop drains: full field over the reachable component
    np.testing.assert_allclose(dist[:5], np.arange(5, dtype=float))
    assert np.isinf(dist[5:]).all()
    # to-set query still reports the first (= nearest) settled anchor
    p, hit, cost = ds.shortest_path_to_set(vox, 0, mask)
    assert hit == 3 and cost == 3.0


def test_source_in_stop_mask_immediate_hit():
    vox = np.array([[x, 0, 0] for x in range(5)], dtype=np.int32)
    mask = np.zeros(5, bool)
    mask[[0, 4]] = True
    p, hit, cost = ds.shortest_path_to_set(vox, 0, mask)
    assert hit == 0 and cost == 0.0
    assert p.tolist() == [[0, 0, 0]]
    # excluding the source from the mask gives the real query
    mask[0] = False
    p, hit, cost = ds.shortest_path_to_set(vox, 0, mask)
    assert hit == 4 and cost == 4.0


def test_unreachable_anchor_returns_empty():
    vox = np.array([[0, 0, 0], [1, 0, 0], [9, 9, 9]], dtype=np.int32)
    mask = np.zeros(3, bool)
    mask[2] = True
    p, hit, cost = ds.shortest_path_to_set(vox, 0, mask)
    assert p.shape == (0, 3) and p.dtype == np.int32
    assert hit == -1 and cost == float("inf")
    p, cost = ds.shortest_path(vox, 0, 2)
    assert p.shape == (0, 3) and cost == float("inf")


def test_multi_source_min_only_composition():
    # field grown from both ends of a line until it first touches the middle anchor
    vox = np.array([[x, 0, 0] for x in range(11)], dtype=np.int32)
    mask = np.zeros(11, bool)
    mask[5] = True
    dist, pred = ds.dijkstra_field(vox, [0, 10], stop_mask=mask)
    assert dist[0] == 0.0 and dist[10] == 0.0
    assert dist[5] == 5.0  # first touch of the anchor set
    assert np.isfinite(dist).sum() == 11  # symmetric growth settles everything <= 5


def test_min_only_false_stops_each_run_independently():
    vox = np.array([[x, 0, 0] for x in range(9)], dtype=np.int32)
    mask = np.zeros(9, bool)
    mask[4] = True
    dist, pred = ds.dijkstra_field(vox, [0, 8], stop_mask=mask, min_only=False)
    assert dist.shape == (2, 9)
    np.testing.assert_allclose(dist[0][4], 4.0)
    np.testing.assert_allclose(dist[1][4], 4.0)
    # each run stopped at the anchor: the far end is untouched in each row
    assert np.isinf(dist[0][8]) and np.isinf(dist[1][0])


def test_shortest_path_point_to_point():
    rng = np.random.default_rng(8)
    vox = random_cloud(rng, 400, box=9)
    full_d, full_p = ds.dijkstra_field(vox, 0)
    target = int(np.flatnonzero(np.isfinite(full_d))[-1])
    p, cost = ds.shortest_path(vox, 0, target)
    np.testing.assert_allclose(cost, full_d[target], rtol=1e-12)
    np.testing.assert_array_equal(p, ds.path(vox, full_p, target))
    assert p.dtype == np.int32


def test_stop_mask_and_free_mask_are_independent():
    # free_mask changes edge costs; stop_mask only changes termination
    vox = np.array([[x, 0, 0] for x in range(8)], dtype=np.int32)
    free = np.zeros(8, bool)
    free[1:5] = True
    stop = np.zeros(8, bool)
    stop[5] = True
    dist, _ = ds.dijkstra_field(vox, 0, free_mask=free, free_eps=1e-6, stop_mask=stop)
    np.testing.assert_allclose(dist[5], 4 * 1e-6 + 1.0, rtol=1e-9)
    assert np.isinf(dist[7])  # stopped before reaching the tail


def test_stop_validation():
    vox = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    with pytest.raises(ValueError, match=r"stop_mask must have shape \(N,\)"):
        ds.dijkstra_field(vox, 0, stop_mask=[True])
    with pytest.raises(ValueError, match="stop_count"):
        ds.dijkstra_field(vox, 0, stop_mask=[True, False], stop_count=-1)
    with pytest.raises(ValueError, match="out of range"):
        ds.shortest_path(vox, 0, 5)


def test_index_kinds_agree_with_stop_mask():
    vox, node_cost, mask = _random_case(9)
    results = [
        ds.dijkstra_field(
            vox, 0, node_cost=node_cost, stop_mask=mask, index_kind=kind, **COST_KW
        )
        for kind in ("sorted", "hash")
    ]
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
