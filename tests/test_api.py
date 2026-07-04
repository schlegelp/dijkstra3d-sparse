"""API surface: dtype handling, edge cases and boundary validation (spec §7.2)."""

import numpy as np
import pytest

import dijkstra3d_sparse as ds


def test_single_voxel():
    vox = np.array([[3, -4, 5]], dtype=np.int32)
    dist, pred = ds.dijkstra_field(vox, 0)
    assert dist.tolist() == [0.0]
    assert pred.tolist() == [-1]
    assert ds.path(vox, pred, 0).tolist() == [[3, -4, 5]]
    assert ds.connected_components(vox) == (1, np.array([0], dtype=np.int32))


def test_empty_inputs():
    vox = np.empty((0, 3), dtype=np.int32)
    dist, pred = ds.dijkstra_field(vox, [])
    assert dist.shape == (0,) and dist.dtype == np.float64
    assert pred.shape == (0,) and pred.dtype == np.int64
    n, labels = ds.connected_components(vox)
    assert n == 0 and labels.shape == (0,)


def test_empty_sources_all_unreached():
    vox = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    dist, pred = ds.dijkstra_field(vox, [])
    assert np.isinf(dist).all()
    assert (pred == -1).all()


def test_scalar_source_and_list_inputs():
    dist, pred = ds.dijkstra_field([[0, 0, 0], [1, 1, 1]], 0)
    np.testing.assert_allclose(dist, [0.0, np.sqrt(3)])


def test_int64_voxels_accepted_and_range_checked():
    vox64 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    dist, _ = ds.dijkstra_field(vox64, 0)
    assert dist[1] == 1.0
    with pytest.raises(ValueError, match="int32 range"):
        ds.dijkstra_field(np.array([[2**40, 0, 0]], dtype=np.int64), 0)


def test_extreme_coordinates_no_overflow():
    m = np.iinfo(np.int32).max
    vox = np.array([[m, m, m], [m - 1, m, m], [-m - 1, -m - 1, -m - 1]], dtype=np.int32)
    dist, _ = ds.dijkstra_field(vox, 0, connectivity=6)
    np.testing.assert_allclose(dist[:2], [0.0, 1.0])
    assert np.isinf(dist[2])
    n, _ = ds.connected_components(vox, connectivity=6)
    assert n == 2


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(connectivity=7), "connectivity"),
        (dict(cost_mode="euclidean"), "cost_mode"),
        (dict(anisotropy=(1.0, 0.0, 1.0)), "anisotropy"),
        (dict(anisotropy=(1.0, -2.0, 1.0)), "anisotropy"),
        (dict(free_eps=0.0), "free_eps"),
        (dict(free_eps=float("nan")), "free_eps"),
        (dict(index_kind="btree"), "index_kind"),
    ],
)
def test_argument_validation(kwargs, match):
    vox = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    with pytest.raises(ValueError, match=match):
        ds.dijkstra_field(vox, 0, **kwargs)


def test_source_out_of_range():
    vox = np.array([[0, 0, 0]], dtype=np.int32)
    with pytest.raises(ValueError, match="out of range"):
        ds.dijkstra_field(vox, 1)
    with pytest.raises(ValueError, match="out of range"):
        ds.dijkstra_field(vox, -1)


def test_node_cost_validation():
    vox = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    with pytest.raises(ValueError, match="finite and non-negative"):
        ds.dijkstra_field(vox, 0, node_cost=[1.0, -0.5])
    with pytest.raises(ValueError, match="finite and non-negative"):
        ds.dijkstra_field(vox, 0, node_cost=[1.0, np.inf])
    with pytest.raises(ValueError, match=r"shape \(N,\)"):
        ds.dijkstra_field(vox, 0, node_cost=[1.0])
    with pytest.raises(ValueError, match=r"shape \(N,\)"):
        ds.dijkstra_field(vox, 0, free_mask=[True])


def test_duplicate_voxels_rejected():
    vox = np.array([[0, 0, 0], [1, 2, 3], [0, 0, 0]], dtype=np.int32)
    for kind in ("sorted", "hash"):
        with pytest.raises(ValueError, match="duplicate"):
            ds.dijkstra_field(vox, 0, index_kind=kind)


def test_bad_voxel_shapes_and_dtypes():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        ds.dijkstra_field(np.zeros((2, 2), dtype=np.int32), 0)
    with pytest.raises(ValueError, match="integer"):
        ds.dijkstra_field(np.zeros((2, 3), dtype=np.float32), 0)


def test_non_contiguous_input_handled():
    # a transposed view is not C-contiguous; the wrapper must cope
    raw = np.array([[0, 1, 2], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    vox = raw.T  # rows: (0,0,0), (1,0,0), (2,0,0)
    dist, _ = ds.dijkstra_field(vox, 0, connectivity=6)
    np.testing.assert_allclose(dist, [0.0, 1.0, 2.0])


def test_index_of():
    vox = np.array([[5, 5, 5], [0, -1, 2], [7, 8, 9]], dtype=np.int32)
    assert ds.index_of(vox, [0, -1, 2]) == 1
    out = ds.index_of(vox, [[7, 8, 9], [5, 5, 5]])
    assert out.tolist() == [2, 0]
    with pytest.raises(KeyError, match="not present"):
        ds.index_of(vox, [1, 1, 1])
    assert ds.index_of(vox, [1, 1, 1], strict=False) == -1


def test_path_input_validation():
    vox = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    _, pred = ds.dijkstra_field(vox, 0)
    with pytest.raises(ValueError, match="out of range"):
        ds.path(vox, pred, 5)
    with pytest.raises(ValueError, match=r"shape \(N,\)"):
        ds.path(vox, pred[:1], 0)
    with pytest.raises(ValueError, match="cycle"):
        ds.path(vox, np.array([1, 0], dtype=np.int64), 0)


def test_node_cost_none_with_explicit_modes_is_geometric():
    vox = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    for mode in ("vertex", "additive", "geometric"):
        dist, _ = ds.dijkstra_field(vox, 0, cost_mode=mode)
        np.testing.assert_allclose(dist, [0.0, 1.0])
