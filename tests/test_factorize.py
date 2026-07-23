"""The ``factorize`` coordinate-deduplication primitive (HANDOFF-factorize).

Dense per-row labels by exact coordinate equality — the sparse
``np.unique(coords, axis=0, return_inverse=True)`` as a hash pass. Unlike every
other primitive here, its input *may contain duplicates*; that is the point.

The oracle is a first-appearance dict (what the labels must match) plus
``np.unique`` for the partition, which pins the equality-grouping as exact.
"""

import numpy as np
import pytest

import dijkstra3d_sparse as ds
from helpers import random_cloud


def reference_factorize(voxels):
    """First-appearance oracle: (n_labels, labels, reps)."""
    voxels = np.asarray(voxels)
    seen = {}
    labels = np.empty(len(voxels), dtype=np.int64)
    reps = []
    for i, v in enumerate(voxels):
        key = tuple(int(c) for c in v)
        if key not in seen:
            seen[key] = len(seen)
            reps.append(i)
        labels[i] = seen[key]
    return len(seen), labels, np.asarray(reps, dtype=np.int64)


# --------------------------------------------------------------------------
# the shapes the handoff calls out by name
# --------------------------------------------------------------------------


def test_all_distinct_is_identity():
    v = np.array([[0, 0, 0], [5, -3, 2], [-7, 8, 9], [1, 2, 3]], dtype=np.int32)
    n, labels, reps = ds.factorize(v, return_index=True)
    assert n == len(v)
    np.testing.assert_array_equal(labels, np.arange(len(v)))
    np.testing.assert_array_equal(reps, np.arange(len(v)))


def test_all_identical_collapses_to_one():
    v = np.tile([3, -4, 5], (6, 1)).astype(np.int32)
    n, labels, reps = ds.factorize(v, return_index=True)
    assert n == 1
    np.testing.assert_array_equal(labels, np.zeros(6, dtype=np.int64))
    np.testing.assert_array_equal(reps, [0])


def test_known_repeats_match_first_appearance_oracle():
    # A=(1,1,1) B=(2,2,2) C=(0,0,0); rows: A B A C B A
    v = np.array(
        [[1, 1, 1], [2, 2, 2], [1, 1, 1], [0, 0, 0], [2, 2, 2], [1, 1, 1]],
        dtype=np.int32,
    )
    n, labels, reps = ds.factorize(v, return_index=True)
    assert n == 3
    np.testing.assert_array_equal(labels, [0, 1, 0, 2, 1, 0])
    np.testing.assert_array_equal(reps, [0, 1, 3])
    # voxels[reps] are exactly the distinct rows in first-appearance order
    np.testing.assert_array_equal(v[reps], [[1, 1, 1], [2, 2, 2], [0, 0, 0]])


def test_return_index_default_is_two_tuple():
    v = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int32)
    out = ds.factorize(v)
    assert isinstance(out, tuple) and len(out) == 2
    n, labels = out
    assert n == 2
    np.testing.assert_array_equal(labels, [0, 0, 1])


# --------------------------------------------------------------------------
# properties: dtype, first-appearance, negatives, reconstruction
# --------------------------------------------------------------------------


def test_negative_and_non_dense_coordinates_roundtrip():
    # far-apart and negative coordinates, no shift assumed anywhere
    v = np.array(
        [[-1000, 5, -7], [0, 0, 0], [-1000, 5, -7], [999999, -3, 2]],
        dtype=np.int32,
    )
    n, labels, reps = ds.factorize(v, return_index=True)
    assert n == 3
    np.testing.assert_array_equal(labels, [0, 1, 0, 2])
    np.testing.assert_array_equal(v[reps][labels], v)


@pytest.mark.parametrize("kind", ["sorted", "hash"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parity_with_np_unique_partition(kind, seed):
    # heavy duplication: draw few distinct coordinates, repeat them a lot
    rng = np.random.default_rng(seed)
    distinct = random_cloud(rng, 120, box=5)  # ~unique small set
    idx = rng.integers(0, len(distinct), size=4000)
    v = distinct[idx].astype(np.int32)  # 4000 rows, ~120 distinct

    n, labels, reps = ds.factorize(v, index_kind=kind, return_index=True)

    # count matches np.unique
    uniq = np.unique(v, axis=0)
    assert n == len(uniq)

    # labels + reps are a consistent factorization: reconstructs v exactly
    np.testing.assert_array_equal(v[reps][labels], v)
    # reps are first-appearances, hence strictly increasing and dense-labelled
    assert np.all(np.diff(reps) > 0)
    np.testing.assert_array_equal(np.unique(labels), np.arange(n))

    # same partition as np.unique's inverse (equality-grouping is exact)
    inv = np.unique(v, axis=0, return_inverse=True)[1].reshape(-1)
    # two rows share a factorize label iff they share a np.unique label
    order = np.lexsort((inv, labels))
    a, b = labels[order], inv[order]
    # within every run of equal `labels`, `inv` is constant, and vice versa
    assert len(np.unique(np.stack([a, b], axis=1), axis=0)) == n


def test_matches_first_appearance_oracle_on_random_cloud():
    rng = np.random.default_rng(9)
    distinct = random_cloud(rng, 200, box=6)
    v = distinct[rng.integers(0, len(distinct), size=3000)].astype(np.int32)
    want_n, want_labels, want_reps = reference_factorize(v)
    got_n, got_labels, got_reps = ds.factorize(v, return_index=True)
    assert got_n == want_n
    np.testing.assert_array_equal(got_labels, want_labels)
    np.testing.assert_array_equal(got_reps, want_reps)


def test_index_kind_gives_identical_result():
    rng = np.random.default_rng(4)
    distinct = random_cloud(rng, 80, box=5)
    v = distinct[rng.integers(0, len(distinct), size=2000)].astype(np.int32)
    a = ds.factorize(v, index_kind="hash", return_index=True)
    b = ds.factorize(v, index_kind="sorted", return_index=True)
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])


def test_accepts_any_integer_dtype():
    base = np.array([[0, 0, 0], [1, 2, 3], [0, 0, 0], [1, 2, 3], [4, 5, 6]], dtype=np.int64)
    want_n, want_labels, _ = ds.factorize(base, return_index=True)
    for dtype in (np.int16, np.uint16, np.int32, np.uint32, np.int64):
        n, labels = ds.factorize(base.astype(dtype))
        assert n == want_n
        np.testing.assert_array_equal(labels, want_labels)


# --------------------------------------------------------------------------
# edge cases and validation
# --------------------------------------------------------------------------


def test_empty_input():
    empty = np.empty((0, 3), dtype=np.int32)
    n, labels = ds.factorize(empty)
    assert n == 0 and labels.shape == (0,) and labels.dtype == np.int64
    n, labels, reps = ds.factorize(empty, return_index=True)
    assert n == 0 and labels.shape == (0,) and reps.shape == (0,) and reps.dtype == np.int64


def test_single_row():
    n, labels, reps = ds.factorize([[7, 8, 9]], return_index=True)
    assert n == 1
    np.testing.assert_array_equal(labels, [0])
    np.testing.assert_array_equal(reps, [0])


def test_input_validation():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        ds.factorize(np.zeros((4, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="integer"):
        ds.factorize(np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="index_kind"):
        ds.factorize([[0, 0, 0]], index_kind="btree")


def test_labels_are_int64_and_row_aligned_under_shuffle():
    rng = np.random.default_rng(12)
    distinct = random_cloud(rng, 100, box=5)
    v = distinct[rng.integers(0, len(distinct), size=1500)].astype(np.int32)
    n, labels = ds.factorize(v)
    assert labels.dtype == np.int64
    # a shuffled input factorizes to the same partition, permuted
    perm = rng.permutation(len(v))
    n2, labels2 = ds.factorize(v[perm])
    assert n2 == n
    # same equivalence classes: rows equal in v iff equal-labelled in both
    same_v = v[perm][:, None, :] == v[perm][None, :, :]
    same_v = same_v.all(axis=2)
    same_lab = labels2[:, None] == labels2[None, :]
    np.testing.assert_array_equal(same_v, same_lab)
