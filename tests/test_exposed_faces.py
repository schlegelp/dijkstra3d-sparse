"""The ``exposed_faces`` surface-extraction primitive (HANDOFF-exposed-faces).

For each voxel, a 6-bit mask of which of its six face-neighbours are *absent*
from the set — the probe every mesh backend runs first, folded into one
coordinate-native call. Bit order is ``+x, -x, +y, -y, +z, -z`` (bits 0..5).

The oracle here is an explicit coordinate-set membership test (the thing the
primitive lets the caller avoid), plus the shipped ``index_of`` probe the
handoff names as the cheapest guarantee that the bit order and the loop agree.
"""

import numpy as np
import pytest

import dijkstra3d_sparse as ds
from helpers import random_cloud

# Face offsets in bit order, matching the native mask and __init__._FACE_OFFSETS.
FACE = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int32,
)
ALL_SIX = 0b111111  # 63


def reference_mask(voxels):
    """Oracle: bit k set iff voxel + FACE[k] is not itself a voxel.

    Steps in unbounded Python ints, never int32: one step off an int32 extreme
    lands outside the representable range, and so is absent by definition —
    it must not wrap around to the far end and match a real voxel there.
    """
    voxels = np.asarray(voxels)
    present = {tuple(int(c) for c in v) for v in voxels}
    mask = np.zeros(len(voxels), dtype=np.uint8)
    for i, v in enumerate(voxels):
        for k, off in enumerate(FACE):
            if tuple(int(v[a]) + int(off[a]) for a in range(3)) not in present:
                mask[i] |= 1 << k
    return mask


# --------------------------------------------------------------------------
# the tiny shapes the handoff calls out by name
# --------------------------------------------------------------------------


def test_single_voxel_all_faces_exposed():
    mask = ds.exposed_faces([[0, 0, 0]])
    assert mask.tolist() == [ALL_SIX]
    assert mask.dtype == np.uint8 and mask.shape == (1,)


def test_pair_along_x_clears_touching_faces():
    # left voxel loses +x (bit 0), right voxel loses -x (bit 1); rest set.
    pair = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    mask = ds.exposed_faces(pair)
    assert mask.tolist() == [ALL_SIX & ~(1 << 0), ALL_SIX & ~(1 << 1)]


@pytest.mark.parametrize("axis,pos_bit,neg_bit", [(0, 0, 1), (1, 2, 3), (2, 4, 5)])
def test_touching_pair_on_each_axis(axis, pos_bit, neg_bit):
    # A pair one step apart along `axis`: the near voxel loses its +axis bit,
    # the far one its -axis bit. Pins the bit order to the axis convention.
    step = np.zeros(3, dtype=np.int32)
    step[axis] = 1
    pair = np.array([[0, 0, 0], step], dtype=np.int32)
    mask = ds.exposed_faces(pair)
    assert mask.tolist() == [ALL_SIX & ~(1 << pos_bit), ALL_SIX & ~(1 << neg_bit)]


def test_fully_surrounded_voxel_is_interior():
    # A voxel with all six face-neighbours present -> mask 0; the neighbours
    # themselves each have exactly the one face touching the centre cleared.
    vox = np.vstack([[0, 0, 0], FACE]).astype(np.int32)
    mask = ds.exposed_faces(vox)
    assert mask[0] == 0
    # each neighbour touches the centre on the face pointing back at it
    for k in range(6):
        opposite = k ^ 1  # (+x,-x) etc. are adjacent bit pairs
        assert mask[1 + k] == ALL_SIX & ~(1 << opposite)


# --------------------------------------------------------------------------
# parity: oracle set-membership, and the shipped index_of probe
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sorted", "hash"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parity_with_membership_oracle(kind, seed):
    rng = np.random.default_rng(seed)
    vox = random_cloud(rng, 500, box=8)
    got = ds.exposed_faces(vox, index_kind=kind)
    np.testing.assert_array_equal(got, reference_mask(vox))


@pytest.mark.parametrize("kind", ["sorted", "hash"])
def test_parity_with_index_of_probe_per_bit(kind):
    # The handoff's cheapest guarantee that the bit order and the loop agree:
    # bit k reproduces exactly `index_of(voxels + FACE[k]) < 0`.
    rng = np.random.default_rng(7)
    vox = random_cloud(rng, 600, box=9)
    mask = ds.exposed_faces(vox, index_kind=kind)
    for k in range(6):
        got = ((mask >> k) & 1).astype(bool)
        want = ds.index_of(vox, vox + FACE[k], strict=False) < 0
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("shuffle", [False, True])
def test_parity_on_a_dense_cloud_across_all_paths(shuffle):
    # Dense enough that most voxels have several face-neighbours, which is
    # what exercises the sorted merge sweep's cursor (the tiny shapes above
    # do not). All four code paths must land on the same mask.
    rng = np.random.default_rng(21)
    vox = np.unique(rng.integers(0, 12, size=(2000, 3)).astype(np.int32), axis=0)
    if shuffle:
        vox = vox[rng.permutation(len(vox))]
    want = reference_mask(vox)
    for got in (
        ds.exposed_faces(vox, index_kind="hash"),
        ds.exposed_faces(vox, index_kind="sorted"),
        ds.Graph(vox, index_kind="hash").exposed_faces(),
        ds.Graph(vox, index_kind="sorted").exposed_faces(),
    ):
        np.testing.assert_array_equal(got, want)


def test_int32_extremes_do_not_wrap_between_axes():
    # A neighbour one step past an axis maximum has no int32 coordinate, so
    # that face is exposed. The trap: the packed key steps by a constant, so
    # (x, INT32_MAX, z) + (0, 1, 0) must not be read as (x + 1, INT32_MIN, z).
    imax, imin = np.iinfo(np.int32).max, np.iinfo(np.int32).min
    vox = np.array(
        [
            [imax, 0, 0], [imax - 1, 0, 0],   # +x extreme, touched from below
            [imin, 0, 0], [imin + 1, 0, 0],   # -x extreme, touched from above
            [3, imax, 9], [4, imin, 9],       # the y -> x carry trap
            [3, 5, imax], [3, 6, imin],       # the z -> y carry trap
        ],
        dtype=np.int32,
    )
    want = reference_mask(vox)
    for kind in ("hash", "sorted"):
        np.testing.assert_array_equal(ds.exposed_faces(vox, index_kind=kind), want)
        np.testing.assert_array_equal(ds.Graph(vox, index_kind=kind).exposed_faces(), want)
    # the extremes themselves: +x exposed at INT32_MAX, -x exposed at INT32_MIN
    assert want[0] & 0b11 == 0b01
    assert want[2] & 0b11 == 0b10
    # nothing may have paired the y/z carry rows with each other
    assert want[4] & (1 << 2) and want[5] & (1 << 3)
    assert want[6] & (1 << 4) and want[7] & (1 << 5)


def test_caller_directional_split_matches_sets():
    # §3: the surface pass splits the mask into six directional voxel sets.
    # Reproduce that split and check each set against direct membership.
    rng = np.random.default_rng(11)
    vox = random_cloud(rng, 400, box=7)
    mask = ds.exposed_faces(vox)
    present = {tuple(int(c) for c in v) for v in vox}
    for k in range(6):
        got = vox[(mask & (1 << k)) != 0]
        want = np.array(
            [v for v in vox if tuple(int(v[a] + FACE[k][a]) for a in range(3)) not in present],
            dtype=vox.dtype,
        ).reshape(-1, 3)
        # both are subsequences of `vox` in row order, so compare directly
        np.testing.assert_array_equal(got, want)


# --------------------------------------------------------------------------
# Graph method parity and reuse
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sorted", "hash"])
def test_graph_method_parity(kind):
    rng = np.random.default_rng(3)
    vox = np.vstack(
        [random_cloud(rng, 300, box=8), random_cloud(rng, 300, box=8, offset=(50, 0, 0))]
    )
    g = ds.Graph(vox, index_kind=kind)
    np.testing.assert_array_equal(
        g.exposed_faces(), ds.exposed_faces(vox, index_kind=kind)
    )


def test_graph_method_reuse_is_stable():
    # repeated calls on one handle, interleaved with another query, all agree
    rng = np.random.default_rng(4)
    vox = random_cloud(rng, 300, box=6)
    g = ds.Graph(vox)
    want = ds.exposed_faces(vox)
    first = g.exposed_faces()
    g.connected_components()  # unrelated query in between
    np.testing.assert_array_equal(first, want)
    np.testing.assert_array_equal(g.exposed_faces(), want)


# --------------------------------------------------------------------------
# edge cases and validation
# --------------------------------------------------------------------------


def test_empty_input():
    for src in (np.empty((0, 3), dtype=np.int32), ds.Graph(np.empty((0, 3), dtype=np.int32))):
        out = src.exposed_faces() if isinstance(src, ds.Graph) else ds.exposed_faces(src)
        assert out.shape == (0,) and out.dtype == np.uint8


def test_single_voxel_graph():
    g = ds.Graph([[3, -4, 5]])
    assert g.exposed_faces().tolist() == [ALL_SIX]


def test_duplicate_coordinate_raises():
    dup = np.array([[0, 0, 0], [1, 2, 3], [0, 0, 0]], dtype=np.int32)
    for kind in ("sorted", "hash"):
        with pytest.raises(ValueError, match="duplicate"):
            ds.exposed_faces(dup, index_kind=kind)
    # and via the constructor, as the caller's fallback relies on
    with pytest.raises(ValueError, match="duplicate"):
        ds.Graph(dup)


def test_input_validation():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        ds.exposed_faces(np.zeros((2, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="integer"):
        ds.exposed_faces(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="index_kind"):
        ds.exposed_faces([[0, 0, 0]], index_kind="btree")


def test_accepts_any_integer_dtype_and_is_row_aligned():
    # uint32 is what mesh() passes; the -1 step must not wrap it.
    base = random_cloud(np.random.default_rng(9), 200, box=6)
    want = ds.exposed_faces(base)
    for dtype in (np.int16, np.uint16, np.int32, np.uint32, np.int64):
        got = ds.exposed_faces(base.astype(dtype))
        np.testing.assert_array_equal(got, want)

    # a shuffled input yields a correspondingly shuffled mask (row-aligned)
    rng = np.random.default_rng(10)
    perm = rng.permutation(len(base))
    np.testing.assert_array_equal(ds.exposed_faces(base[perm]), want[perm])
