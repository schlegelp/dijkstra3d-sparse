"""The ``ray_exits`` ray-casting primitive (HANDOFF-ray-exits).

Where a ray leaves the object: a 3-D DDA (Amanatides & Woo) over the sparse
voxel set, reporting the parametric ``t`` of every occupancy flip. Cell ``c``
spans ``[c - 0.5, c + 0.5)``, so a ball of radius ``R`` exits at exactly
``R + 0.5`` along an axis — most of the analytic cases below need no tolerance
at all.

The oracles are independent by construction: a fixed-step *sampler* against a
Python ``set`` (sharing no arithmetic, no packing and no data structure with
the DDA), and a run-length walk of the coordinate set for the axis-aligned
cases. The sampler comparison is deliberately one-sided — a sampler can only
*miss* a crossing (a ray clipping a cell corner can stay inside it for less
than one step), never invent one — so it is asserted that way.
"""

import math

import numpy as np
import pytest

import dijkstra3d_sparse as ds
from helpers import random_cloud

STEP = 1e-4  # sampler resolution


def cell_of(p):
    """The cell containing point ``p`` — ``floor(p + 0.5)``, per the half-open
    ``[c - 0.5, c + 0.5)`` convention (``round`` would misplace ``-0.5``)."""
    return tuple(math.floor(float(c) + 0.5) for c in p)


def as_set(voxels):
    return {tuple(int(c) for c in v) for v in np.asarray(voxels)}


# The two samplers below run ~10^5 steps per ray, so their inner loops stay in
# plain Python floats: one NumPy scalar op per sample would cost more than the
# whole rest of the suite. They still share no arithmetic, no packing and no
# data structure with the DDA — the point of an independent oracle.


def _unpack(o, d):
    """(o, d) as six plain floats, hoisted out of the sampling loop."""
    o, d = np.asarray(o, float), np.asarray(d, float)
    return (*(float(c) for c in o), *(float(c) for c in d))


def sampled_crossings(present, o, d, max_dist, k=8, step=STEP):
    """Brute-force oracle: march in fixed steps, report the ``t`` of the first
    sample whose cell occupancy differs from the previous one. The true
    crossing lies in ``(t - step, t]``."""
    ox, oy, oz, dx, dy, dz = _unpack(o, d)

    def at(t):
        return (
            math.floor(ox + t * dx + 0.5),
            math.floor(oy + t * dy + 0.5),
            math.floor(oz + t * dz + 0.5),
        )

    inside = at(0.0) in present
    out, t = [], 0.0
    while t <= max_dist and len(out) < k:
        t += step
        now = at(t) in present
        if now != inside:
            out.append(t)
            inside = now
    return out


def sampled_path(o, d, max_dist, step=STEP):
    """``[(cell, t_entry), ...]`` for the cells the segment passes through, in
    order — the DDA's cell path, obtained by sampling instead of stepping. The
    origin cell comes first, at ``t = 0``; each later ``t_entry`` is the first
    sample inside that cell, so the true entry lies in ``(t - step, t]``."""
    ox, oy, oz, dx, dy, dz = _unpack(o, d)

    def at(t):
        return (
            math.floor(ox + t * dx + 0.5),
            math.floor(oy + t * dy + 0.5),
            math.floor(oz + t * dz + 0.5),
        )

    path, t = [(at(0.0), 0.0)], 0.0
    while t <= max_dist:
        t += step
        c = at(t)
        if c != path[-1][0]:
            path.append((c, t))
    return path


def random_rays(rng, vox, n, generic=False):
    """``n`` rays starting inside random voxels of ``vox``. With ``generic``,
    the directions are nudged off the axes: a ray aimed exactly at a cell
    corner is a measure-zero tie the samplers above cannot resolve."""
    origins = vox[rng.integers(0, len(vox), n)] + rng.uniform(-0.4, 0.4, (n, 3))
    dirs = rng.uniform(-1, 1, (n, 3))
    if generic:
        dirs = dirs + np.array([0.013, 0.027, 0.041])
    return origins, dirs


def ball(r):
    """A solid rasterized ball: every cell whose centre is within ``r``."""
    g = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1].reshape(3, -1).T
    return g[(g**2).sum(1) <= r * r].astype(np.int32)


def cube(lo, hi):
    """A solid axis-aligned block, inclusive bounds per axis."""
    g = np.mgrid[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1]
    return g.reshape(3, -1).T.astype(np.int32)


AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)


# --------------------------------------------------------------------------
# analytic: exits with no tolerance at all
# --------------------------------------------------------------------------


@pytest.mark.parametrize("r", [1, 4, 9])
def test_ball_exits_on_the_six_axes_are_exactly_r_plus_half(r):
    vox = ball(r)
    t, n_hits = ds.ray_exits(vox, np.zeros((6, 3)), AXES, max_dist=100.0)
    assert t.shape == (6, 1) and t.dtype == np.float64
    assert n_hits.dtype == np.int32
    np.testing.assert_array_equal(n_hits, np.ones(6, dtype=np.int32))
    np.testing.assert_array_equal(t[:, 0], np.full(6, r + 0.5))


def test_cube_body_diagonal_exits_at_the_far_face():
    # Along (1, 1, 1) all three axes cross simultaneously, so the exit is the
    # far face of the last cell: L + 0.5, exactly.
    vox = cube((-5, -5, -5), (5, 5, 5))
    t, n_hits = ds.ray_exits(vox, [[0, 0, 0]], [[1, 1, 1]], max_dist=100.0)
    assert n_hits[0] == 1
    assert t[0, 0] == 5.5


def test_directions_are_index_space_and_need_not_be_unit():
    # §2: the caller divides a physically-unit direction by the voxel spacing,
    # so `t` comes back in physical units — no anisotropy parameter here.
    vox = ball(5)
    spacing = np.array([2.0, 2.0, 8.0])  # anisotropic voxels
    unit = np.array([[1.0, 0, 0], [0, 0, 1.0]])  # physical directions
    t, _ = ds.ray_exits(vox, np.zeros((2, 3)), unit / spacing, max_dist=1000.0)
    # 5.5 cells along x is 11 physical units; along z it is 44
    np.testing.assert_array_equal(t[:, 0], [11.0, 44.0])
    # halving a direction doubles t; the geometry is untouched
    t2, _ = ds.ray_exits(vox, [[0, 0, 0]], [[0.5, 0, 0]], max_dist=100.0)
    assert t2[0, 0] == 11.0


def test_origin_on_a_cell_boundary_does_not_step_backwards():
    # -0.5 is the first point of cell 0 (half-open), and +0.5 the first point
    # of cell 1 — the exit is measured from there, not from a rounded centre.
    vox = ball(4)
    t, _ = ds.ray_exits(
        vox,
        [[0.5, 0, 0], [0.5, 0, 0], [-0.5, 0, 0]],
        [[1, 0, 0], [-1, 0, 0], [-1, 0, 0]],
        max_dist=100.0,
    )
    np.testing.assert_array_equal(t[:, 0], [4.0, 5.0, 4.0])


# --------------------------------------------------------------------------
# a real DDA, not a sampler: every cell on the path is visited
# --------------------------------------------------------------------------


def test_every_cell_on_the_path_is_visited():
    # Punch a single-voxel hole in a solid block at each cell of the ray's
    # path in turn: the exit must move to the entry face of that very cell.
    # A sampler — which is what this primitive must *not* be — would step over
    # some of them. Nothing else in the suite would catch that regression.
    block = cube((-2, -6, -6), (14, 6, 6))
    o, d = np.array([0.3, 0.1, -0.2]), np.array([1.0, 0.31, 0.17])
    cap = 20.0
    path = sampled_path(o, d, cap)

    # the solid block's own exit, through its +x face at x = 14.5
    base_t = ds.ray_exits(block, [o], [d], max_dist=cap)[0][0, 0]
    assert np.isfinite(base_t)

    checked = 0
    for cell, want in path[1:]:
        # stop short of the block's own face, where a hole would merge with it
        if want >= base_t - 1.5:
            break
        pruned = block[~np.all(block == np.asarray(cell), axis=1)]
        assert len(pruned) == len(block) - 1
        t, n_hits = ds.ray_exits(pruned, [o], [d], max_dist=cap, max_crossings=2)
        assert n_hits[0] == 2, f"the hole at {cell} was stepped over"
        # exit into the hole, then straight back in one cell later
        assert want - STEP < t[0, 0] <= want
        assert t[0, 1] > t[0, 0]
        checked += 1
    assert checked > 8, f"only {checked} cells on the path were probed"


def test_cells_adjacent_to_the_path_are_not_visited():
    # The converse: a hole *next to* the path must not perturb the exit at
    # all, so the walk is not merely a superset of the true cell sequence.
    block = cube((-2, -6, -6), (14, 6, 6))
    o, d = np.array([0.3, 0.1, -0.2]), np.array([1.0, 0.31, 0.17])
    cap = 20.0
    walked = [cell for cell, _ in sampled_path(o, d, cap)]
    path = set(walked)
    base_t = ds.ray_exits(block, [o], [d], max_dist=cap)[0][0, 0]
    assert np.isfinite(base_t)

    off_path = [
        (x, y + 2, z + 2)
        for (x, y, z) in walked[:12]
        if (x, y + 2, z + 2) not in path and abs(y + 2) <= 6 and abs(z + 2) <= 6
    ]
    assert len(off_path) > 5
    for cell in off_path:
        pruned = block[~np.all(block == np.asarray(cell), axis=1)]
        t, n_hits = ds.ray_exits(pruned, [o], [d], max_dist=cap)
        assert n_hits[0] == 1 and t[0, 0] == base_t, f"{cell} was probed but is off-path"


# --------------------------------------------------------------------------
# parity with the brute-force sampler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_first_exit_brackets_the_sampler_on_a_random_cloud(seed):
    rng = np.random.default_rng(seed)
    vox = random_cloud(rng, 900, box=10)
    present = as_set(vox)
    cap = 15.0

    origins, dirs = random_rays(rng, vox, 60, generic=True)

    t, n_hits = ds.ray_exits(vox, origins, dirs, max_dist=cap)
    for i, (o, d) in enumerate(zip(origins, dirs)):
        want = sampled_crossings(present, o, d, cap, k=1)
        if not want or want[0] > cap:
            continue
        assert n_hits[i] == 1, f"ray {i} missed the exit the sampler found"
        # the sampler reports the first sample *past* the crossing
        assert want[0] - STEP < t[i, 0] <= want[0]


@pytest.mark.parametrize("seed", [3, 4])
def test_all_crossings_bracket_the_sampler(seed):
    rng = np.random.default_rng(seed)
    vox = random_cloud(rng, 700, box=9)
    present = as_set(vox)
    cap, k = 10.0, 8
    # a coarser sampler here: this test walks every crossing of every ray, and
    # the assertion is a bracket, so the step only sets how wide it is
    step = 5e-4

    origins, dirs = random_rays(rng, vox, 24, generic=True)
    t, n_hits = ds.ray_exits(vox, origins, dirs, max_dist=cap, max_crossings=k)

    for i, (o, d) in enumerate(zip(origins, dirs)):
        got = t[i, : n_hits[i]]
        want = [w for w in sampled_crossings(present, o, d, cap, k=k, step=step) if w <= cap]
        # one-sided: the sampler can only miss crossings, never invent them
        assert len(got) >= len(want) or n_hits[i] == k
        for w in want[: n_hits[i]]:
            assert any(w - step < g <= w for g in got), f"ray {i}: nothing brackets {w}"
        # crossings are strictly increasing
        assert np.all(np.diff(got) > 0)
        # ...and everything past n_hits is +inf padding
        assert np.all(np.isinf(t[i, n_hits[i] :]))


def test_axis_aligned_exits_match_a_run_length_oracle():
    # Along an axis no corner ties are possible, so this one is exact: the
    # exit is the far face of the last cell of the occupied run.
    rng = np.random.default_rng(5)
    vox = random_cloud(rng, 800, box=9)
    present = as_set(vox)
    origins = vox.astype(np.float64)

    for axis, sign in [(0, 1), (0, -1), (1, 1), (2, -1)]:
        d = np.zeros(3)
        d[axis] = sign
        t, n_hits = ds.ray_exits(vox, origins, np.tile(d, (len(vox), 1)), max_dist=50.0)
        want = np.empty(len(vox))
        for i, v in enumerate(vox):
            steps = 1
            while tuple(int(c) for c in (v + steps * d)) in present:
                steps += 1
            want[i] = steps - 0.5
        np.testing.assert_array_equal(n_hits, np.ones(len(vox), dtype=np.int32))
        np.testing.assert_array_equal(t[:, 0], want)


# --------------------------------------------------------------------------
# alternation, max_dist, max_crossings
# --------------------------------------------------------------------------


def test_alternation_across_two_slabs():
    # Slabs [0, 2] and [6, 8] along x: exit 2.5, re-entry 5.5, exit 8.5.
    vox = np.array([[x, 0, 0] for x in [0, 1, 2, 6, 7, 8]], dtype=np.int32)
    t, n_hits = ds.ray_exits(vox, [[0, 0, 0]], [[1, 0, 0]], max_dist=100.0, max_crossings=4)
    assert n_hits[0] == 3
    np.testing.assert_array_equal(t[0], [2.5, 5.5, 8.5, np.inf])


def test_max_crossings_truncates_but_never_changes_the_prefix():
    vox = np.array([[x, 0, 0] for x in [0, 1, 2, 6, 7, 8, 12]], dtype=np.int32)
    full, n_full = ds.ray_exits(vox, [[0, 0, 0]], [[1, 0, 0]], max_dist=100.0, max_crossings=8)
    for k in (1, 2, 3, 4, 8):
        t, n_hits = ds.ray_exits(vox, [[0, 0, 0]], [[1, 0, 0]], max_dist=100.0, max_crossings=k)
        assert t.shape == (1, k)
        assert n_hits[0] == min(k, n_full[0])
        np.testing.assert_array_equal(t[0, : n_hits[0]], full[0, : n_hits[0]])


def test_max_dist_bounds_t_and_is_per_ray():
    # A corridor longer than the cap: the ray "escapes" (n_hits == 0) and the
    # walk stops at the cap rather than running to the far end.
    corridor = np.array([[x, 0, 0] for x in range(2000)], dtype=np.int32)
    t, n_hits = ds.ray_exits(corridor, [[0, 0, 0]], [[1, 0, 0]], max_dist=50.0)
    assert n_hits[0] == 0 and np.isinf(t[0, 0])
    t, n_hits = ds.ray_exits(corridor, [[0, 0, 0]], [[1, 0, 0]], max_dist=5000.0)
    assert n_hits[0] == 1 and t[0, 0] == 1999.5

    # per-ray caps clip different rays at different points in one batch
    vox = ball(6)
    caps = np.array([100.0, 2.0, 6.5, 6.4])
    t, n_hits = ds.ray_exits(vox, np.zeros((4, 3)), np.tile([1.0, 0, 0], (4, 1)), max_dist=caps)
    np.testing.assert_array_equal(n_hits, [1, 0, 1, 0])  # 6.5 is inclusive
    np.testing.assert_array_equal(t[:, 0], [6.5, np.inf, 6.5, np.inf])


# --------------------------------------------------------------------------
# backend parity and the Graph handle
# --------------------------------------------------------------------------


def test_backends_are_bit_identical():
    # They differ only in how a coordinate is looked up, so this is
    # assert_array_equal, never allclose (tests/test_graph.py's rule).
    rng = np.random.default_rng(6)
    vox = random_cloud(rng, 800, box=10)
    origins, dirs = random_rays(rng, vox, 200)
    a = ds.ray_exits(vox, origins, dirs, max_dist=20.0, max_crossings=4, index_kind="hash")
    b = ds.ray_exits(vox, origins, dirs, max_dist=20.0, max_crossings=4, index_kind="sorted")
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("kind", ["sorted", "hash"])
def test_graph_method_parity(kind):
    rng = np.random.default_rng(7)
    vox = random_cloud(rng, 600, box=9)
    origins, dirs = random_rays(rng, vox, 120)
    want = ds.ray_exits(vox, origins, dirs, max_dist=15.0, max_crossings=3, index_kind=kind)
    got = ds.Graph(vox, index_kind=kind).ray_exits(origins, dirs, max_dist=15.0, max_crossings=3)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def test_graph_handle_serves_many_chunks_unchanged():
    # The motivating use: one handle, rays fired chunk by chunk. Chunking must
    # not change a single value, and an unrelated query in between must not
    # disturb the index.
    rng = np.random.default_rng(8)
    vox = random_cloud(rng, 500, box=8)
    g = ds.Graph(vox)
    origins, dirs = random_rays(rng, vox, 90)

    whole_t, whole_n = g.ray_exits(origins, dirs, max_dist=12.0, max_crossings=2)
    g.connected_components()  # unrelated query in between
    chunks = [
        g.ray_exits(origins[i : i + 17], dirs[i : i + 17], max_dist=12.0, max_crossings=2)
        for i in range(0, len(origins), 17)
    ]
    np.testing.assert_array_equal(np.vstack([c[0] for c in chunks]), whole_t)
    np.testing.assert_array_equal(np.concatenate([c[1] for c in chunks]), whole_n)


# --------------------------------------------------------------------------
# degenerate input and validation
# --------------------------------------------------------------------------


def test_zero_direction_components_and_all_zero_direction():
    vox = ball(3)
    t, n_hits = ds.ray_exits(
        vox,
        np.zeros((3, 3)),
        [[0, 1, 0], [0, 0, 0], [0, 0, -1]],
        max_dist=100.0,
    )
    # a zero component never crosses that axis; an all-zero direction reports
    # nothing at all (and must terminate, not spin)
    np.testing.assert_array_equal(n_hits, [1, 0, 1])
    np.testing.assert_array_equal(t[:, 0], [3.5, np.inf, 3.5])


def test_origin_in_an_empty_cell_reports_nothing():
    # §2: the origin cell is assumed occupied and never reported; if it is
    # empty the ray has nothing to exit, which reads the same as "escaped".
    vox = ball(3)
    t, n_hits = ds.ray_exits(
        vox, [[50, 0, 0], [4, 0, 0]], [[1, 0, 0], [-1, 0, 0]], max_dist=100.0
    )
    np.testing.assert_array_equal(n_hits, [0, 0])
    assert np.all(np.isinf(t))


def test_empty_batch_and_empty_voxel_set():
    vox = ball(2)
    empty = np.empty((0, 3))
    t, n_hits = ds.ray_exits(vox, empty, empty, max_dist=1.0, max_crossings=3)
    assert t.shape == (0, 3) and t.dtype == np.float64
    assert n_hits.shape == (0,) and n_hits.dtype == np.int32

    t, n_hits = ds.ray_exits(np.empty((0, 3), np.int32), [[0, 0, 0]], [[1, 0, 0]], max_dist=5.0)
    assert n_hits[0] == 0 and np.isinf(t[0, 0])

    g = ds.Graph(np.empty((0, 3), np.int32))
    t, n_hits = g.ray_exits(empty, empty, max_dist=1.0)
    assert t.shape == (0, 1) and n_hits.shape == (0,)


def test_int32_extremes_stop_instead_of_wrapping():
    # Both extremes are occupied, so a wrap would show up as a spurious
    # re-entry immediately after the exit.
    imax, imin = np.iinfo(np.int32).max, np.iinfo(np.int32).min
    vox = np.array([[imax, 0, 0], [imin, 0, 0]], dtype=np.int32)
    origins = np.array([[imax, 0, 0], [imin, 0, 0]], dtype=np.float64)
    dirs = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    for kind in ("hash", "sorted"):
        t, n_hits = ds.ray_exits(
            vox, origins, dirs, max_dist=1e18, max_crossings=4, index_kind=kind
        )
        # the step out of the lone voxel is a real exit (no voxel can live
        # outside the int32 range) and is reported; nothing beyond it is
        np.testing.assert_array_equal(n_hits, [1, 1])
        np.testing.assert_array_equal(t[:, 0], [0.5, 0.5])
        assert np.all(np.isinf(t[:, 1:]))


def test_integer_ray_arrays_are_accepted():
    vox = ball(4)
    t_int, n_int = ds.ray_exits(vox, np.zeros((1, 3), np.int64), np.array([[1, 0, 0]]), max_dist=9)
    t_flt, n_flt = ds.ray_exits(vox, np.zeros((1, 3)), np.array([[1.0, 0, 0]]), max_dist=9.0)
    np.testing.assert_array_equal(t_int, t_flt)
    np.testing.assert_array_equal(n_int, n_flt)


def test_non_finite_input_is_rejected():
    vox = ball(2)
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="origins must be finite"):
            ds.ray_exits(vox, [[bad, 0, 0]], [[1, 0, 0]], max_dist=5.0)
        with pytest.raises(ValueError, match="directions must be finite"):
            ds.ray_exits(vox, [[0, 0, 0]], [[bad, 0, 0]], max_dist=5.0)
        with pytest.raises(ValueError, match="max_dist must be finite"):
            ds.ray_exits(vox, [[0, 0, 0]], [[1, 0, 0]], max_dist=bad)


def test_input_validation():
    vox = ball(2)
    with pytest.raises(ValueError, match=r"origins must have shape \(R, 3\)"):
        ds.ray_exits(vox, [0, 0, 0], [[1, 0, 0]], max_dist=5.0)
    with pytest.raises(ValueError, match=r"directions must have shape \(R, 3\)"):
        ds.ray_exits(vox, [[0, 0, 0]], [[1, 0]], max_dist=5.0)
    with pytest.raises(ValueError, match="same length"):
        ds.ray_exits(vox, [[0, 0, 0]], [[1, 0, 0], [1, 0, 0]], max_dist=5.0)
    with pytest.raises(ValueError, match=r"max_dist must be a scalar or have shape"):
        ds.ray_exits(vox, [[0, 0, 0]], [[1, 0, 0]], max_dist=[1.0, 2.0])
    with pytest.raises(ValueError, match="max_dist must be finite and >= 0"):
        ds.ray_exits(vox, [[0, 0, 0]], [[1, 0, 0]], max_dist=-1.0)
    with pytest.raises(ValueError, match="max_crossings must be >= 1"):
        ds.ray_exits(vox, [[0, 0, 0]], [[1, 0, 0]], max_dist=5.0, max_crossings=0)
    with pytest.raises(ValueError, match="index_kind"):
        ds.ray_exits(vox, [[0, 0, 0]], [[1, 0, 0]], max_dist=5.0, index_kind="btree")
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        ds.ray_exits(np.zeros((2, 2), np.int32), [[0, 0, 0]], [[1, 0, 0]], max_dist=5.0)
    # and the same checks on the handle
    g = ds.Graph(vox)
    with pytest.raises(ValueError, match="max_crossings must be >= 1"):
        g.ray_exits([[0, 0, 0]], [[1, 0, 0]], max_dist=5.0, max_crossings=0)


def test_duplicate_coordinates_are_rejected():
    dup = np.array([[0, 0, 0], [1, 2, 3], [0, 0, 0]], dtype=np.int32)
    for kind in ("sorted", "hash"):
        with pytest.raises(ValueError, match="duplicate"):
            ds.ray_exits(dup, [[0, 0, 0]], [[1, 0, 0]], max_dist=5.0, index_kind=kind)


def test_is_deterministic():
    rng = np.random.default_rng(9)
    vox = random_cloud(rng, 400, box=8)
    origins, dirs = random_rays(rng, vox, 50)
    first = ds.ray_exits(vox, origins, dirs, max_dist=15.0, max_crossings=3)
    for _ in range(3):
        again = ds.ray_exits(vox, origins, dirs, max_dist=15.0, max_crossings=3)
        np.testing.assert_array_equal(again[0], first[0])
        np.testing.assert_array_equal(again[1], first[1])


def test_row_order_of_voxels_does_not_matter():
    # The primitive is a pure function of the *set*: shuffling the rows of
    # `voxels` (which changes every row index) must not move a crossing.
    rng = np.random.default_rng(10)
    vox = random_cloud(rng, 500, box=8)
    origins, dirs = random_rays(rng, vox, 60)
    want = ds.ray_exits(vox, origins, dirs, max_dist=15.0, max_crossings=3)
    shuffled = vox[rng.permutation(len(vox))]
    got = ds.ray_exits(shuffled, origins, dirs, max_dist=15.0, max_crossings=3)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


# --------------------------------------------------------------------------
# the caller's shape (§3 of the handoff)
# --------------------------------------------------------------------------


def test_caller_radius_escaped_reentered_pattern():
    # What `sparsecubes/tube.py::_raycast` collapses to: one call, then three
    # cheap numpy reads. A ball is star-shaped, so nothing re-enters; a ring
    # in the z = 0 plane, cast from its centre, escapes through the hole.
    vox = ball(6)
    n_theta = 64
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    dirs = np.stack([np.cos(theta), np.sin(theta), np.zeros(n_theta)], axis=1)
    origins = np.zeros((n_theta, 3))
    caps = np.full(n_theta, 20.0)

    t, n_hits = ds.ray_exits(vox, origins, dirs, max_dist=caps, max_crossings=2)
    radius = np.where(n_hits > 0, t[:, 0], caps)
    escaped = n_hits == 0
    reentered = n_hits > 1
    assert not escaped.any() and not reentered.any()
    # the exit of a rasterized disc of radius 6 wobbles by about a cell with
    # the angle — along the axes it is exactly 6.5, on the diagonals less
    assert np.all((radius > 5.0) & (radius <= 6.5))
    assert radius[0] == 6.5  # theta = 0 is the +x axis

    # now hollow out the centre: the rays start outside the object, so they
    # report nothing — the case the caller flags per node with one index_of
    shell = vox[(vox**2).sum(1) > 9]
    t, n_hits = ds.ray_exits(shell, origins, dirs, max_dist=caps, max_crossings=2)
    assert np.all(n_hits == 0)
