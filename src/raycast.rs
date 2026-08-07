//! Ray casting over the implicit sparse grid: where a ray crosses the
//! object's boundary.
//!
//! Amanatides & Woo (1987), one ray at a time, straight against the
//! `SpatialIndex` the rest of the library already builds. Voxel centres sit
//! on integer coordinates, so cell `c` occupies `[c - 0.5, c + 0.5)` on each
//! axis; the walk visits exactly the cells the ray passes through — no
//! sampling, no interpolation — and records the parametric `t` at which
//! occupancy flips.
//!
//! The entire per-ray state is thirteen scalars that live in registers, and
//! the output slots are the caller's buffer, so the hot loop allocates
//! nothing. That is the whole point of doing this here: a vectorized DDA has
//! to keep the live ray set in `(R, 3)` arrays and touch all of it to advance
//! any single ray by one cell, which costs an order of magnitude more than
//! the index probes it wraps.

use crate::index::{key_of, SpatialIndex};

/// One batch of rays, borrowed from the caller's arrays. `origins` and
/// `directions` are row-major `(R, 3)`; `max_dist` holds one cap per ray (the
/// caller scales it per ray, so a single global value is not enough) and so
/// its length *is* `R`.
pub struct Rays<'a> {
    pub origins: &'a [f64],
    pub directions: &'a [f64],
    pub max_dist: &'a [f64],
    /// Output slots per ray, and equally the stopping rule for the walk.
    pub max_crossings: usize,
}

/// Index of the smallest of three values; ties go to the lowest axis, which
/// is what makes the walk deterministic on rays aimed exactly at an edge or
/// a corner.
#[inline(always)]
fn argmin3(v: &[f64; 3]) -> usize {
    let a = if v[0] <= v[1] { 0 } else { 1 };
    if v[a] <= v[2] {
        a
    } else {
        2
    }
}

/// The cell containing `p` on one axis, as a float (the range check happens
/// at the caller). Cell `c` spans `[c - 0.5, c + 0.5)`, so this is
/// `floor(p + 0.5)` and *not* `round`: rounding goes half away from zero, so
/// it would put `-0.5` — the first point of cell `0` — in cell `-1`.
#[inline(always)]
fn cell_of(p: f64) -> f64 {
    (p + 0.5).floor()
}

/// Walk one ray; write its crossings into `out` (in increasing `t`) and
/// return how many there were.
///
/// A crossing is a `t` at which the ray moves from an occupied cell to an
/// empty one or back, so the values strictly alternate: `out[0]` is the first
/// exit, `out[1]` the first re-entry after it, and so on. Walking stops at
/// `out.len()` crossings or at `max_dist`, whichever comes first; a ray that
/// reaches `max_dist` without crossing returns 0.
///
/// The origin cell is the walk's starting state and is never itself reported.
/// It is assumed occupied — one probe checks that, outside the loop, and a
/// ray starting in empty space returns 0 (there is nothing to exit).
pub fn ray_exits_one(
    origin: [f64; 3],
    dir: [f64; 3],
    max_dist: f64,
    index: &SpatialIndex,
    out: &mut [f64],
) -> i32 {
    let mut cur = [0i32; 3];
    let mut step = [0i32; 3];
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];

    for a in 0..3 {
        let c = cell_of(origin[a]);
        // Also the NaN guard: every comparison against NaN is false, so a
        // non-finite origin leaves here instead of entering the loop with a
        // saturated cell index.
        if !(c >= i32::MIN as f64 && c <= i32::MAX as f64) {
            return 0;
        }
        cur[a] = c as i32;
        if dir[a] != 0.0 {
            step[a] = if dir[a] > 0.0 { 1 } else { -1 };
            t_delta[a] = (1.0 / dir[a]).abs();
            // Next face crossing on this axis. Clamped at 0 so an origin
            // sitting exactly on a cell boundary cannot step backwards.
            t_max[a] = ((cur[a] as f64 + 0.5 * step[a] as f64) - origin[a]) / dir[a];
            if t_max[a] < 0.0 {
                t_max[a] = 0.0;
            }
        }
        // dir[a] == 0.0 keeps t_max/t_delta at +inf: that axis is never
        // crossed, and an all-zero direction therefore reports nothing.
    }

    if index.get(key_of(cur[0], cur[1], cur[2])).is_none() {
        return 0;
    }

    let mut inside = true;
    let mut hits = 0usize;
    loop {
        let a = argmin3(&t_max);
        let t = t_max[a];
        // `!(t <= max_dist)` rather than `t > max_dist`, deliberately: an
        // all-infinite t_max (a direction that never crosses anything) or a
        // non-finite cap must terminate the walk, and `inf > inf` is false.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(t <= max_dist) {
            break;
        }
        // i64 first: a ray is *expected* to walk off the end of the object,
        // so leaving the coordinate range is a reachable path here. A cell
        // outside it holds no voxel — coordinates are i32 — so the step is
        // into empty space: report the exit if we were inside, then stop,
        // since there is nothing further to probe.
        let next = cur[a] as i64 + step[a] as i64;
        if next < i32::MIN as i64 || next > i32::MAX as i64 {
            if inside {
                out[hits] = t;
                hits += 1;
            }
            break;
        }
        cur[a] = next as i32;
        t_max[a] += t_delta[a];

        let occupied = index.get(key_of(cur[0], cur[1], cur[2])).is_some();
        if occupied != inside {
            out[hits] = t;
            hits += 1;
            inside = occupied;
            if hits == out.len() {
                break;
            }
        }
    }
    hits as i32
}

/// Walk a whole batch of rays. Returns `(t, n_hits)`: `t` flat of length
/// `r * max_crossings` (the Python wrapper reshapes it), padded with `+inf`
/// beyond each ray's `n_hits` entries.
///
/// Rays are completely independent — no shared mutable state, a fixed output
/// slot each — so this loop is the natural first `rayon` target in the
/// library. Left serial until it is measured.
pub fn ray_exits(index: &SpatialIndex, rays: &Rays<'_>) -> (Vec<f64>, Vec<i32>) {
    let r = rays.max_dist.len();
    debug_assert_eq!(rays.origins.len(), 3 * r);
    debug_assert_eq!(rays.directions.len(), 3 * r);
    debug_assert!(rays.max_crossings >= 1);

    let mut t = vec![f64::INFINITY; r * rays.max_crossings];
    let mut n_hits = vec![0i32; r];

    for (row, (slots, hits)) in t
        .chunks_exact_mut(rays.max_crossings)
        .zip(n_hits.iter_mut())
        .enumerate()
    {
        let b = row * 3;
        *hits = ray_exits_one(
            [rays.origins[b], rays.origins[b + 1], rays.origins[b + 2]],
            [
                rays.directions[b],
                rays.directions[b + 1],
                rays.directions[b + 2],
            ],
            rays.max_dist[row],
            index,
            slots,
        );
    }
    (t, n_hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexKind;
    use rustc_hash::FxHashSet;

    const KINDS: [IndexKind; 2] = [IndexKind::Sorted, IndexKind::Hash];

    fn build(coords: &[i32], kind: IndexKind) -> SpatialIndex {
        SpatialIndex::build(coords, coords.len() / 3, kind).unwrap()
    }

    /// A solid rasterized ball: every cell whose centre is within `r`.
    fn ball(r: i32) -> Vec<i32> {
        let mut coords = Vec::new();
        for x in -r..=r {
            for y in -r..=r {
                for z in -r..=r {
                    if x * x + y * y + z * z <= r * r {
                        coords.extend_from_slice(&[x, y, z]);
                    }
                }
            }
        }
        coords
    }

    /// A solid axis-aligned cube `[-l, l]^3`.
    fn cube(l: i32) -> Vec<i32> {
        let mut coords = Vec::new();
        for x in -l..=l {
            for y in -l..=l {
                for z in -l..=l {
                    coords.extend_from_slice(&[x, y, z]);
                }
            }
        }
        coords
    }

    fn walk(index: &SpatialIndex, o: [f64; 3], d: [f64; 3], max_dist: f64, k: usize) -> Vec<f64> {
        let mut out = vec![f64::INFINITY; k];
        let hits = ray_exits_one(o, d, max_dist, index, &mut out);
        out.truncate(hits as usize);
        out
    }

    #[test]
    fn ball_exits_on_the_axes_are_exactly_r_plus_half() {
        let coords = ball(6);
        for kind in KINDS {
            let index = build(&coords, kind);
            for axis in 0..3 {
                for sign in [1.0f64, -1.0] {
                    let mut d = [0.0; 3];
                    d[axis] = sign;
                    // The last occupied cell on the axis is at |6|, so the
                    // exit is its far face: no tolerance needed.
                    assert_eq!(walk(&index, [0.0; 3], d, 100.0, 1), vec![6.5], "{kind:?}");
                }
            }
        }
    }

    #[test]
    fn cube_body_diagonal_exits_at_the_far_face() {
        // All three axes tie the whole way down the diagonal; the first step
        // out of the cube happens on the lowest-index axis, at t = l + 0.5.
        let coords = cube(4);
        for kind in KINDS {
            let index = build(&coords, kind);
            assert_eq!(
                walk(&index, [0.0; 3], [1.0, 1.0, 1.0], 100.0, 1),
                vec![4.5],
                "{kind:?}"
            );
        }
    }

    #[test]
    fn direction_length_scales_t() {
        // `t` is parametric in the given direction, not in cells: halving the
        // direction doubles it. This is how the caller gets physical units
        // out without the primitive knowing about voxel spacing.
        let coords = ball(5);
        let index = build(&coords, IndexKind::Hash);
        assert_eq!(
            walk(&index, [0.0; 3], [0.5, 0.0, 0.0], 100.0, 1),
            vec![11.0]
        );
        assert_eq!(
            walk(&index, [0.0; 3], [2.0, 0.0, 0.0], 100.0, 1),
            vec![2.75]
        );
    }

    #[test]
    fn crossings_alternate_across_a_gap() {
        // Two slabs along x: [0, 2] and [6, 8]. From the origin the ray exits
        // at 2.5, re-enters at 5.5 and exits again at 8.5.
        let mut coords = Vec::new();
        for x in [0, 1, 2, 6, 7, 8] {
            coords.extend_from_slice(&[x, 0, 0]);
        }
        for kind in KINDS {
            let index = build(&coords, kind);
            assert_eq!(
                walk(&index, [0.0; 3], [1.0, 0.0, 0.0], 100.0, 4),
                vec![2.5, 5.5, 8.5],
                "{kind:?}"
            );
            // max_crossings truncates, it does not change what comes first
            assert_eq!(walk(&index, [0.0; 3], [1.0, 0.0, 0.0], 100.0, 1), vec![2.5]);
            assert_eq!(
                walk(&index, [0.0; 3], [1.0, 0.0, 0.0], 100.0, 2),
                vec![2.5, 5.5]
            );
        }
    }

    #[test]
    fn max_dist_bounds_t_not_the_cell_count() {
        let mut coords = Vec::new();
        for x in 0..2000 {
            coords.extend_from_slice(&[x, 0, 0]);
        }
        let index = build(&coords, IndexKind::Hash);
        // A corridor longer than the cap: nothing is reported, and the walk
        // stops at the cap rather than running to the end of the corridor.
        assert!(walk(&index, [0.0; 3], [1.0, 0.0, 0.0], 50.0, 1).is_empty());
        // Raise the cap past the far end and the exit appears.
        assert_eq!(
            walk(&index, [0.0; 3], [1.0, 0.0, 0.0], 5000.0, 1),
            vec![1999.5]
        );
        // The cap is on `t`, so a half-length direction needs twice the cap.
        assert!(walk(&index, [0.0; 3], [0.5, 0.0, 0.0], 3000.0, 1).is_empty());
    }

    #[test]
    fn degenerate_directions_and_origins() {
        let coords = ball(3);
        let index = build(&coords, IndexKind::Hash);
        // an all-zero direction crosses nothing and must terminate
        assert!(walk(&index, [0.0; 3], [0.0; 3], 100.0, 2).is_empty());
        // a zero component simply never crosses that axis
        assert_eq!(walk(&index, [0.0; 3], [0.0, 1.0, 0.0], 100.0, 1), vec![3.5]);
        // an origin in an empty cell has nothing to exit
        assert!(walk(&index, [50.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 1).is_empty());
        // ...including a cell just outside the object
        assert!(walk(&index, [4.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 1).is_empty());
        // non-finite input is rejected at the Python boundary; here it must at
        // least terminate rather than spin
        assert!(walk(&index, [f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 1).is_empty());
        assert!(walk(&index, [0.0; 3], [f64::NAN, 0.0, 0.0], 100.0, 1).is_empty());
    }

    #[test]
    fn origin_on_a_cell_boundary_does_not_step_backwards() {
        let coords = ball(4);
        let index = build(&coords, IndexKind::Sorted);
        // Exactly on the +x face of cell 0, which is the first point of cell 1
        // under the half-open convention: the exit is still the far face of
        // cell 4, i.e. 4.5 - 0.5 = 4.0 away.
        assert_eq!(
            walk(&index, [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 1),
            vec![4.0]
        );
        // The same point walked the other way: cell 1, then 0, -1, ..., -4.
        assert_eq!(
            walk(&index, [0.5, 0.0, 0.0], [-1.0, 0.0, 0.0], 100.0, 1),
            vec![5.0]
        );
        // A negative boundary: -0.5 belongs to cell 0, not cell -1.
        assert_eq!(
            walk(&index, [-0.5, 0.0, 0.0], [-1.0, 0.0, 0.0], 100.0, 1),
            vec![4.0]
        );
    }

    #[test]
    fn walking_off_the_i32_range_stops_instead_of_wrapping() {
        // A lone voxel at the +x extreme: the ray leaves it and immediately
        // runs out of representable coordinates. The exit is real — no voxel
        // can live outside the i32 range — so it is reported, and the walk
        // then stops rather than wrapping to the far end of the axis.
        // Both extremes are occupied, so a wrap would show up as a spurious
        // re-entry immediately after the exit.
        let coords = vec![i32::MAX, 0, 0, i32::MIN, 0, 0];
        for kind in KINDS {
            let index = build(&coords, kind);
            assert_eq!(
                walk(
                    &index,
                    [i32::MAX as f64, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    1e18,
                    4
                ),
                vec![0.5],
                "{kind:?}"
            );
            assert_eq!(
                walk(
                    &index,
                    [i32::MIN as f64, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    1e18,
                    4
                ),
                vec![0.5],
                "{kind:?}"
            );
        }
    }

    /// Brute-force oracle: march in tiny fixed steps and report the `t` of
    /// the first sample whose cell occupancy differs from the previous one.
    /// Shares no arithmetic and no data structure with the DDA.
    fn sampled_crossings(
        present: &FxHashSet<(i32, i32, i32)>,
        o: [f64; 3],
        d: [f64; 3],
        max_dist: f64,
        k: usize,
        step: f64,
    ) -> Vec<f64> {
        let cell = |t: f64| {
            (
                cell_of(o[0] + t * d[0]) as i32,
                cell_of(o[1] + t * d[1]) as i32,
                cell_of(o[2] + t * d[2]) as i32,
            )
        };
        let mut inside = present.contains(&cell(0.0));
        let mut out = Vec::new();
        let mut t = 0.0;
        while t <= max_dist && out.len() < k {
            t += step;
            let now = present.contains(&cell(t));
            if now != inside {
                out.push(t);
                inside = now;
            }
        }
        out
    }

    /// Deterministic xorshift64: the raw bits, for callers that want integers.
    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    /// The same generator as a stream of floats in `[-0.5, 0.5)`.
    fn rand_centered(seed: u64) -> impl FnMut() -> f64 {
        let mut bits = xorshift(seed);
        move || (bits() >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    /// A shuffled pseudo-random cloud, dense enough that rays cross its
    /// boundary many times — the case the alternation logic has to get right.
    fn scattered_cloud() -> Vec<i32> {
        let mut seen = FxHashSet::default();
        let mut coords = Vec::new();
        let mut bits = xorshift(0x2545_F491_4F6C_DD1D);
        for _ in 0..6000 {
            let state = bits();
            let p = (
                (state % 15) as i32 - 7,
                ((state >> 8) % 15) as i32 - 7,
                ((state >> 16) % 15) as i32 - 7,
            );
            if seen.insert(p) {
                coords.extend_from_slice(&[p.0, p.1, p.2]);
            }
        }
        coords
    }

    #[test]
    fn matches_a_brute_force_sampler_on_a_scattered_cloud() {
        let coords = scattered_cloud();
        let present: FxHashSet<(i32, i32, i32)> =
            coords.chunks_exact(3).map(|v| (v[0], v[1], v[2])).collect();
        let index = build(&coords, IndexKind::Hash);

        let step = 1e-4;
        let mut rand = rand_centered(0x9E37_79B9_7F4A_7C15);

        let mut checked = 0;
        for _ in 0..200 {
            let o = [rand() * 4.0, rand() * 4.0, rand() * 4.0];
            // Deliberately generic directions: a ray aimed exactly at a cell
            // corner is a measure-zero tie the sampler cannot resolve.
            let d = [rand() + 0.013, rand() + 0.027, rand() + 0.041];
            if !present.contains(&(
                cell_of(o[0]) as i32,
                cell_of(o[1]) as i32,
                cell_of(o[2]) as i32,
            )) {
                continue;
            }
            let got = walk(&index, o, d, 12.0, 32);
            let want = sampled_crossings(&present, o, d, 12.0, 32, step);
            // One-sided by construction: the sampler can only *miss* a
            // crossing (a ray clipping a cell corner can stay inside it for
            // less than one sampling step), never invent one. So every
            // crossing it does find must be bracketed by an exact one — but
            // the DDA is allowed to find more.
            assert!(
                got.len() >= want.len(),
                "o={o:?} d={d:?}: {got:?} vs {want:?}"
            );
            for w in want.iter().filter(|w| **w <= 12.0) {
                assert!(
                    got.iter().any(|g| *g <= *w && *g > *w - step),
                    "o={o:?} d={d:?}: no exact crossing in ({}, {w}], got {got:?}",
                    w - step
                );
            }
            checked += 1;
        }
        assert!(checked > 50, "only {checked} rays started inside the cloud");
    }

    #[test]
    fn backends_are_bit_identical() {
        let coords = scattered_cloud();
        let sorted = build(&coords, IndexKind::Sorted);
        let hash = build(&coords, IndexKind::Hash);
        let mut rand = rand_centered(0x1234_5678_9ABC_DEF0);
        for _ in 0..300 {
            let o = [rand() * 6.0, rand() * 6.0, rand() * 6.0];
            let d = [rand(), rand(), rand()];
            assert_eq!(
                walk(&sorted, o, d, 20.0, 4),
                walk(&hash, o, d, 20.0, 4),
                "o={o:?} d={d:?}"
            );
        }
    }

    #[test]
    fn batch_matches_the_per_ray_walk_and_pads_with_inf() {
        let coords = ball(4);
        let index = build(&coords, IndexKind::Hash);
        let origins = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0];
        let directions = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let max_dist = vec![100.0, 100.0, 100.0];
        let rays = Rays {
            origins: &origins,
            directions: &directions,
            max_dist: &max_dist,
            max_crossings: 2,
        };
        let (t, hits) = ray_exits(&index, &rays);
        assert_eq!(hits, vec![1, 0, 0]);
        assert_eq!(t[0], 4.5);
        // every slot past `n_hits` stays +inf
        assert!(t[1].is_infinite() && t[2..].iter().all(|v| v.is_infinite()));
    }

    #[test]
    fn empty_batch_and_empty_voxel_set() {
        let index = build(&[], IndexKind::Hash);
        let rays = Rays {
            origins: &[],
            directions: &[],
            max_dist: &[],
            max_crossings: 1,
        };
        assert_eq!(ray_exits(&index, &rays), (Vec::new(), Vec::new()));

        // No voxels at all: every origin is empty, so nothing is reported.
        let origins = [0.0, 0.0, 0.0];
        let directions = [1.0, 0.0, 0.0];
        let max_dist = [10.0];
        let rays = Rays {
            origins: &origins,
            directions: &directions,
            max_dist: &max_dist,
            max_crossings: 1,
        };
        let (t, hits) = ray_exits(&index, &rays);
        assert_eq!(hits, vec![0]);
        assert!(t[0].is_infinite());
    }
}
