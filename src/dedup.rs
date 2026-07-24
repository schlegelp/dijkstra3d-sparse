//! Coordinate deduplication: dense labelling of integer coordinates by exact
//! equality (the `factorize` primitive).
//!
//! The sparse analogue of `np.unique(coords, axis=0, return_inverse=True)`,
//! done as one pass over the rows instead of a sort. Unlike `connected_components`
//! (which groups by spatial *adjacency* and rejects duplicate coordinates),
//! `factorize` groups rows by *identical* coordinate and is built **for**
//! duplicated input — collapsing an array that is mostly repeats (e.g. the
//! 4-corners-per-quad a mesher emits) into dense per-row labels.
//!
//! Equality is exact integer equality on the three coordinates, however it is
//! resolved — the same relation `key_of` gives the spatial index, so equality
//! here means what membership means everywhere else in the crate. Which of the
//! three resolutions `factorize` picks is a cost decision only (see below); it
//! is invisible in the output.

use crate::index::key_of;
use rustc_hash::FxHashMap;
use std::hash::Hash;

/// Cells of bounding box per input row at which the direct-address table
/// stops being worth it. At the limit the table is `4 * 2 = 8` bytes/row —
/// the size of the `labels` output, and well under the ~35 bytes/row the
/// hash map reserves up front whatever the input looks like.
const DENSE_CELLS_PER_ROW: u128 = 2;

/// Assign each row a dense label, equal for identical coordinates. Labels run
/// `0 .. n_labels-1` in order of first appearance by row (the same rule
/// `connected_components` uses). With `want_index`, also returns `reps`, where
/// `reps[k]` is the row index of the first occurrence of label `k` — so
/// `coords[reps]` is the deduplicated coordinate set, `labels`-aligned.
///
/// Duplicates are the point, not an error: nothing here grows past the
/// `n_labels` distinct coordinates however many times each one repeats.
///
/// One cheap pass over the coordinates picks between three ways of resolving
/// a coordinate to a label, all of which produce identical output:
///
/// 1. Bounding box no bigger than `DENSE_CELLS_PER_ROW` cells per row —
///    a direct-address table, no hashing at all (`dense_pass`). This is the
///    normal case for coordinates *derived from* a voxel grid, which is what
///    the primitive is for (mesher corners, downsampled cells).
/// 2. Bounding box addressable in 64 bits — the same hash pass, but keyed on
///    the box offset rather than the 128-bit spatial key, which halves the
///    map's footprint and its cost per probe.
/// 3. Anything else (coordinates spread across the full i32 range) — keyed
///    on `key_of`, exactly as before.
pub fn factorize(
    coords: &[i32],
    n: usize,
    want_index: bool,
) -> (usize, Vec<i64>, Option<Vec<i64>>) {
    debug_assert_eq!(coords.len(), 3 * n);
    if n == 0 {
        return (
            0,
            Vec::new(),
            if want_index { Some(Vec::new()) } else { None },
        );
    }
    let (lo, ext, volume) = bounds(coords);

    if volume <= DENSE_CELLS_PER_ROW * n as u128 {
        // `n <= i32::MAX` (checked at the boundary), so the table length
        // fits a usize on every platform the crate builds for.
        dense_pass(coords, n, want_index, volume as usize, lo, ext)
    } else if volume <= u64::MAX as u128 {
        hash_pass(coords, n, want_index, |v| box_offset(v, lo, ext))
    } else {
        hash_pass(coords, n, want_index, |v| key_of(v[0], v[1], v[2]))
    }
}

/// Per-axis minimum and extent of the input, and the cell count of the
/// bounding box. The count is `u128` because each extent can span the whole
/// i32 range; the extents themselves cannot overflow, being differences of
/// i32 widened to i64.
fn bounds(coords: &[i32]) -> ([i32; 3], [u64; 3], u128) {
    let (mut lo, mut hi) = ([i32::MAX; 3], [i32::MIN; 3]);
    for v in coords.chunks_exact(3) {
        for a in 0..3 {
            lo[a] = lo[a].min(v[a]);
            hi[a] = hi[a].max(v[a]);
        }
    }
    let ext = [0, 1, 2].map(|a| (hi[a] as i64 - lo[a] as i64 + 1) as u64);
    let volume = ext[0] as u128 * ext[1] as u128 * ext[2] as u128;
    (lo, ext, volume)
}

/// Row-major offset of a coordinate within the bounding box, in `[0, volume)`.
/// Only called once `volume` is known to fit in 64 bits, which bounds every
/// intermediate here by `volume - 1`.
#[inline(always)]
fn box_offset(v: &[i32], lo: [i32; 3], ext: [u64; 3]) -> u64 {
    let d = [0, 1, 2].map(|a| (v[a] as i64 - lo[a] as i64) as u64);
    (d[0] * ext[1] + d[1]) * ext[2] + d[2]
}

/// Direct-address labelling: one `u32` slot per cell of the bounding box,
/// holding `label + 1` (`0` = not yet seen). No hashing, no collisions, one
/// sequential-ish read and write per row.
///
/// The slot vector is allocated *zeroed*, so it costs no memset and only the
/// pages actually touched are ever faulted in — a box that is mostly empty
/// never pays for the cells it does not use.
fn dense_pass(
    coords: &[i32],
    n: usize,
    want_index: bool,
    volume: usize,
    lo: [i32; 3],
    ext: [u64; 3],
) -> (usize, Vec<i64>, Option<Vec<i64>>) {
    let mut slot = vec![0u32; volume];
    let mut labels = vec![0i64; n];
    let mut reps: Vec<i64> = Vec::new();
    let mut next = 0u32;
    for (row, (v, label)) in coords.chunks_exact(3).zip(labels.iter_mut()).enumerate() {
        let cell = &mut slot[box_offset(v, lo, ext) as usize];
        if *cell == 0 {
            next += 1; // stored biased by one, so `0` can mean "unseen"
            *cell = next;
            if want_index {
                reps.push(row as i64);
            }
        }
        *label = (*cell - 1) as i64;
    }
    (
        next as usize,
        labels,
        if want_index { Some(reps) } else { None },
    )
}

/// Hash labelling, over whichever key the bounding box makes available.
fn hash_pass<K, F>(
    coords: &[i32],
    n: usize,
    want_index: bool,
    key: F,
) -> (usize, Vec<i64>, Option<Vec<i64>>)
where
    K: Eq + Hash,
    F: Fn(&[i32]) -> K,
{
    // The map holds only distinct coordinates (`n_labels` of them); sizing for
    // a roughly half-duplicate input avoids most rehashes as it fills, and it
    // then stays hot for the remaining lookups.
    let mut map: FxHashMap<K, i64> = FxHashMap::default();
    map.reserve(n / 2 + 1);
    let mut labels = vec![0i64; n];
    let mut reps: Vec<i64> = Vec::new();
    let mut next = 0i64;
    for (row, (v, label)) in coords.chunks_exact(3).zip(labels.iter_mut()).enumerate() {
        let id = *map.entry(key(v)).or_insert_with(|| {
            let id = next;
            next += 1;
            if want_index {
                reps.push(row as i64);
            }
            id
        });
        *label = id;
    }
    (
        next as usize,
        labels,
        if want_index { Some(reps) } else { None },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_distinct_is_identity() {
        // Three different coordinates: every row is its own label, and each is
        // its own representative.
        let coords = vec![0, 0, 0, 5, -3, 2, -7, 8, 9];
        let (n_labels, labels, reps) = factorize(&coords, 3, true);
        assert_eq!(n_labels, 3);
        assert_eq!(labels, vec![0, 1, 2]);
        assert_eq!(reps, Some(vec![0, 1, 2]));
    }

    #[test]
    fn all_identical_collapses_to_one() {
        let coords = vec![4, 4, 4, 4, 4, 4, 4, 4, 4];
        let (n_labels, labels, reps) = factorize(&coords, 3, true);
        assert_eq!(n_labels, 1);
        assert_eq!(labels, vec![0, 0, 0]);
        assert_eq!(reps, Some(vec![0]));
    }

    #[test]
    fn first_appearance_order_and_reps() {
        // rows:   A  B  A  C  B  A   (A=(1,1,1) B=(2,2,2) C=(0,0,0))
        // labels: 0  1  0  2  1  0   reps: A@0, B@1, C@3
        let coords = vec![1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1];
        let (n_labels, labels, reps) = factorize(&coords, 6, true);
        assert_eq!(n_labels, 3);
        assert_eq!(labels, vec![0, 1, 0, 2, 1, 0]);
        assert_eq!(reps, Some(vec![0, 1, 3]));
    }

    #[test]
    fn want_index_false_omits_reps() {
        let coords = vec![0, 0, 0, 0, 0, 0];
        let (n_labels, labels, reps) = factorize(&coords, 2, false);
        assert_eq!(n_labels, 1);
        assert_eq!(labels, vec![0, 0]);
        assert_eq!(reps, None);
    }

    #[test]
    fn empty_input() {
        let (n_labels, labels, reps) = factorize(&[], 0, true);
        assert_eq!(n_labels, 0);
        assert_eq!(labels, Vec::<i64>::new());
        assert_eq!(reps, Some(Vec::<i64>::new()));
    }

    /// Which of the three resolution strategies `factorize` will take —
    /// asserted in the tests below so they cannot silently stop covering the
    /// path they were written for.
    #[derive(PartialEq, Eq, Debug)]
    enum Path {
        Dense,
        Hash64,
        Hash128,
    }

    fn path_taken(coords: &[i32], n: usize) -> Path {
        let (_, _, volume) = bounds(coords);
        if volume <= DENSE_CELLS_PER_ROW * n as u128 {
            Path::Dense
        } else if volume <= u64::MAX as u128 {
            Path::Hash64
        } else {
            Path::Hash128
        }
    }

    /// Oracle: labels by first appearance, reps by first occurrence.
    fn reference(coords: &[i32], n: usize) -> (usize, Vec<i64>, Vec<i64>) {
        let mut seen: Vec<(i32, i32, i32)> = Vec::new();
        let mut labels = Vec::with_capacity(n);
        let mut reps = Vec::new();
        for row in 0..n {
            let p = (coords[3 * row], coords[3 * row + 1], coords[3 * row + 2]);
            match seen.iter().position(|&q| q == p) {
                Some(k) => labels.push(k as i64),
                None => {
                    labels.push(seen.len() as i64);
                    reps.push(row as i64);
                    seen.push(p);
                }
            }
        }
        (seen.len(), labels, reps)
    }

    /// Pseudo-random rows drawn from a box of `span` cells per axis, offset to
    /// `origin`, with roughly `n / repeat` distinct coordinates.
    fn cloud(n: usize, span: i64, origin: [i64; 3], seed: u64) -> Vec<i32> {
        let mut state = seed | 1;
        let mut coords = Vec::with_capacity(3 * n);
        for _ in 0..n {
            for o in origin {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                coords.push((o + (state % span as u64) as i64) as i32);
            }
        }
        coords
    }

    #[test]
    fn all_paths_agree_with_the_oracle() {
        // Same shape of input, three bounding boxes: compact (direct-address),
        // wide but 64-bit addressable, and spread across the whole i32 range.
        let cases = [
            (cloud(400, 6, [0, 0, 0], 0x9E37), Path::Dense),
            (cloud(400, 6, [-3, -100_000, 7], 0x1234), Path::Dense),
            (cloud(400, 40_000, [0, 0, 0], 0xABCD), Path::Hash64),
            (
                cloud(400, 3_000_000_000, [-1_500_000_000; 3], 0x5EED),
                Path::Hash128,
            ),
        ];
        for (coords, want_path) in cases {
            let n = coords.len() / 3;
            assert_eq!(path_taken(&coords, n), want_path);
            let (wn, wlabels, wreps) = reference(&coords, n);
            let (gn, glabels, greps) = factorize(&coords, n, true);
            assert_eq!((gn, &glabels, greps.as_ref()), (wn, &wlabels, Some(&wreps)));
            // and `want_index = false` must not disturb the labelling
            assert_eq!(factorize(&coords, n, false), (wn, wlabels, None));
        }
    }

    #[test]
    fn dense_path_handles_negative_and_degenerate_boxes() {
        // A box thinner than one cell on two axes, at negative coordinates:
        // the offset must stay inside the table.
        let coords: Vec<i32> = vec![-5, -5, -5, -5, -5, -4, -5, -5, -5, -5, -5, -3];
        assert_eq!(path_taken(&coords, 4), Path::Dense);
        assert_eq!(
            factorize(&coords, 4, true),
            (3, vec![0, 1, 0, 2], Some(vec![0, 1, 3]))
        );
        // A single row is its own box.
        assert_eq!(
            factorize(&[7, -3, 11], 1, true),
            (1, vec![0], Some(vec![0]))
        );
    }

    #[test]
    fn i32_extremes_do_not_overflow_the_box() {
        // Extents of 2^32 on every axis: the volume overflows u64, so this is
        // the `key_of` path — and the extents themselves must not wrap while
        // being measured.
        let coords: Vec<i32> = vec![
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MAX,
            i32::MAX,
            i32::MAX,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            0,
            0,
            0,
        ];
        let (lo, ext, volume) = bounds(&coords);
        assert_eq!(lo, [i32::MIN; 3]);
        assert_eq!(ext, [1u64 << 32; 3]);
        assert_eq!(volume, 1u128 << 96);
        assert_eq!(path_taken(&coords, 4), Path::Hash128);
        assert_eq!(
            factorize(&coords, 4, true),
            (3, vec![0, 1, 0, 2], Some(vec![0, 1, 3]))
        );
    }

    #[test]
    fn one_axis_spanning_i32_still_fits_64_bits() {
        // Full i32 extent on x alone (2^32 cells) with thin y/z: 64-bit
        // addressable, so this must take the box-offset hash, not `key_of`.
        let coords: Vec<i32> = vec![i32::MIN, 0, 0, i32::MAX, 1, 0, i32::MIN, 0, 0];
        let (_, ext, volume) = bounds(&coords);
        assert_eq!((ext, volume), ([1u64 << 32, 2, 1], 1u128 << 33));
        assert_eq!(path_taken(&coords, 3), Path::Hash64);
        assert_eq!(
            factorize(&coords, 3, true),
            (2, vec![0, 1, 0], Some(vec![0, 1]))
        );
    }
}
