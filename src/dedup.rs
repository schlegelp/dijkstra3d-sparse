//! Coordinate deduplication: dense labelling of integer coordinates by exact
//! equality (the `factorize` primitive).
//!
//! The sparse analogue of `np.unique(coords, axis=0, return_inverse=True)`,
//! done as one hash pass instead of a sort. Unlike `connected_components`
//! (which groups by spatial *adjacency* and rejects duplicate coordinates),
//! `factorize` groups rows by *identical* coordinate and is built **for**
//! duplicated input — collapsing an array that is mostly repeats (e.g. the
//! 4-corners-per-quad a mesher emits) into dense per-row labels. It reuses the
//! same `key_of` coordinate hash as the spatial index, so equality here means
//! exactly what membership means everywhere else in the crate.

use crate::index::key_of;
use rustc_hash::FxHashMap;

/// Assign each row a dense label, equal for identical coordinates. Labels run
/// `0 .. n_labels-1` in order of first appearance by row (the same rule
/// `connected_components` uses). With `want_index`, also returns `reps`, where
/// `reps[k]` is the row index of the first occurrence of label `k` — so
/// `coords[reps]` is the deduplicated coordinate set, `labels`-aligned.
///
/// Duplicates are the point, not an error: the map tops out at `n_labels`
/// entries however many times each coordinate repeats.
pub fn factorize(
    coords: &[i32],
    n: usize,
    want_index: bool,
) -> (usize, Vec<i64>, Option<Vec<i64>>) {
    // The map holds only distinct coordinates (`n_labels` of them); sizing for
    // a roughly half-duplicate input avoids most rehashes as it fills, and it
    // then stays hot for the remaining lookups.
    let mut map: FxHashMap<u128, i64> = FxHashMap::default();
    map.reserve(n / 2 + 1);
    let mut labels = vec![0i64; n];
    let mut reps: Vec<i64> = Vec::new();
    let mut next = 0i64;
    for (row, label) in labels.iter_mut().enumerate() {
        let b = row * 3;
        let key = key_of(coords[b], coords[b + 1], coords[b + 2]);
        let id = *map.entry(key).or_insert_with(|| {
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
}
