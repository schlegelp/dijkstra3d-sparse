//! Connected components and label adjacency over the implicit sparse grid.
//! Both walk the same neighbour probe as Dijkstra — no edge list is built.

use crate::index::{duplicate_msg, key_of, SpatialIndex};
use rustc_hash::FxHashSet;

struct Dsu {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Dsu {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            // path halving
            let grand = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = grand;
            x = grand;
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        match self.rank[ra as usize].cmp(&self.rank[rb as usize]) {
            std::cmp::Ordering::Less => self.parent[ra as usize] = rb,
            std::cmp::Ordering::Greater => self.parent[rb as usize] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb as usize] = ra;
                self.rank[ra as usize] += 1;
            }
        }
    }
}

/// Visit every undirected neighbour pair `(row, nbr)` of the implicit grid
/// exactly once. Shared by `connected_components` and `label_adjacency` —
/// the only place the neighbour probe is written.
fn for_each_adjacent_pair<F>(
    coords: &[i32],
    n: usize,
    index: &SpatialIndex,
    offsets: &[(i32, i32, i32, f64)],
    mut visit: F,
) where
    F: FnMut(usize, usize),
{
    // Each undirected neighbour pair only needs probing once: keep the
    // lexicographically-positive half of the (symmetric) offset set.
    let half: Vec<(i32, i32, i32)> = offsets
        .iter()
        .map(|&(dx, dy, dz, _)| (dx, dy, dz))
        .filter(|&(dx, dy, dz)| (dx, dy, dz) > (0, 0, 0))
        .collect();

    for row in 0..n {
        let b = row * 3;
        let (x, y, z) = (coords[b], coords[b + 1], coords[b + 2]);
        for &(dx, dy, dz) in &half {
            let nx = x as i64 + dx as i64;
            let ny = y as i64 + dy as i64;
            let nz = z as i64 + dz as i64;
            if nx < i32::MIN as i64
                || nx > i32::MAX as i64
                || ny < i32::MIN as i64
                || ny > i32::MAX as i64
                || nz < i32::MIN as i64
                || nz > i32::MAX as i64
            {
                continue;
            }
            if let Some(nbr) = index.get(key_of(nx as i32, ny as i32, nz as i32)) {
                visit(row, nbr as usize);
            }
        }
    }
}

/// Label the connected components of the voxel set. Labels are assigned in
/// order of first appearance by row index (0, 1, 2, ...), so the output is
/// deterministic. Returns `(n_components, labels)`.
///
/// With `group`, two coordinate-adjacent voxels are only connected when
/// their group values are equal — i.e. the components of the sub-graph
/// induced by each group value. Grouping *constrains* unions; it never
/// merges spatially separate runs that happen to share a value. Voxels
/// whose group differs from all their neighbours' become singletons.
pub fn connected_components(
    coords: &[i32],
    n: usize,
    index: &SpatialIndex,
    offsets: &[(i32, i32, i32, f64)],
    group: Option<&[i64]>,
) -> (usize, Vec<i32>) {
    let mut dsu = Dsu::new(n);

    for_each_adjacent_pair(coords, n, index, offsets, |row, nbr| {
        if let Some(g) = group {
            if g[row] != g[nbr] {
                return;
            }
        }
        dsu.union(row as u32, nbr as u32);
    });

    let mut labels = vec![-1i32; n];
    let mut root_label = vec![-1i32; n];
    let mut next = 0i32;
    for (row, label) in labels.iter_mut().enumerate() {
        let root = dsu.find(row as u32) as usize;
        if root_label[root] < 0 {
            root_label[root] = next;
            next += 1;
        }
        *label = root_label[root];
    }
    (next as usize, labels)
}

/// The six face-neighbour offsets, in the bit order `exposed_faces`
/// reports: `+x, -x, +y, -y, +z, -z`. Bit `k` of a voxel's mask is set
/// iff the neighbour at `FACES[k]` is absent from the set.
#[cfg(test)]
const FACES: [(i32, i32, i32); 6] = [
    (1, 0, 0),  // bit 0: +x
    (-1, 0, 0), // bit 1: -x
    (0, 1, 0),  // bit 2: +y
    (0, -1, 0), // bit 3: -y
    (0, 0, 1),  // bit 4: +z
    (0, 0, -1), // bit 5: -z
];

/// Every face exposed: the state a voxel starts in, before the sweeps clear
/// the faces that turn out to be covered.
const ALL_FACES: u8 = 0b111_111;

/// Bit position of an axis' 32-bit field inside a `key_of` key: `x` occupies
/// `[64, 96)`, `y` `[32, 64)`, `z` `[0, 32)`. Stepping one voxel along
/// `+axis` is exactly `key + (1 << axis_shift(axis))`, which is why the
/// sweeps never touch coordinates at all.
const fn axis_shift(axis: u32) -> u32 {
    64 - 32 * axis
}

/// Clear both face bits of every voxel pair adjacent along one axis.
///
/// `at(i)` yields the `(key, row)` of the `i`-th voxel *in ascending key
/// order*. Since `key_of` is order-preserving and the step is a constant
/// `delta`, the target `key + delta` is strictly increasing in `i`, so the
/// cursor `j` never rewinds: one merge pass over the sorted keys replaces
/// `n` binary searches, and every probe is a sequential read.
///
/// Each hit clears *both* sides — the `+axis` bit of the near voxel and the
/// `-axis` bit of the far one — so only the three positive axes are swept.
fn sweep_axis<const AXIS: u32, F>(n: usize, at: F, mask: &mut [u8])
where
    F: Fn(usize) -> (u128, u32),
{
    let shift = axis_shift(AXIS);
    let delta = 1u128 << shift;
    let (pos, neg) = (1u8 << (2 * AXIS), 1u8 << (2 * AXIS + 1));

    let mut j = 0usize;
    for i in 0..n {
        let (key, row) = at(i);
        // One step past the axis maximum is not an i32 coordinate, so that
        // face stays exposed. It is also exactly the case where the add
        // would carry into the next axis' field and name a different voxel.
        if (key >> shift) as u32 == u32::MAX {
            continue;
        }
        let target = key + delta;
        while j < n && at(j).0 < target {
            j += 1;
        }
        if j == n {
            break; // every later target is larger still
        }
        let (nkey, nrow) = at(j);
        if nkey == target {
            mask[row as usize] &= !pos;
            mask[nrow as usize] &= !neg;
        }
    }
    // The `-axis` face of a voxel at the axis minimum is likewise never
    // cleared, because no voxel can sit one step below it.
}

/// Hash-backend fallback: probe the three *positive* faces per row and
/// scatter the result to both sides of each hit. Halves the lookups a
/// six-offset probe would do, at no extra memory — the mask write to the
/// neighbour's row is one byte into an `N`-byte array.
fn probe_positive_faces(coords: &[i32], n: usize, index: &SpatialIndex, mask: &mut [u8]) {
    debug_assert_eq!(coords.len(), 3 * n);
    for (row, v) in coords.chunks_exact(3).enumerate() {
        let key = key_of(v[0], v[1], v[2]);
        for axis in 0..3u32 {
            let shift = axis_shift(axis);
            if (key >> shift) as u32 == u32::MAX {
                continue; // no i32 neighbour past the axis maximum
            }
            if let Some(nbr) = index.get(key + (1u128 << shift)) {
                mask[row] &= !(1u8 << (2 * axis));
                mask[nbr as usize] &= !(1u8 << (2 * axis + 1));
            }
        }
    }
}

/// Per voxel, a 6-bit mask of which face-neighbours are *absent* from the
/// set. This is the surface-extraction probe `find_surface_voxels` runs —
/// the same coordinate walk as `connected_components`, minus the union-find:
/// one pass, no `(N, 3)` neighbour temporaries.
///
/// Bit `k` of `mask[row]` is set iff `voxels[row] + FACES[k]` is not in the
/// set. A fully interior voxel yields `0`; a lone voxel yields `0b111111`
/// (`63`).
///
/// The mask is directional, but the *probing* need not be: an adjacency
/// found from the `+axis` side clears the far voxel's `-axis` bit too, so
/// three offsets suffice — the `half`-offset trick the pair primitives use,
/// with a scatter instead of a single visit. A sorted index then goes one
/// better and skips lookups entirely (see `sweep_axis`); the free function
/// takes `exposed_faces_unindexed`, which builds neither index.
pub fn exposed_faces(coords: &[i32], n: usize, index: &SpatialIndex) -> Vec<u8> {
    let mut mask = vec![ALL_FACES; n];
    match index {
        // The sorted backend already holds exactly what the sweep needs.
        SpatialIndex::Sorted { keys, rows } => {
            debug_assert_eq!(keys.len(), n);
            let at = |i: usize| (keys[i], rows[i]);
            sweep_axis::<0, _>(keys.len(), at, &mut mask);
            sweep_axis::<1, _>(keys.len(), at, &mut mask);
            sweep_axis::<2, _>(keys.len(), at, &mut mask);
        }
        SpatialIndex::Hash(_) => probe_positive_faces(coords, n, index, &mut mask),
    }
    mask
}

/// `exposed_faces` for a caller holding no index — the free-function path.
///
/// Builds the least the sweep needs and nothing more: one `u128` per voxel,
/// the 96-bit key with the row index riding in the 32 bits `key_of` leaves
/// free at the top. Sorting that array *is* the index — 16 B/voxel, against
/// the ~2.5x of `SpatialIndex::Sorted` (which allocates a `(key, row)` pair
/// array on top of its two output vectors) and the ~4x of a hash table that
/// rounds its bucket count up to a power of two. It is dropped before the
/// mask is returned, which is the memory point of the primitive.
///
/// Duplicate coordinates are rejected with the message `SpatialIndex::build`
/// produces, since callers rely on it.
pub fn exposed_faces_unindexed(coords: &[i32], n: usize) -> Result<Vec<u8>, String> {
    debug_assert_eq!(coords.len(), 3 * n);
    let mut packed: Vec<u128> = coords
        .chunks_exact(3)
        .enumerate()
        .map(|(row, v)| (key_of(v[0], v[1], v[2]) << 32) | row as u128)
        .collect();
    // Sorts by key, then by row; `n <= i32::MAX` keeps the row in 32 bits.
    // Already-sorted input (what `np.argwhere` and friends hand over) costs
    // one scan here, not a full sort.
    packed.sort_unstable();
    for w in packed.windows(2) {
        if w[0] >> 32 == w[1] >> 32 {
            return Err(duplicate_msg(coords, w[0] as u32, w[1] as u32));
        }
    }

    let mut mask = vec![ALL_FACES; n];
    let at = |i: usize| (packed[i] >> 32, packed[i] as u32);
    sweep_axis::<0, _>(n, at, &mut mask);
    sweep_axis::<1, _>(n, at, &mut mask);
    sweep_axis::<2, _>(n, at, &mut mask);
    Ok(mask)
}

/// Which pairs of *distinct* labels touch: for every adjacent voxel pair
/// `(u, v)` with `labels[u] != labels[v]`, the pair `(lo, hi)`. Returns
/// deduplicated, lexicographically sorted pairs.
///
/// The set is deduplicated *as adjacencies are found*, never collected
/// first: the caller's point is to avoid materializing the edge list, and
/// on real inputs the distinct-pair count is orders of magnitude below the
/// adjacency count.
pub fn label_adjacency(
    coords: &[i32],
    n: usize,
    index: &SpatialIndex,
    offsets: &[(i32, i32, i32, f64)],
    labels: &[i64],
) -> Vec<(i64, i64)> {
    let mut seen: FxHashSet<(i64, i64)> = FxHashSet::default();

    for_each_adjacent_pair(coords, n, index, offsets, |row, nbr| {
        let (a, b) = (labels[row], labels[nbr]);
        if a != b {
            seen.insert(if a < b { (a, b) } else { (b, a) });
        }
    });

    let mut out: Vec<(i64, i64)> = seen.into_iter().collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexKind, SpatialIndex};
    use crate::offsets::build_offsets;

    #[test]
    fn two_clusters_and_connectivity_sensitivity() {
        // (0,0,0)-(1,1,0) touch diagonally: one component at 18/26-conn,
        // two at 6-conn. (9,9,9) is always separate.
        let coords: Vec<i32> = vec![0, 0, 0, 1, 1, 0, 9, 9, 9];
        let index = SpatialIndex::build(&coords, 3, IndexKind::Sorted).unwrap();

        let offs26 = build_offsets(26, [1.0; 3]);
        let (n, labels) = connected_components(&coords, 3, &index, &offs26, None);
        assert_eq!(n, 2);
        assert_eq!(labels, vec![0, 0, 1]);

        let offs6 = build_offsets(6, [1.0; 3]);
        let (n, labels) = connected_components(&coords, 3, &index, &offs6, None);
        assert_eq!(n, 3);
        assert_eq!(labels, vec![0, 1, 2]);
    }

    #[test]
    fn group_splits_a_run_but_does_not_merge_by_value() {
        // A straight 6-voxel line. Grouping splits it at the value change...
        let coords: Vec<i32> = (0..6).flat_map(|x| [x, 0, 0]).collect();
        let index = SpatialIndex::build(&coords, 6, IndexKind::Hash).unwrap();
        let offs = build_offsets(26, [1.0; 3]);

        let (n, labels) =
            connected_components(&coords, 6, &index, &offs, Some(&[0, 0, 0, 1, 1, 1]));
        assert_eq!(n, 2);
        assert_eq!(labels, vec![0, 0, 0, 1, 1, 1]);

        // ...but the same value in two spatially separate runs stays two
        // components: grouping constrains unions, it does not merge.
        let (n, labels) =
            connected_components(&coords, 6, &index, &offs, Some(&[0, 0, 1, 1, 0, 0]));
        assert_eq!(n, 3);
        assert_eq!(labels, vec![0, 0, 1, 1, 2, 2]);

        // A uniform group is indistinguishable from no group at all.
        let uniform = connected_components(&coords, 6, &index, &offs, Some(&[7; 6]));
        assert_eq!(
            uniform,
            connected_components(&coords, 6, &index, &offs, None)
        );
    }

    #[test]
    fn exposed_faces_bit_order_and_interior() {
        // A single voxel: all six faces exposed.
        let coords = vec![0, 0, 0];
        let index = SpatialIndex::build(&coords, 1, IndexKind::Hash).unwrap();
        assert_eq!(exposed_faces(&coords, 1, &index), vec![0b111111]);

        // A 2x1x1 pair along x: the touching faces clear. Left voxel (row 0)
        // loses its +x bit (bit 0); right voxel (row 1) loses its -x (bit 1).
        let coords = vec![0, 0, 0, 1, 0, 0];
        let index = SpatialIndex::build(&coords, 2, IndexKind::Sorted).unwrap();
        let mask = exposed_faces(&coords, 2, &index);
        assert_eq!(mask, vec![0b111111 & !(1 << 0), 0b111111 & !(1 << 1)]);

        // A voxel fully surrounded by its six face-neighbours: mask == 0.
        let mut coords = vec![0, 0, 0];
        for &(dx, dy, dz) in FACES.iter() {
            coords.extend_from_slice(&[dx, dy, dz]);
        }
        let index = SpatialIndex::build(&coords, 7, IndexKind::Hash).unwrap();
        assert_eq!(exposed_faces(&coords, 7, &index)[0], 0);
    }

    #[test]
    fn exposed_faces_empty_input() {
        let index = SpatialIndex::build(&[], 0, IndexKind::Hash).unwrap();
        assert_eq!(exposed_faces(&[], 0, &index), Vec::<u8>::new());
        assert_eq!(exposed_faces_unindexed(&[], 0).unwrap(), Vec::<u8>::new());
    }

    /// Brute-force oracle: bit `k` set iff `voxel + FACES[k]` is absent.
    fn reference_mask(coords: &[i32], n: usize) -> Vec<u8> {
        let present: FxHashSet<(i32, i32, i32)> = (0..n)
            .map(|r| (coords[3 * r], coords[3 * r + 1], coords[3 * r + 2]))
            .collect();
        (0..n)
            .map(|r| {
                let (x, y, z) = (coords[3 * r], coords[3 * r + 1], coords[3 * r + 2]);
                let mut m = 0u8;
                for (k, &(dx, dy, dz)) in FACES.iter().enumerate() {
                    let nbr = (x.checked_add(dx), y.checked_add(dy), z.checked_add(dz));
                    let inside = match nbr {
                        (Some(a), Some(b), Some(c)) => present.contains(&(a, b, c)),
                        _ => false,
                    };
                    if !inside {
                        m |= 1 << k;
                    }
                }
                m
            })
            .collect()
    }

    /// A shuffled pseudo-random cloud dense enough that most voxels have
    /// several face-neighbours — the case the sweep's cursor logic has to
    /// get right, unlike the hand-built shapes above.
    fn scattered_cloud() -> (Vec<i32>, usize) {
        let mut seen = FxHashSet::default();
        let mut coords = Vec::new();
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..4000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let p = (
                (state % 11) as i32 - 5,
                ((state >> 8) % 11) as i32 - 5,
                ((state >> 16) % 11) as i32 - 5,
            );
            if seen.insert(p) {
                coords.extend_from_slice(&[p.0, p.1, p.2]);
            }
        }
        let n = coords.len() / 3;
        (coords, n)
    }

    #[test]
    fn exposed_faces_all_three_paths_match_the_oracle() {
        let (coords, n) = scattered_cloud();
        let want = reference_mask(&coords, n);
        for kind in [IndexKind::Sorted, IndexKind::Hash] {
            let index = SpatialIndex::build(&coords, n, kind).unwrap();
            assert_eq!(exposed_faces(&coords, n, &index), want, "{kind:?}");
        }
        assert_eq!(exposed_faces_unindexed(&coords, n).unwrap(), want);
    }

    #[test]
    fn exposed_faces_at_the_i32_extremes() {
        // A step off an axis extreme has no i32 coordinate, so that face is
        // exposed. The trap the sweeps have to dodge: the packed key steps by
        // a constant, so `(x, i32::MAX, z) + (0, 1, 0)` must not carry into
        // the x field and be read as the real voxel `(x + 1, i32::MIN, z)`.
        #[rustfmt::skip]
        let coords: Vec<i32> = vec![
            i32::MAX, 0, 0,   i32::MAX - 1, 0, 0,  // +x extreme, touched from below
            i32::MIN, 7, 7,   i32::MIN + 1, 7, 7,  // -x extreme, touched from above
            3, i32::MAX, 9,   4, i32::MIN, 9,      // the y -> x carry trap
            3, 5, i32::MAX,   3, 6, i32::MIN,      // the z -> y carry trap
        ];
        let n = coords.len() / 3;
        let want = reference_mask(&coords, n);
        for kind in [IndexKind::Sorted, IndexKind::Hash] {
            let index = SpatialIndex::build(&coords, n, kind).unwrap();
            assert_eq!(exposed_faces(&coords, n, &index), want, "{kind:?}");
        }
        assert_eq!(exposed_faces_unindexed(&coords, n).unwrap(), want);
        // The extremes themselves: +x of the i32::MAX voxel is exposed, its
        // -x is covered by row 1; mirrored for the i32::MIN voxel.
        assert_eq!(want[0] & 0b11, 0b01);
        assert_eq!(want[2] & 0b11, 0b10);
        // ...and neither carry pair may have been matched to the other.
        assert_eq!(want[4..], [ALL_FACES; 4]);
    }

    #[test]
    fn exposed_faces_unindexed_rejects_duplicates() {
        let coords: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 1, 2, 3];
        let err = exposed_faces_unindexed(&coords, 3).unwrap_err();
        assert!(err.contains("duplicate"), "{err}");
        assert!(err.contains("(1, 2, 3)"), "{err}");
        assert!(err.contains("rows 0 and 2"), "{err}");
    }

    #[test]
    fn adjacency_pairs_are_deduped_sorted_and_connectivity_sensitive() {
        // A line of 4 alternating between two (negative, non-dense) labels:
        // three inter-label adjacencies collapse to one distinct pair.
        let coords: Vec<i32> = (0..4).flat_map(|x| [x, 0, 0]).collect();
        let index = SpatialIndex::build(&coords, 4, IndexKind::Hash).unwrap();
        let offs = build_offsets(6, [1.0; 3]);
        let pairs = label_adjacency(&coords, 4, &index, &offs, &[-5, 9, -5, 9]);
        assert_eq!(pairs, vec![(-5, 9)]);

        // Same label everywhere -> no pairs at all.
        assert!(label_adjacency(&coords, 4, &index, &offs, &[3; 4]).is_empty());

        // Diagonal touch only counts at 18/26-connectivity.
        let diag: Vec<i32> = vec![0, 0, 0, 1, 1, 0];
        let index = SpatialIndex::build(&diag, 2, IndexKind::Sorted).unwrap();
        let labels = [2i64, 1];
        let offs26 = build_offsets(26, [1.0; 3]);
        assert_eq!(
            label_adjacency(&diag, 2, &index, &offs26, &labels),
            vec![(1, 2)]
        );
        let offs6 = build_offsets(6, [1.0; 3]);
        assert!(label_adjacency(&diag, 2, &index, &offs6, &labels).is_empty());
    }
}
