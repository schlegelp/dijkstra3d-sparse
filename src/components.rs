//! Connected components and label adjacency over the implicit sparse grid.
//! Both walk the same neighbour probe as Dijkstra — no edge list is built.

use crate::index::{key_of, SpatialIndex};
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
const FACES: [(i32, i32, i32); 6] = [
    (1, 0, 0),  // bit 0: +x
    (-1, 0, 0), // bit 1: -x
    (0, 1, 0),  // bit 2: +y
    (0, -1, 0), // bit 3: -y
    (0, 0, 1),  // bit 4: +z
    (0, 0, -1), // bit 5: -z
];

/// Per voxel, a 6-bit mask of which face-neighbours are *absent* from the
/// set. This is the surface-extraction probe `find_surface_voxels` runs —
/// the same coordinate walk as `connected_components`, minus the union-find:
/// one pass, no `(N, 3)` neighbour temporaries, and (in the free-function
/// binding) the spatial index is built and freed within the call.
///
/// Bit `k` of `mask[row]` is set iff `voxels[row] + FACES[k]` is not in the
/// set. A fully interior voxel yields `0`; a lone voxel yields `0b111111`
/// (`63`). Directional by nature — the `half`-offset trick the pair
/// primitives use does not apply, so every row probes all six offsets.
pub fn exposed_faces(coords: &[i32], n: usize, index: &SpatialIndex) -> Vec<u8> {
    let mut mask = vec![0u8; n];
    for (row, m) in mask.iter_mut().enumerate() {
        let b = row * 3;
        let (x, y, z) = (coords[b], coords[b + 1], coords[b + 2]);
        for (k, &(dx, dy, dz)) in FACES.iter().enumerate() {
            // i64 arithmetic + bounds check, as elsewhere: a neighbour that
            // overflows i32 cannot be a voxel, so its face is exposed.
            let nx = x as i64 + dx as i64;
            let ny = y as i64 + dy as i64;
            let nz = z as i64 + dz as i64;
            let present = nx >= i32::MIN as i64
                && nx <= i32::MAX as i64
                && ny >= i32::MIN as i64
                && ny <= i32::MAX as i64
                && nz >= i32::MIN as i64
                && nz <= i32::MAX as i64
                && index.get(key_of(nx as i32, ny as i32, nz as i32)).is_some();
            if !present {
                *m |= 1 << k;
            }
        }
    }
    mask
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
