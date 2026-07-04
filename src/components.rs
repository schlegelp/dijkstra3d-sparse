//! Connected components over the implicit sparse grid via union-find.
//! Uses the same neighbour probe as Dijkstra — no edge list is built.

use crate::index::{key_of, SpatialIndex};

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

/// Label the connected components of the voxel set. Labels are assigned in
/// order of first appearance by row index (0, 1, 2, ...), so the output is
/// deterministic. Returns `(n_components, labels)`.
pub fn connected_components(
    coords: &[i32],
    n: usize,
    index: &SpatialIndex,
    offsets: &[(i32, i32, i32, f64)],
) -> (usize, Vec<i32>) {
    let mut dsu = Dsu::new(n);

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
                dsu.union(row as u32, nbr);
            }
        }
    }

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
        let (n, labels) = connected_components(&coords, 3, &index, &offs26);
        assert_eq!(n, 2);
        assert_eq!(labels, vec![0, 0, 1]);

        let offs6 = build_offsets(6, [1.0; 3]);
        let (n, labels) = connected_components(&coords, 3, &index, &offs6);
        assert_eq!(n, 3);
        assert_eq!(labels, vec![0, 1, 2]);
    }
}
