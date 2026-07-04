//! Multi-source Dijkstra over the implicit sparse grid.
//!
//! Identical in structure to dense `dijkstra3d`: binary heap with lazy
//! deletion, edge relaxation, parent tracking. The only difference is that
//! a neighbour probe is a spatial-index lookup instead of a dense array
//! read. All working memory is O(N).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::index::{key_of, SpatialIndex};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CostMode {
    Geometric,
    Vertex,
    Additive,
}

impl CostMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "geometric" => Ok(CostMode::Geometric),
            "vertex" => Ok(CostMode::Vertex),
            "additive" => Ok(CostMode::Additive),
            other => Err(format!(
                "cost_mode must be 'vertex', 'additive' or 'geometric', got '{other}'"
            )),
        }
    }
}

/// Everything that stays fixed across (possibly repeated) Dijkstra runs.
pub struct Problem<'a> {
    /// Row-major `(N, 3)` voxel coordinates.
    pub coords: &'a [i32],
    pub n: usize,
    pub index: &'a SpatialIndex,
    /// `(dx, dy, dz, step_length)` neighbour offsets.
    pub offsets: &'a [(i32, i32, i32, f64)],
    pub node_cost: Option<&'a [f32]>,
    pub cost_mode: CostMode,
    pub free_mask: Option<&'a [bool]>,
    pub free_eps: f64,
}

impl Problem<'_> {
    /// Cost of the directed edge `cur -> nbr`. `free_mask[nbr]` overrides
    /// the *total* edge cost with `free_eps` (small, strictly positive, so
    /// the Dijkstra invariant holds).
    #[inline(always)]
    fn edge_cost(&self, nbr: usize, step_len: f64) -> f64 {
        if let Some(mask) = self.free_mask {
            if mask[nbr] {
                return self.free_eps;
            }
        }
        match self.cost_mode {
            CostMode::Geometric => step_len,
            CostMode::Vertex => self.node_cost.expect("node_cost required")[nbr] as f64 * step_len,
            CostMode::Additive => {
                self.node_cost.expect("node_cost required")[nbr] as f64 + step_len
            }
        }
    }
}

/// Run multi-source Dijkstra, writing into caller-provided `dist` / `pred`
/// slices (length N each). Unreached voxels keep `+inf` / `-1`; sources get
/// `dist = 0`, `pred = -1`. `pred` holds row indices into `coords`.
///
/// Determinism: heap entries are `(distance_bits, row)` so ties break on the
/// smaller row index; relaxation uses strict `<`, so among equal-cost
/// predecessors the first one found (fixed offset order) wins. Output is
/// therefore reproducible across runs and platforms.
pub fn dijkstra(p: &Problem<'_>, sources: &[i64], dist: &mut [f64], pred: &mut [i64]) {
    dist.fill(f64::INFINITY);
    pred.fill(-1);
    let mut settled = vec![false; p.n];

    // Distances are finite and non-negative, so `f64::to_bits` is a
    // monotone map to u64 — the heap orders raw bits, no float-ord wrapper.
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    for &s in sources {
        let s = s as usize;
        dist[s] = 0.0;
        heap.push(Reverse((0u64, s as u32)));
    }

    while let Some(Reverse((dbits, row))) = heap.pop() {
        let row_us = row as usize;
        if settled[row_us] || dbits != dist[row_us].to_bits() {
            continue; // lazy deletion: stale heap entry
        }
        settled[row_us] = true;
        let d = dist[row_us];
        let b = row_us * 3;
        let (x, y, z) = (p.coords[b], p.coords[b + 1], p.coords[b + 2]);

        for &(dx, dy, dz, step_len) in p.offsets {
            // i64 arithmetic so coordinates at the i32 boundary can't overflow
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
            let Some(nbr) = p.index.get(key_of(nx as i32, ny as i32, nz as i32)) else {
                continue;
            };
            let nbr_us = nbr as usize;
            if settled[nbr_us] {
                continue;
            }
            let nd = d + p.edge_cost(nbr_us, step_len);
            if nd < dist[nbr_us] {
                dist[nbr_us] = nd;
                pred[nbr_us] = row as i64;
                heap.push(Reverse((nd.to_bits(), nbr)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexKind, SpatialIndex};
    use crate::offsets::build_offsets;

    fn line(n: i32) -> Vec<i32> {
        (0..n).flat_map(|x| [x, 0, 0]).collect()
    }

    #[test]
    fn straight_line_geometric() {
        let coords = line(5);
        let index = SpatialIndex::build(&coords, 5, IndexKind::Sorted).unwrap();
        let offsets = build_offsets(26, [1.0; 3]);
        let p = Problem {
            coords: &coords,
            n: 5,
            index: &index,
            offsets: &offsets,
            node_cost: None,
            cost_mode: CostMode::Geometric,
            free_mask: None,
            free_eps: 1e-6,
        };
        let mut dist = vec![0.0; 5];
        let mut pred = vec![0i64; 5];
        dijkstra(&p, &[0], &mut dist, &mut pred);
        for i in 0..5 {
            assert!((dist[i] - i as f64).abs() < 1e-12);
        }
        assert_eq!(pred, vec![-1, 0, 1, 2, 3]);
    }

    #[test]
    fn multi_source_and_unreachable() {
        // two voxels far apart -> second cluster unreachable
        let coords: Vec<i32> = vec![0, 0, 0, 1, 0, 0, 100, 100, 100];
        let index = SpatialIndex::build(&coords, 3, IndexKind::Hash).unwrap();
        let offsets = build_offsets(6, [1.0; 3]);
        let p = Problem {
            coords: &coords,
            n: 3,
            index: &index,
            offsets: &offsets,
            node_cost: None,
            cost_mode: CostMode::Geometric,
            free_mask: None,
            free_eps: 1e-6,
        };
        let mut dist = vec![0.0; 3];
        let mut pred = vec![0i64; 3];
        dijkstra(&p, &[0, 1], &mut dist, &mut pred);
        assert_eq!(dist[0], 0.0);
        assert_eq!(dist[1], 0.0);
        assert!(dist[2].is_infinite());
        assert_eq!(pred, vec![-1, -1, -1]);
    }
}
