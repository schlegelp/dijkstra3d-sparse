//! Sparse spatial index: `(x, y, z) -> compact row index [0, N)`.
//!
//! This is the single component that replaces `dijkstra3d`'s dense
//! bbox-sized array, making all working memory `O(N)`. Coordinates are
//! packed into an order-preserving `u128` key (32 bits per axis, raw i32
//! range — no extent limit), and looked up through one of two
//! interchangeable backends:
//!
//! 1. `Sorted` — keys sorted once, neighbours probed by binary search.
//!    Allocation-free probes, cache-friendly, deterministic.
//! 2. `Hash` — `FxHashMap<u128, u32>`, O(1) probes, better amortized at
//!    large N at the price of hash-map overhead.

use rustc_hash::FxHashMap;

/// Order-preserving injection `(i32, i32, i32) -> u128`.
/// XOR with the sign bit maps i32 to u32 monotonically, so the full i32
/// coordinate range is supported (no 21-bit packing limit).
#[inline(always)]
pub fn key_of(x: i32, y: i32, z: i32) -> u128 {
    let ux = (x as u32) ^ 0x8000_0000;
    let uy = (y as u32) ^ 0x8000_0000;
    let uz = (z as u32) ^ 0x8000_0000;
    ((ux as u128) << 64) | ((uy as u128) << 32) | (uz as u128)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IndexKind {
    Sorted,
    Hash,
}

impl IndexKind {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "sorted" => Ok(IndexKind::Sorted),
            "hash" => Ok(IndexKind::Hash),
            other => Err(format!(
                "index_kind must be 'sorted' or 'hash', got '{other}'"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            IndexKind::Sorted => "sorted",
            IndexKind::Hash => "hash",
        }
    }
}

#[derive(Debug)]
pub enum SpatialIndex {
    Sorted { keys: Vec<u128>, rows: Vec<u32> },
    Hash(FxHashMap<u128, u32>),
}

impl SpatialIndex {
    /// Build from row-major `(N, 3)` coordinates (`coords.len() == 3 * n`).
    /// Errors on duplicate coordinates — duplicates would silently alias
    /// rows in every downstream algorithm.
    pub fn build(coords: &[i32], n: usize, kind: IndexKind) -> Result<Self, String> {
        debug_assert_eq!(coords.len(), 3 * n);
        match kind {
            IndexKind::Sorted => {
                let mut pairs: Vec<(u128, u32)> = (0..n)
                    .map(|row| {
                        let b = row * 3;
                        (key_of(coords[b], coords[b + 1], coords[b + 2]), row as u32)
                    })
                    .collect();
                pairs.sort_unstable();
                for w in pairs.windows(2) {
                    if w[0].0 == w[1].0 {
                        return Err(duplicate_msg(coords, w[0].1, w[1].1));
                    }
                }
                let mut keys = Vec::with_capacity(n);
                let mut rows = Vec::with_capacity(n);
                for (k, r) in pairs {
                    keys.push(k);
                    rows.push(r);
                }
                Ok(SpatialIndex::Sorted { keys, rows })
            }
            IndexKind::Hash => {
                let mut map = FxHashMap::default();
                map.reserve(n);
                for row in 0..n {
                    let b = row * 3;
                    let key = key_of(coords[b], coords[b + 1], coords[b + 2]);
                    if let Some(prev) = map.insert(key, row as u32) {
                        return Err(duplicate_msg(coords, prev, row as u32));
                    }
                }
                Ok(SpatialIndex::Hash(map))
            }
        }
    }

    #[inline(always)]
    pub fn get(&self, key: u128) -> Option<u32> {
        match self {
            SpatialIndex::Sorted { keys, rows } => keys.binary_search(&key).ok().map(|i| rows[i]),
            SpatialIndex::Hash(map) => map.get(&key).copied(),
        }
    }
}

pub fn duplicate_msg(coords: &[i32], a: u32, b: u32) -> String {
    let base = a as usize * 3;
    format!(
        "duplicate voxel coordinate ({}, {}, {}) at rows {} and {}",
        coords[base],
        coords[base + 1],
        coords[base + 2],
        a.min(b),
        a.max(b)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_injective_and_handles_negatives() {
        let pts = [
            (0, 0, 0),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
            (1, 2, 3),
            (i32::MIN, i32::MAX, 0),
            (i32::MAX, i32::MIN, -5),
        ];
        let mut keys: Vec<u128> = pts.iter().map(|&(x, y, z)| key_of(x, y, z)).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), pts.len());
    }

    #[test]
    fn build_and_probe_both_kinds() {
        let coords: Vec<i32> = vec![0, 0, 0, 5, -3, 2, -7, 8, 9];
        for kind in [IndexKind::Sorted, IndexKind::Hash] {
            let idx = SpatialIndex::build(&coords, 3, kind).unwrap();
            assert_eq!(idx.get(key_of(0, 0, 0)), Some(0));
            assert_eq!(idx.get(key_of(5, -3, 2)), Some(1));
            assert_eq!(idx.get(key_of(-7, 8, 9)), Some(2));
            assert_eq!(idx.get(key_of(1, 1, 1)), None);
        }
    }

    #[test]
    fn duplicates_rejected() {
        let coords: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 1, 2, 3];
        for kind in [IndexKind::Sorted, IndexKind::Hash] {
            let err = SpatialIndex::build(&coords, 3, kind).unwrap_err();
            assert!(err.contains("duplicate"), "{err}");
            assert!(err.contains("(1, 2, 3)"), "{err}");
        }
    }
}
