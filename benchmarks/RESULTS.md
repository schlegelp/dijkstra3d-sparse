# Benchmark results

Workload: thin helical tube (radius-3 cross-section) sweeping a cylinder,
N = 1,502,554 voxels in a bounding box of 16,267 x 4,005 x 4,006 = 2.61e+11 cells.
Single-source Dijkstra, 26-connectivity, geometric costs (identical inputs
for every row). Each run is a fresh subprocess; peak RSS includes the
~60 MiB interpreter + NumPy/SciPy import baseline.

| task | index | N | wall time | peak RSS | RSS bytes / voxel |
|---|---|---|---|---|---|
| dijkstra | sorted | 1,502,554 | 1.53 s | 452.8 MiB | 316 |
| dijkstra | hash | 1,502,554 | 0.41 s | 472.9 MiB | 330 |
| components | sorted | 1,502,554 | 0.72 s | 430.0 MiB | 300 |
| components | hash | 1,502,554 | 0.25 s | 448.4 MiB | 313 |
| scipy-csr | - | 1,502,554 | 3.54 s | 1.6 GiB | 1125 |

`scipy-csr` is `scipy.sparse.csgraph.dijkstra` on the explicit CSR graph of the same grid (30,482,462 stored edges): 3.40 s to build the CSR (vectorized NumPy) + 0.14 s to solve. Materializing the edge list is exactly the work and memory the implicit-grid walk avoids.

Dense baseline for the same bounding box (dist f64 + pred i64 + visited u8 = 17 B/cell): **4.0 TiB** — 2,624x the largest peak RSS measured here.

Memory gate: peak RSS must stay O(N), far below the dense bbox baseline.
