# Benchmark results

Workload: thin helical tube (radius-3 cross-section) sweeping a cylinder,
N = 1,502,554 voxels in a bounding box of 16,267 x 4,005 x 4,006 = 2.61e+11 cells.
Single-source Dijkstra, 26-connectivity, geometric costs (identical inputs
for every row). Each run is a fresh subprocess; peak RSS includes the
~60 MiB interpreter + NumPy/SciPy import baseline.

| task | index | N | wall time | peak RSS | RSS bytes / voxel |
|---|---|---|---|---|---|
| dijkstra | sorted | 1,502,554 | 1.51 s | 470.3 MiB | 328 |
| dijkstra | hash | 1,502,554 | 0.43 s | 490.4 MiB | 342 |
| components | sorted | 1,502,554 | 0.75 s | 445.8 MiB | 311 |
| components | hash | 1,502,554 | 0.25 s | 466.3 MiB | 325 |
| scipy-csr | - | 1,502,554 | 3.74 s | 1.6 GiB | 1126 |
| graft | - | 1,502,554 | 1.37 s | 521.6 MiB | 364 |
| reuse | hash | 1,502,554 | 0.84 s | 580.0 MiB | 405 |

`scipy-csr` is `scipy.sparse.csgraph.dijkstra` on the explicit CSR graph of the same grid (30,482,462 stored edges): 3.60 s to build the CSR (vectorized NumPy) + 0.13 s to solve. Materializing the edge list is exactly the work and memory the implicit-grid walk avoids.

`graft` is incremental grafting via early-terminating `stop_mask` queries: 60 random query voxels connected one after another to a growing anchor set (each returned path joins the anchors). Mean voxels touched per query: first 10 queries 150,401 (10.0% of N) → last 10 queries 4,589 (0.3% of N). A full field per query would touch 100% of N every time; the whole 60-query loop ran in 1.37 s.

`reuse` is the reusable `Graph` handle (hash index): K early-terminating
`shortest_path_to_set` queries over the same voxels (anchors on every 64th
voxel, so each search is tiny and per-call cost is dominated by the spatial-
index build). The free functions rebuild that index on every call; one
`Graph` builds it once (handle construction is included in its time):

| K queries | free functions | one `Graph` + K calls | speedup |
|---|---|---|---|
| 2 | 0.03 s | 0.02 s | 2.1x |
| 5 | 0.07 s | 0.01 s | 5.0x |
| 50 | 0.64 s | 0.05 s | 13.9x |

Dense baseline for the same bounding box (dist f64 + pred i64 + visited u8 = 17 B/cell): **4.0 TiB** — 2,622x the largest peak RSS measured here.

Memory gate: peak RSS must stay O(N), far below the dense bbox baseline.
