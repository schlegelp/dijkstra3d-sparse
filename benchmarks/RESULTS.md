# Benchmark results

Workload: thin helical tube (radius-3 cross-section) sweeping a cylinder,
N = 1,502,554 voxels in a bounding box of 16,267 x 4,005 x 4,006 = 2.61e+11 cells.
Single-source Dijkstra, 26-connectivity, geometric costs (identical inputs
for every row). Each run is a fresh subprocess; peak RSS includes the
~60 MiB interpreter + NumPy/SciPy import baseline.

| task | index | N | wall time | peak RSS | RSS bytes / voxel |
|---|---|---|---|---|---|
| dijkstra | sorted | 1,502,554 | 1.42 s | 470.0 MiB | 328 |
| dijkstra | hash | 1,502,554 | 0.42 s | 490.0 MiB | 342 |
| components | sorted | 1,502,554 | 0.69 s | 448.2 MiB | 313 |
| components | hash | 1,502,554 | 0.24 s | 468.3 MiB | 327 |
| scipy-csr | - | 1,502,554 | 3.59 s | 1.6 GiB | 1126 |
| graft | - | 1,502,554 | 1.27 s | 522.6 MiB | 365 |
| reuse | hash | 1,502,554 | 0.72 s | 579.2 MiB | 404 |
| wavefront | hash | 1,502,554 | 1.14 s | 514.4 MiB | 359 |
| wavefront-edges | hash | 1,502,554 | 5.66 s | 3.6 GiB | 2595 |

`scipy-csr` is `scipy.sparse.csgraph.dijkstra` on the explicit CSR graph of the same grid (30,482,462 stored edges): 3.47 s to build the CSR (vectorized NumPy) + 0.12 s to solve. Materializing the edge list is exactly the work and memory the implicit-grid walk avoids.

`graft` is incremental grafting via early-terminating `stop_mask` queries: 60 random query voxels connected one after another to a growing anchor set (each returned path joins the anchors). Mean voxels touched per query: first 10 queries 150,401 (10.0% of N) → last 10 queries 4,589 (0.3% of N). A full field per query would touch 100% of N every time; the whole 60-query loop ran in 1.27 s.

`reuse` is the reusable `Graph` handle (hash index): K early-terminating
`shortest_path_to_set` queries over the same voxels (anchors on every 64th
voxel, so each search is tiny and per-call cost is dominated by the spatial-
index build). The free functions rebuild that index on every call; one
`Graph` builds it once (handle construction is included in its time):

| K queries | free functions | one `Graph` + K calls | speedup |
|---|---|---|---|
| 2 | 0.03 s | 0.01 s | 2.5x |
| 5 | 0.06 s | 0.01 s | 4.7x |
| 50 | 0.56 s | 0.03 s | 19.5x |

`wavefront` is the graph half of wavefront skeletonization over one
`Graph`: components → geodesic field → components of each geodesic level
set (`connected_components(group=level)`) → which of those rings touch
(`label_adjacency`). `wavefront-edges` is the same pipeline for a caller
without those two primitives: the explicit edge list, SciPy connected
components on the same-level sub-graph, and a NumPy `unique` to contract.
Both agree on the result (2,430 rings, 2,429 ring edges); the
field stage is shared and identical.

| stage | coordinate-native | edge list |
|---|---|---|
| components + geodesic field | 0.66 s | 0.59 s |
| edge list (30,482,462 rows) | — | 3.31 s |
| rings (components per level) | 0.24 s | 0.64 s |
| contract onto rings | 0.23 s | 1.11 s |
| **total** | **1.14 s** | **5.66 s** |
| **peak RSS** | **514.4 MiB** | **3.6 GiB** |

The edge list is the entire difference: 30,482,462 adjacencies materialized to yield 2,429 distinct ring pairs. `label_adjacency`
deduplicates as it probes, so that intermediate never exists.

Dense baseline for the same bounding box (dist f64 + pred i64 + visited u8 = 17 B/cell): **4.0 TiB** — 1,138x the largest peak RSS measured here.

Memory gate: peak RSS must stay O(N), far below the dense bbox baseline.
