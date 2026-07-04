# Sparse-voxel Dijkstra — standalone library handoff

> Working name: **`sparsedijkstra`** (rename at will). A Rust core with Python
> bindings that computes shortest paths, distance fields and connected components
> over **sparse** 3D voxel sets given as an `(N, 3)` integer coordinate array —
> a sparse analogue of [seung-lab/dijkstra3d](https://github.com/seung-lab/dijkstra3d),
> which operates on *dense* 3D arrays. This document is a self-contained
> implementation spec for a fresh repository; it has no dependency on, and makes no
> reference to, any existing codebase.

## 1. Purpose & motivating idea

`dijkstra3d` is fast because it does **not** build an explicit graph. It walks an
*implicit* rectangular grid: a voxel at linear index `loc` has neighbours
`loc + offset` (`±1`, `±sx`, `±sx·sy`, …), and the dense array is merely a hash
`coordinate → payload` (field value, distance, parent). The only thing making it
"dense" — and the reason it needs memory proportional to the **bounding-box
volume** `W·H·D` — is that this hash is a dense array sized to the full box.

For sparse objects (a thin structure inside a large box, `N ≪ W·H·D`) that is
wasteful. This library keeps the implicit-grid walk but swaps the one dense
component:

> **dense-array-indexed-by-bbox  →  sparse hash `coordinate → compact index [0, N)`**

Everything else — binary heap, edge relaxation, parent tracking, path
reconstruction — is unchanged. Neighbours are still generated implicitly by
coordinate offset; we probe a hash instead of reading a dense cell. **No adjacency
list is ever materialized** (this is explicitly *not* a CSR / explicit-graph
Dijkstra), and all working memory is `O(N)`, independent of the bounding box.

|                     | Explicit-graph Dijkstra (CSR) | Dense `dijkstra3d`   | **This library (sparse)**            |
|---------------------|-------------------------------|----------------------|--------------------------------------|
| Graph               | `~26·N` edges stored           | implicit grid        | **implicit grid (0 edges stored)**   |
| coord → payload     | node index table              | dense array `[W·H·D]`| sparse hash / sorted keys → `[0, N)` |
| Working memory      | `O(N) + O(26·N)` edges         | `O(W·H·D)`           | **`O(N)`**                           |
| Neighbour lookup    | precomputed edge list         | index arithmetic     | coord offset + hash probe            |

## 2. Scope (v1)

A focused core, designed to grow. **In scope for v1:**

- Single-source and **multi-source** Dijkstra returning a distance field + a
  predecessor field over the sparse voxels.
- **Connected components** over the same implicit graph (union-find).
- Path reconstruction from the predecessor field.
- 6 / 18 / 26 connectivity, anisotropic voxel spacing.
- Configurable edge-cost model (see §4), including an optional per-voxel node cost
  and an optional "free-node" mask.

**Deferred (design so these slot in later, don't build yet):** A\*/compass
heuristic, bidirectional search, a Euclidean/chamfer distance-transform field
(`euclidean_distance_field`), and GPU/threaded scaling beyond per-component
parallelism.

## 3. Public API (Python)

Inputs are NumPy arrays; the field is aligned 1:1 with the voxel rows (the key
difference from `dijkstra3d`, whose field is a dense 3D array).

```python
def dijkstra_field(
    voxels:    "np.ndarray[(N,3), int32]",   # integer voxel coords; unsorted OK
    sources:   "np.ndarray[(S,), int64]",    # row indices into `voxels`
    *,
    node_cost:  "np.ndarray[(N,), float32] | None" = None,  # per-voxel weight (see §4)
    connectivity: int = 26,                  # 6 | 18 | 26
    anisotropy:  "tuple[float,float,float]" = (1.0, 1.0, 1.0),
    cost_mode:  str = "vertex",              # "vertex" | "additive" | "geometric"
    free_mask:  "np.ndarray[(N,), bool] | None" = None,  # incoming edges to these ~free
    free_eps:   float = 1e-6,
    min_only:   bool = True,                  # multi-source: distance to nearest source
) -> "tuple[np.ndarray[(N,), float64], np.ndarray[(N,), int64]]":
    """Returns (dist, pred). Unreached voxels: dist = +inf, pred = -1.
       Source voxels: pred = -1. `pred` indexes into `voxels`."""

def path(pred: "np.ndarray[(N,), int64]", target: int) -> "np.ndarray[(M,3), int32]":
    """Reconstruct target -> source path by walking predecessors."""

def connected_components(
    voxels: "np.ndarray[(N,3), int32]", *, connectivity: int = 26,
) -> "tuple[int, np.ndarray[(N,), int32]]":
    """(n_components, labels)."""
```

Design notes:
- Returning `(dist, pred)` (rather than a bundled object) makes it a **drop-in** for
  `scipy.sparse.csgraph.dijkstra(..., return_predecessors=True)`, easing migration
  from CSR-based code. `-1` predecessor sentinel matches SciPy's "no predecessor".
- `sources` as row indices (not coordinates) avoids a hash lookup at the boundary;
  offer a small helper `index_of(voxels, coords)` for callers holding coordinates.
- Keep everything importable and usable without the caller ever constructing a
  graph object.

## 4. Edge-cost model (correctness-critical)

Precompute per-offset **step-length multipliers** from `anisotropy = (wx, wy, wz)`,
matching `dijkstra3d` exactly (verified against its `dijkstra3d.hpp`):

- axis moves: `wx`, `wy`, `wz`
- face diagonals (18-conn): `sqrt(wa² + wb²)`
- corner diagonals (26-conn): `sqrt(wa² + wb² + wc²)`

Then `cost(cur → nbr)` depends on `cost_mode`:

| mode        | `cost(cur → nbr)`                    | matches                                   |
|-------------|--------------------------------------|-------------------------------------------|
| `geometric` | `length_mult[offset]`                | anisotropic geodesic distance             |
| `vertex`    | `node_cost[nbr] · length_mult[offset]` | `dijkstra3d` vertex-weighting (its default) |
| `additive`  | `length_mult[offset] + node_cost[nbr]` | "geometric length + a per-voxel penalty" (integral of a penalty field along a path) |

`free_mask`: when set and `free_mask[nbr]` is true, the **total** edge cost into
`nbr` is `free_eps` (a small strictly-positive value that keeps Dijkstra
well-behaved). This supports incremental/grafting path extraction where later paths
should ride an already-selected node set for ~free before diverging.

Weights must be finite and non-negative. Validate at the boundary; document that
negative costs are unsupported (Dijkstra invariant).

## 5. Rust internals

- **Spatial index.** Pack each shifted coordinate into a `u64` Morton/linear key
  (e.g. `(x << 42) | (y << 21) | z` after subtracting `min - 1` per axis; 21 bits ⇒
  extent `< 2²¹` per axis — document this, or key on the raw `(i32,i32,i32)` triple
  in the map to lift the limit). Two interchangeable implementations, pick by
  benchmark:
  1. Sort the `N` keys once; probe neighbours by **binary search** (allocation-free,
     cache-friendly, deterministic).
  2. `rustc_hash::FxHashMap<u64, u32>` (`key → row index`) — `O(1)` probes, better
     amortized at large `N`, ~8–16 B/entry overhead.
  Each popped node generates ≤26 candidate neighbour keys and probes them; present
  candidates are relaxed. This is the whole "implicit graph."
- **Label arrays** sized `O(N)`: `dist: Vec<f64>` (init `+inf`), `pred: Vec<i64>`
  (init `-1`), plus a `settled` bitset. **Not** `O(bbox)` — the memory win.
- **Priority queue:** `std::collections::BinaryHeap` with lazy deletion (skip a
  popped entry whose recorded `dist` is stale). Float keys via `ordered-float` or a
  monotone `u64` bit-mapping of `f64`. (Weights are floats, so bucket/radix heaps
  don't apply.)
- **Multi-source:** seed the heap with all `sources` at distance 0; standard
  multi-source Dijkstra yields `min_only` distances and predecessors pointing back
  to each voxel's nearest source.
- **Connected components:** union-find over the implicit graph using the same probe;
  no edge list needed.
- **Determinism:** break heap ties by row index so output is reproducible across
  runs/platforms (ties otherwise yield different but equally-valid shortest paths).
- **Parallelism (optional, later):** connected components are independent ⇒
  per-component Dijkstra is embarrassingly parallel via `rayon`. Land correctness
  single-threaded first.

## 6. Rust / packaging stack

- **Bindings:** `pyo3`.
- **NumPy interop:** `rust-numpy` / `numpy` crate — zero-copy `PyReadonlyArray2<i32>`
  → `ndarray::ArrayView2` in; `PyArray1::from_vec` out.
- **Build backend:** `maturin` (PEP 517) via `pyproject.toml`; mixed layout so pure
  Python (type stubs, thin wrappers) and the compiled `_native` module ship in one
  package.
- **Crates:** `rustc-hash`, `ordered-float`, optional `rayon`.
- **Wheels:** build with `pyo3/abi3-py39` so **one** wheel per platform covers Python
  3.9–3.13; use `maturin-action` / `cibuildwheel` in CI (Linux `manylinux`, macOS
  arm64+x86_64, Windows). Provide an sdist so `pip install` can build from source
  where no wheel matches.
- **Python floor:** 3.9+. Type stubs (`.pyi`) for the public API.

## 7. Testing & verification

1. **Parity vs SciPy.** On random sparse clouds, assert `dijkstra_field` `dist`
   matches `scipy.sparse.csgraph.dijkstra` on the equivalent explicit CSR graph to
   `~1e-5`. Predecessors may differ on ties → assert reconstructed **path cost**
   equality, not node-for-node identity. Same for `connected_components` vs SciPy
   labels (up to label permutation).
2. **Cost-mode coverage.** Unit tests for each of `geometric` / `vertex` / `additive`
   and each connectivity (6/18/26), including anisotropic spacing, `free_mask`
   behaviour, unreachable voxels (`+inf` / `-1`), single-voxel and empty inputs.
3. **Property tests.** `dist(source)=0`; triangle-inequality along recovered paths;
   `path()` endpoints and 26-adjacency of consecutive nodes.
4. **Benchmark + memory gate.** A large synthetic sparse structure (e.g. a long
   thin tube in a big bounding box): record wall-time and **peak RSS**, and assert
   memory scales with `N`, not `W·H·D` (contrast with a dense baseline). This is the
   headline value proposition — measure it explicitly.
5. **CI:** run tests on all target platforms/Pythons; build and smoke-import wheels.

## 8. First-PR checklist

- [ ] Repo scaffold: `pyproject.toml` (maturin), `Cargo.toml`, `src/lib.rs`, Python
      package with `.pyi` stubs.
- [ ] Spatial index (sorted-keys binary search first; FxHashMap behind a bench).
- [ ] Single-source Dijkstra (`geometric` mode) + `path()` reconstruction.
- [ ] `node_cost`, `cost_mode` (`vertex`/`additive`), `free_mask`, anisotropy,
      6/18/26 connectivity.
- [ ] Multi-source + `min_only`.
- [ ] `connected_components` (union-find).
- [ ] Test suite §7.1–§7.3 green; benchmark §7.4 recorded.
- [ ] CI wheels (abi3) for Linux/macOS/Windows × 3.9–3.13; sdist.
- [ ] README with the §1 pitch, API, and a runnable example.

## 9. Open decisions for the implementer

- Final library name and PyPI slug.
- 21-bit packed keys (fast, extent-limited) vs raw-triple map keys (unbounded).
  Recommendation: raw-triple or 64-bit-with-wider-fields map for a general library;
  keep packing only if benchmarks favour it.
- Whether `vertex` (dijkstra3d-compatible) or `additive` is the documented default.
  Recommendation: `vertex`, to match the well-known `dijkstra3d` semantics.
- `f32` vs `f64` for the distance field (precision vs memory on huge `N`).
