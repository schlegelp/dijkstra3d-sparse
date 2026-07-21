"""Benchmark + memory gate (spec §7.4).

Workload: a thin helical tube winding through a huge bounding box — the
motivating case where N (voxel count) is tiny relative to W*H*D (bbox
volume). Each measurement runs in a fresh subprocess so peak RSS is
attributable to that one workload.

Usage:
    python benchmarks/bench.py                 # full suite, prints a report
    python benchmarks/bench.py --quick         # smaller N for smoke runs
    python benchmarks/bench.py --write-results # also update RESULTS.md

The dense baseline is *computed*, not run: a dense solver needs at minimum
dist(f64) + pred(i64) + visited(1B) = 17 bytes per bbox cell, which for this
workload would be hundreds of GiB.
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DENSE_BYTES_PER_CELL = 17  # dist f64 + pred i64 + visited u8


def helix_tube(n_target: int, radius: int = 3, helix_r: float = 2000.0) -> np.ndarray:
    """~n_target unique voxels forming a thin tube on a helix. The helix
    sweeps a cylinder of radius ~2*helix_r, so the bbox volume is enormous
    while the voxel count stays ~n_target."""
    ball = [
        (dx, dy, dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
        if dx * dx + dy * dy + dz * dz <= radius * radius
    ]
    ball = np.array(ball, dtype=np.int64)  # ~123 voxels for r=3

    # roughly one center per voxel step along the curve; oversample 3x so
    # consecutive balls overlap into a connected tube
    n_centers = max(2, (3 * n_target) // len(ball))
    t = np.linspace(0.0, 1.0, n_centers)
    length = n_centers / 3.0  # ~1 voxel of x-advance per 3 centers
    centers = np.stack(
        [
            t * length,
            helix_r * np.sin(t * length * 2 * np.pi / 5000.0),
            helix_r * np.cos(t * length * 2 * np.pi / 5000.0),
        ],
        axis=1,
    ).astype(np.int64)

    vox = (centers[:, None, :] + ball[None, :, :]).reshape(-1, 3)
    vox = np.unique(vox, axis=0)
    return vox.astype(np.int32)


def peak_rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru if sys.platform == "darwin" else ru * 1024  # linux reports KiB


def build_csr(vox: np.ndarray, connectivity: int = 26):
    """Explicit CSR graph of the implicit grid, built with vectorized
    sorted-key lookups (no Python-level per-voxel loop) so the SciPy
    baseline is not handicapped by construction overhead."""
    from scipy.sparse import csr_matrix

    v = (vox - vox.min(axis=0)).astype(np.int64) + 1  # 1-cell pad: shifts stay >= 0
    extent = v.max(axis=0) + 2

    def pack(a):
        return (a[:, 0] * extent[1] + a[:, 1]) * extent[2] + a[:, 2]

    keys = pack(v)
    order = np.argsort(keys).astype(np.int32)
    skeys = keys[order]

    max_manhattan = {6: 1, 18: 2, 26: 3}[connectivity]
    rows, cols, data = [], [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if manhattan == 0 or manhattan > max_manhattan:
                    continue
                nkeys = pack(v + (dx, dy, dz))
                pos = np.minimum(np.searchsorted(skeys, nkeys), len(skeys) - 1)
                found = skeys[pos] == nkeys
                src = np.nonzero(found)[0].astype(np.int32)
                rows.append(src)
                cols.append(order[pos[found]])
                data.append(np.full(len(src), np.sqrt(manhattan)))
    n = len(vox)
    return csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )


def edge_pairs(vox: np.ndarray, connectivity: int = 26) -> np.ndarray:
    """Explicit undirected edge list `(E, 2)` of the implicit grid, built the
    way a caller without `connected_components(group=...)` /
    `label_adjacency` has to build it: vectorized sorted-key probes."""
    v = (vox - vox.min(axis=0)).astype(np.int64) + 1
    extent = v.max(axis=0) + 2

    def pack(a):
        return (a[:, 0] * extent[1] + a[:, 1]) * extent[2] + a[:, 2]

    keys = pack(v)
    order = np.argsort(keys).astype(np.int32)
    skeys = keys[order]

    max_manhattan = {6: 1, 18: 2, 26: 3}[connectivity]
    src, dst = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if manhattan == 0 or manhattan > max_manhattan:
                    continue
                nkeys = pack(v + (dx, dy, dz))
                pos = np.minimum(np.searchsorted(skeys, nkeys), len(skeys) - 1)
                found = skeys[pos] == nkeys
                src.append(np.nonzero(found)[0].astype(np.int64))
                dst.append(order[pos[found]].astype(np.int64))
    return np.stack([np.concatenate(src), np.concatenate(dst)], axis=1)


def wavefront_stages(vox: np.ndarray, backend: str, index_kind: str, step: float = 20.0) -> dict:
    """The wavefront-skeletonization graph pipeline: components -> geodesic
    field -> components of each level set ("rings") -> which rings touch.

    `backend="d3s"` runs it through the coordinate-native primitives;
    `backend="edges"` through the explicit edge list + SciPy/NumPy, which is
    what a caller has to do without them. Both must agree on the ring count.
    """
    import dijkstra3d_sparse as ds

    n = len(vox)
    extra = {}
    t0 = time.perf_counter()
    g = ds.Graph(vox, index_kind=index_kind)
    n_comp, comp = g.connected_components()
    seeds = [int(np.flatnonzero(comp == c)[0]) for c in range(n_comp)]
    dist, _ = g.dijkstra_field(seeds, cost_mode="geometric")
    lvl = np.floor(np.where(np.isfinite(dist), dist, 0.0) / step).astype(np.int64)
    extra["field_seconds"] = time.perf_counter() - t0
    extra["rss_after_field_bytes"] = peak_rss_bytes()

    t1 = time.perf_counter()
    if backend == "d3s":
        n_rings, rings = g.connected_components(group=lvl)
        extra["rings_seconds"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        sk_edges = g.label_adjacency(rings)
        extra["contract_seconds"] = time.perf_counter() - t2
    else:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components as scipy_cc

        e = edge_pairs(vox, 26)
        extra["edge_list_seconds"] = time.perf_counter() - t1
        extra["n_edges"] = len(e)
        t2 = time.perf_counter()
        same = e[lvl[e[:, 0]] == lvl[e[:, 1]]]
        n_rings, rings = scipy_cc(
            coo_matrix((np.ones(len(same), dtype=np.int8), (same[:, 0], same[:, 1])), shape=(n, n)),
            directed=False,
        )
        extra["rings_seconds"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        pairs = np.sort(rings[e].astype(np.int64), axis=1)
        sk_edges = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
        extra["contract_seconds"] = time.perf_counter() - t3

    extra["n_rings"] = int(n_rings)
    extra["n_ring_edges"] = int(len(sk_edges))
    return extra


def run_workload(task: str, n_target: int, index_kind: str) -> dict:
    vox = helix_tube(n_target)
    n = len(vox)
    bbox = vox.max(axis=0).astype(np.int64) - vox.min(axis=0) + 1
    rss_before = peak_rss_bytes()
    extra = {}

    t0 = time.perf_counter()
    if task == "dijkstra":
        import dijkstra3d_sparse as ds

        dist, pred = ds.dijkstra_field(vox, 0, connectivity=26, index_kind=index_kind)
        assert np.isfinite(dist).all(), "tube should be fully connected"
    elif task == "components":
        import dijkstra3d_sparse as ds

        n_comp, labels = ds.connected_components(vox, connectivity=26, index_kind=index_kind)
        assert n_comp == 1, f"tube should be one component, got {n_comp}"
    elif task == "scipy-csr":
        from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

        csr = build_csr(vox, connectivity=26)
        extra["csr_build_seconds"] = time.perf_counter() - t0
        extra["nnz"] = int(csr.nnz)
        t1 = time.perf_counter()
        dist = scipy_dijkstra(csr, directed=True, indices=0)
        extra["solve_seconds"] = time.perf_counter() - t1
        assert np.isfinite(dist).all(), "tube should be fully connected"
    elif task == "graft":
        # incremental grafting: repeatedly connect a random query voxel to a
        # growing anchor set via early-terminating search-to-a-set. The win
        # to record: per-query touched-voxel count collapses as anchors
        # densify, versus N for a full field every time.
        import dijkstra3d_sparse as ds

        rng = np.random.default_rng(0)
        anchor = np.zeros(n, dtype=bool)
        anchor[0] = True
        n_queries = 60
        touched = []
        for q in rng.integers(0, n, n_queries):
            dist, pred = ds.dijkstra_field(vox, int(q), stop_mask=anchor, stop_count=1)
            touched.append(int(np.isfinite(dist).sum()))
            finite_anchors = np.flatnonzero(anchor & np.isfinite(dist))
            hit = int(finite_anchors[np.argmin(dist[finite_anchors])])
            rows = [hit]  # graft the new spur onto the anchor set
            while pred[rows[-1]] >= 0:
                rows.append(int(pred[rows[-1]]))
            anchor[rows] = True
        extra["n_queries"] = n_queries
        extra["touched_first10_mean"] = float(np.mean(touched[:10]))
        extra["touched_last10_mean"] = float(np.mean(touched[-10:]))
    elif task == "reuse":
        # same voxels, K repeated queries: the free functions rebuild the
        # spatial index on every call, one Graph handle builds it once.
        # Anchors on every 64th voxel keep each early-terminating search
        # tiny, so the per-call cost is dominated by the index build — the
        # cost the handle amortizes (the TEASAR grafting regime).
        import dijkstra3d_sparse as ds

        rng = np.random.default_rng(2)
        anchor = np.zeros(n, dtype=bool)
        anchor[::64] = True
        ks = [2, 5, 50]
        queries = [int(q) for q in rng.integers(0, n, max(ks))]
        for k in ks:
            t1 = time.perf_counter()
            for q in queries[:k]:
                ds.shortest_path_to_set(vox, q, anchor, index_kind=index_kind)
            extra[f"free_k{k}_seconds"] = time.perf_counter() - t1
            t1 = time.perf_counter()
            g = ds.Graph(vox, index_kind=index_kind)  # handle build is included
            for q in queries[:k]:
                g.shortest_path_to_set(q, anchor)
            extra[f"graph_k{k}_seconds"] = time.perf_counter() - t1
        extra["ks"] = ks
    elif task in ("wavefront", "wavefront-edges"):
        extra.update(
            wavefront_stages(vox, "d3s" if task == "wavefront" else "edges", index_kind)
        )
    else:
        raise ValueError(task)
    elapsed = time.perf_counter() - t0

    return {
        "task": task,
        "index_kind": index_kind,
        "n": n,
        "bbox": bbox.tolist(),
        "bbox_cells": int(np.prod(bbox)),
        "seconds": elapsed,
        "peak_rss_bytes": peak_rss_bytes(),
        "rss_before_bytes": rss_before,
        **extra,
    }


def run_in_subprocess(task: str, n_target: int, index_kind: str) -> dict:
    out = subprocess.run(
        [sys.executable, __file__, "--worker", task, str(n_target), index_kind],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def fmt_bytes(b: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PiB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=3, metavar=("TASK", "N", "KIND"), help=argparse.SUPPRESS)
    ap.add_argument("--quick", action="store_true", help="smaller workload for smoke runs")
    ap.add_argument("--write-results", action="store_true", help="update benchmarks/RESULTS.md")
    args = ap.parse_args()

    if args.worker:
        task, n_target, kind = args.worker
        print(json.dumps(run_workload(task, int(n_target), kind)))
        return

    n_target = 200_000 if args.quick else 2_000_000
    runs = [("dijkstra", "sorted"), ("dijkstra", "hash"),
            ("components", "sorted"), ("components", "hash"),
            ("scipy-csr", "-"), ("graft", "-"), ("reuse", "hash"),
            ("wavefront", "hash"), ("wavefront-edges", "hash")]
    results = []
    for task, kind in runs:
        r = run_in_subprocess(task, n_target, kind)
        results.append(r)
        print(
            f"{task:10s} index={kind:6s} N={r['n']:>9,} "
            f"time={r['seconds']:7.2f}s peak_rss={fmt_bytes(r['peak_rss_bytes'])}"
        )

    r0 = results[0]
    scipy_r = next(r for r in results if r["task"] == "scipy-csr")
    graft_r = next(r for r in results if r["task"] == "graft")
    reuse_r = next(r for r in results if r["task"] == "reuse")
    wave_r = next(r for r in results if r["task"] == "wavefront")
    wave_e = next(r for r in results if r["task"] == "wavefront-edges")
    # the two wavefront backends must agree — this is a correctness check on
    # the coordinate-native primitives at benchmark scale, not just a timing
    assert (wave_r["n_rings"], wave_r["n_ring_edges"]) == (
        wave_e["n_rings"],
        wave_e["n_ring_edges"],
    ), f"wavefront backends disagree: {wave_r} vs {wave_e}"
    dense_bytes = r0["bbox_cells"] * DENSE_BYTES_PER_CELL
    lines = [
        "# Benchmark results",
        "",
        "Workload: thin helical tube (radius-3 cross-section) sweeping a cylinder,",
        f"N = {r0['n']:,} voxels in a bounding box of {r0['bbox'][0]:,} x "
        f"{r0['bbox'][1]:,} x {r0['bbox'][2]:,} = {r0['bbox_cells']:.2e} cells.",
        "Single-source Dijkstra, 26-connectivity, geometric costs (identical inputs",
        "for every row). Each run is a fresh subprocess; peak RSS includes the",
        "~60 MiB interpreter + NumPy/SciPy import baseline.",
        "",
        "| task | index | N | wall time | peak RSS | RSS bytes / voxel |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['task']} | {r['index_kind']} | {r['n']:,} | {r['seconds']:.2f} s "
            f"| {fmt_bytes(r['peak_rss_bytes'])} | {r['peak_rss_bytes'] / r['n']:.0f} |"
        )
    lines += [
        "",
        f"`scipy-csr` is `scipy.sparse.csgraph.dijkstra` on the explicit CSR graph of "
        f"the same grid ({scipy_r['nnz']:,} stored edges): "
        f"{scipy_r['csr_build_seconds']:.2f} s to build the CSR (vectorized NumPy) + "
        f"{scipy_r['solve_seconds']:.2f} s to solve. Materializing the edge list is "
        "exactly the work and memory the implicit-grid walk avoids.",
        "",
        f"`graft` is incremental grafting via early-terminating `stop_mask` queries: "
        f"{graft_r['n_queries']} random query voxels connected one after another to a "
        f"growing anchor set (each returned path joins the anchors). Mean voxels "
        f"touched per query: first 10 queries "
        f"{graft_r['touched_first10_mean']:,.0f} "
        f"({100 * graft_r['touched_first10_mean'] / graft_r['n']:.1f}% of N) → last 10 "
        f"queries {graft_r['touched_last10_mean']:,.0f} "
        f"({100 * graft_r['touched_last10_mean'] / graft_r['n']:.1f}% of N). A full "
        f"field per query would touch 100% of N every time; the whole "
        f"{graft_r['n_queries']}-query loop ran in {graft_r['seconds']:.2f} s.",
        "",
        "`reuse` is the reusable `Graph` handle (hash index): K early-terminating",
        "`shortest_path_to_set` queries over the same voxels (anchors on every 64th",
        "voxel, so each search is tiny and per-call cost is dominated by the spatial-",
        "index build). The free functions rebuild that index on every call; one",
        "`Graph` builds it once (handle construction is included in its time):",
        "",
        "| K queries | free functions | one `Graph` + K calls | speedup |",
        "|---|---|---|---|",
    ]
    for k in reuse_r["ks"]:
        free_s = reuse_r[f"free_k{k}_seconds"]
        graph_s = reuse_r[f"graph_k{k}_seconds"]
        lines.append(f"| {k} | {free_s:.2f} s | {graph_s:.2f} s | {free_s / graph_s:.1f}x |")
    lines += [
        "",
        "`wavefront` is the graph half of wavefront skeletonization over one",
        "`Graph`: components → geodesic field → components of each geodesic level",
        "set (`connected_components(group=level)`) → which of those rings touch",
        "(`label_adjacency`). `wavefront-edges` is the same pipeline for a caller",
        "without those two primitives: the explicit edge list, SciPy connected",
        "components on the same-level sub-graph, and a NumPy `unique` to contract.",
        "Both agree on the result "
        f"({wave_r['n_rings']:,} rings, {wave_r['n_ring_edges']:,} ring edges); the",
        "field stage is shared and identical.",
        "",
        "| stage | coordinate-native | edge list |",
        "|---|---|---|",
        f"| components + geodesic field | {wave_r['field_seconds']:.2f} s "
        f"| {wave_e['field_seconds']:.2f} s |",
        f"| edge list ({wave_e['n_edges']:,} rows) | — | {wave_e['edge_list_seconds']:.2f} s |",
        f"| rings (components per level) | {wave_r['rings_seconds']:.2f} s "
        f"| {wave_e['rings_seconds']:.2f} s |",
        f"| contract onto rings | {wave_r['contract_seconds']:.2f} s "
        f"| {wave_e['contract_seconds']:.2f} s |",
        f"| **total** | **{wave_r['seconds']:.2f} s** | **{wave_e['seconds']:.2f} s** |",
        f"| **peak RSS** | **{fmt_bytes(wave_r['peak_rss_bytes'])}** "
        f"| **{fmt_bytes(wave_e['peak_rss_bytes'])}** |",
        "",
        f"The edge list is the entire difference: "
        f"{wave_e['n_edges']:,} adjacencies materialized to yield "
        f"{wave_r['n_ring_edges']:,} distinct ring pairs. `label_adjacency`",
        "deduplicates as it probes, so that intermediate never exists.",
        "",
        f"Dense baseline for the same bounding box (dist f64 + pred i64 + visited u8 = "
        f"{DENSE_BYTES_PER_CELL} B/cell): **{fmt_bytes(dense_bytes)}** — "
        f"{dense_bytes / max(r['peak_rss_bytes'] for r in results):,.0f}x the "
        "largest peak RSS measured here.",
        "",
        "Memory gate: peak RSS must stay O(N), far below the dense bbox baseline.",
    ]
    report = "\n".join(lines)
    print("\n" + report)

    # ---- the memory gate itself (our runs only; the CSR baseline is *expected*
    # to blow past it — that is the point of the comparison) ----
    ours = [r for r in results if r["task"] not in ("scipy-csr", "wavefront-edges")]
    worst = max(r["peak_rss_bytes"] for r in ours)
    assert worst < 0.01 * dense_bytes, (
        f"peak RSS {fmt_bytes(worst)} is not << dense baseline {fmt_bytes(dense_bytes)}"
    )
    assert worst < 1000 * r0["n"], (
        f"peak RSS {fmt_bytes(worst)} exceeds 1000 bytes/voxel — memory no longer O(N)?"
    )
    print("\nmemory gate PASSED: peak RSS scales with N, not with the bounding box")

    if args.write_results:
        (HERE / "RESULTS.md").write_text(report + "\n")
        print(f"wrote {HERE / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
