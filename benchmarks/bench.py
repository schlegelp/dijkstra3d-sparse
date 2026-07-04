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
            ("scipy-csr", "-")]
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
    ours = [r for r in results if r["task"] != "scipy-csr"]
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
