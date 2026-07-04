"""Shared test helpers: reference offset/cost model and explicit CSR graphs.

The tests validate the library against `scipy.sparse.csgraph` on the
*explicit* graph equivalent of the implicit grid — built independently here
in pure Python/NumPy.
"""

import numpy as np
from scipy.sparse import csr_matrix


def offsets(connectivity, anisotropy=(1.0, 1.0, 1.0)):
    """[(offset (3,), step_length)] for the given connectivity."""
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if manhattan == 0 or manhattan > {6: 1, 18: 2, 26: 3}[connectivity]:
                    continue
                length = np.sqrt(
                    (dx * anisotropy[0]) ** 2
                    + (dy * anisotropy[1]) ** 2
                    + (dz * anisotropy[2]) ** 2
                )
                out.append((np.array([dx, dy, dz]), length))
    return out


def edge_cost(j, step_length, cost_mode, node_cost, free_mask, free_eps):
    """Directed edge cost into voxel j — mirrors the spec's cost table."""
    if free_mask is not None and free_mask[j]:
        return free_eps
    if cost_mode == "geometric" or node_cost is None:
        return step_length
    # match Rust: node_cost is f32, widened to f64 per edge
    c = float(np.float32(node_cost[j]))
    if cost_mode == "vertex":
        return c * step_length
    if cost_mode == "additive":
        return step_length + c
    raise ValueError(cost_mode)


def build_csr(
    voxels,
    connectivity=26,
    anisotropy=(1.0, 1.0, 1.0),
    cost_mode="geometric",
    node_cost=None,
    free_mask=None,
    free_eps=1e-6,
):
    """Explicit directed CSR graph equivalent of the implicit sparse grid."""
    lookup = {tuple(v): i for i, v in enumerate(np.asarray(voxels))}
    rows, cols, weights = [], [], []
    for i, v in enumerate(np.asarray(voxels)):
        for off, length in offsets(connectivity, anisotropy):
            j = lookup.get(tuple(v + off))
            if j is None:
                continue
            rows.append(i)
            cols.append(j)
            weights.append(edge_cost(j, length, cost_mode, node_cost, free_mask, free_eps))
    n = len(voxels)
    return csr_matrix((weights, (rows, cols)), shape=(n, n))


def random_cloud(rng, n, box, offset=(0, 0, 0)):
    """~n unique random voxels in the given box, shuffled (unsorted input)."""
    pts = rng.integers(0, box, size=(n, 3)) + np.asarray(offset)
    pts = np.unique(pts, axis=0)
    rng.shuffle(pts)
    return pts.astype(np.int32)


def path_cost(voxels, rows, connectivity, anisotropy, cost_mode, node_cost, free_mask, free_eps):
    """Total cost of a path given as row indices."""
    total = 0.0
    for a, b in zip(rows[:-1], rows[1:]):
        d = np.asarray(voxels[b]) - np.asarray(voxels[a])
        length = np.sqrt(
            (d[0] * anisotropy[0]) ** 2 + (d[1] * anisotropy[1]) ** 2 + (d[2] * anisotropy[2]) ** 2
        )
        total += edge_cost(b, length, cost_mode, node_cost, free_mask, free_eps)
    return total
