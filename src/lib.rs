//! Python bindings for the sparse-voxel Dijkstra core.
//!
//! Thin glue only: validation with friendly messages lives in the Python
//! wrapper (`dijkstra3d_sparse/__init__.py`); the checks here are the
//! defense-in-depth needed for memory safety.

mod components;
mod dijkstra;
mod index;
mod offsets;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::dijkstra::{CostMode, Problem};
use crate::index::{key_of, IndexKind, SpatialIndex};
use crate::offsets::build_offsets;

fn err(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

/// `(dist, pred, hits)` output arrays of `dijkstra_field`.
type FieldArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

/// `(dist, pred, hits)` as plain vectors, produced inside `allow_threads`.
type FieldVecs = (Vec<f64>, Vec<i64>, Vec<i64>);

/// Extract `(coords_slice, n)` from an `(N, 3)` C-contiguous array.
fn coords_slice<'a>(voxels: &'a PyReadonlyArray2<'a, i32>) -> PyResult<(&'a [i32], usize)> {
    let shape = voxels.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(err(format!("voxels must have shape (N, 3), got {shape:?}")));
    }
    let n = shape[0];
    if n > i32::MAX as usize {
        return Err(err(format!("at most 2^31 - 1 voxels supported, got {n}")));
    }
    let slice = voxels
        .as_slice()
        .map_err(|_| err("voxels must be C-contiguous".into()))?;
    Ok((slice, n))
}

fn check_sources(sources: &[i64], n: usize) -> PyResult<()> {
    for &s in sources {
        if s < 0 || s as usize >= n {
            return Err(err(format!("source index {s} out of range for {n} voxels")));
        }
    }
    Ok(())
}

fn check_len(name: &str, len: usize, n: usize) -> PyResult<()> {
    if len != n {
        return Err(err(format!(
            "{name} must have length N = {n} (one entry per voxel), got {len}"
        )));
    }
    Ok(())
}

/// Multi-source Dijkstra over the implicit sparse grid.
///
/// Returns flat `(dist, pred, hits)` vectors: `dist`/`pred` of length `N`
/// when `min_only`, else `S * N` (row-major, one Dijkstra run per source;
/// the Python wrapper reshapes to `(S, N)`). `hits` holds the first-settled
/// `stop_mask` row per run (`-1` if none): length 1 when `min_only`, else
/// `S`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (voxels, sources, node_cost, connectivity, anisotropy, cost_mode, free_mask, free_eps, min_only, stop_mask, stop_count, index_kind))]
fn dijkstra_field<'py>(
    py: Python<'py>,
    voxels: PyReadonlyArray2<'py, i32>,
    sources: PyReadonlyArray1<'py, i64>,
    node_cost: Option<PyReadonlyArray1<'py, f32>>,
    connectivity: u8,
    anisotropy: (f64, f64, f64),
    cost_mode: &str,
    free_mask: Option<PyReadonlyArray1<'py, bool>>,
    free_eps: f64,
    min_only: bool,
    stop_mask: Option<PyReadonlyArray1<'py, bool>>,
    stop_count: usize,
    index_kind: &str,
) -> PyResult<FieldArrays<'py>> {
    let (coords, n) = coords_slice(&voxels)?;
    let sources = sources
        .as_slice()
        .map_err(|_| err("sources must be C-contiguous".into()))?;
    check_sources(sources, n)?;

    let node_cost_slice = match &node_cost {
        Some(arr) => {
            let s = arr
                .as_slice()
                .map_err(|_| err("node_cost must be C-contiguous".into()))?;
            check_len("node_cost", s.len(), n)?;
            Some(s)
        }
        None => None,
    };
    let free_mask_slice = match &free_mask {
        Some(arr) => {
            let s = arr
                .as_slice()
                .map_err(|_| err("free_mask must be C-contiguous".into()))?;
            check_len("free_mask", s.len(), n)?;
            Some(s)
        }
        None => None,
    };
    let stop_mask_slice = match &stop_mask {
        Some(arr) => {
            let s = arr
                .as_slice()
                .map_err(|_| err("stop_mask must be C-contiguous".into()))?;
            check_len("stop_mask", s.len(), n)?;
            Some(s)
        }
        None => None,
    };

    let mode = CostMode::parse(cost_mode).map_err(err)?;
    if mode != CostMode::Geometric && node_cost_slice.is_none() {
        return Err(err(format!("cost_mode '{cost_mode}' requires node_cost")));
    }
    let kind = IndexKind::parse(index_kind).map_err(err)?;
    if !(free_eps.is_finite() && free_eps > 0.0) {
        return Err(err(format!(
            "free_eps must be finite and > 0, got {free_eps}"
        )));
    }
    let aniso = [anisotropy.0, anisotropy.1, anisotropy.2];
    if aniso.iter().any(|w| !(w.is_finite() && *w > 0.0)) {
        return Err(err(format!(
            "anisotropy must be three finite positive values, got {anisotropy:?}"
        )));
    }
    if !matches!(connectivity, 6 | 18 | 26) {
        return Err(err(format!(
            "connectivity must be 6, 18 or 26, got {connectivity}"
        )));
    }

    let (dist, pred, hits) = py
        .allow_threads(|| -> Result<FieldVecs, String> {
            let sindex = SpatialIndex::build(coords, n, kind)?;
            let offsets = build_offsets(connectivity, aniso);
            let problem = Problem {
                coords,
                n,
                index: &sindex,
                offsets: &offsets,
                node_cost: node_cost_slice,
                cost_mode: mode,
                free_mask: free_mask_slice,
                free_eps,
            };
            if min_only {
                let mut dist = vec![0.0f64; n];
                let mut pred = vec![0i64; n];
                let hit = dijkstra::dijkstra(
                    &problem,
                    sources,
                    stop_mask_slice,
                    stop_count,
                    &mut dist,
                    &mut pred,
                );
                Ok((dist, pred, vec![hit]))
            } else {
                // One independent run per source, reusing the spatial index.
                let s_count = sources.len();
                let mut dist = vec![0.0f64; n * s_count];
                let mut pred = vec![0i64; n * s_count];
                let mut hits = Vec::with_capacity(s_count);
                for (i, &s) in sources.iter().enumerate() {
                    hits.push(dijkstra::dijkstra(
                        &problem,
                        &[s],
                        stop_mask_slice,
                        stop_count,
                        &mut dist[i * n..(i + 1) * n],
                        &mut pred[i * n..(i + 1) * n],
                    ));
                }
                Ok((dist, pred, hits))
            }
        })
        .map_err(err)?;

    Ok((
        dist.into_pyarray(py),
        pred.into_pyarray(py),
        hits.into_pyarray(py),
    ))
}

/// Connected components over the implicit sparse grid (union-find).
#[pyfunction]
#[pyo3(signature = (voxels, connectivity, index_kind))]
fn connected_components<'py>(
    py: Python<'py>,
    voxels: PyReadonlyArray2<'py, i32>,
    connectivity: u8,
    index_kind: &str,
) -> PyResult<(usize, Bound<'py, PyArray1<i32>>)> {
    let (coords, n) = coords_slice(&voxels)?;
    let kind = IndexKind::parse(index_kind).map_err(err)?;
    if !matches!(connectivity, 6 | 18 | 26) {
        return Err(err(format!(
            "connectivity must be 6, 18 or 26, got {connectivity}"
        )));
    }

    let (n_components, labels) = py
        .allow_threads(|| -> Result<(usize, Vec<i32>), String> {
            let sindex = SpatialIndex::build(coords, n, kind)?;
            let offsets = build_offsets(connectivity, [1.0; 3]);
            Ok(components::connected_components(
                coords, n, &sindex, &offsets,
            ))
        })
        .map_err(err)?;

    Ok((n_components, labels.into_pyarray(py)))
}

/// Row index in `voxels` for each coordinate in `queries`; -1 if absent.
#[pyfunction]
#[pyo3(signature = (voxels, queries, index_kind))]
fn index_of<'py>(
    py: Python<'py>,
    voxels: PyReadonlyArray2<'py, i32>,
    queries: PyReadonlyArray2<'py, i32>,
    index_kind: &str,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (coords, n) = coords_slice(&voxels)?;
    let (qcoords, qn) = coords_slice(&queries)?;
    let kind = IndexKind::parse(index_kind).map_err(err)?;

    let out = py
        .allow_threads(|| -> Result<Vec<i64>, String> {
            let sindex = SpatialIndex::build(coords, n, kind)?;
            Ok((0..qn)
                .map(|i| {
                    let b = i * 3;
                    sindex
                        .get(key_of(qcoords[b], qcoords[b + 1], qcoords[b + 2]))
                        .map_or(-1, |row| row as i64)
                })
                .collect())
        })
        .map_err(err)?;

    Ok(out.into_pyarray(py))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dijkstra_field, m)?)?;
    m.add_function(wrap_pyfunction!(connected_components, m)?)?;
    m.add_function(wrap_pyfunction!(index_of, m)?)?;
    Ok(())
}
