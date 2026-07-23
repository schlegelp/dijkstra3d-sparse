//! Python bindings for the sparse-voxel Dijkstra core.
//!
//! Thin glue only: validation with friendly messages lives in the Python
//! wrapper (`dijkstra3d_sparse/__init__.py`); the checks here are the
//! defense-in-depth needed for memory safety.
//!
//! Two entry styles share the same per-call cores (`run_field`,
//! `probe_rows`, `components::connected_components`): the stateless
//! `#[pyfunction]`s build a throwaway `SpatialIndex` per call, while the
//! `Graph` pyclass builds it once at construction and reuses it.

mod components;
mod dedup;
mod dijkstra;
mod index;
mod offsets;

use numpy::{
    Element, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
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

/// `(n_labels, labels, reps)` output of `factorize`; `reps` is `None` unless
/// `return_index` was requested.
type FactorizeArrays<'py> = (
    usize,
    Bound<'py, PyArray1<i64>>,
    Option<Bound<'py, PyArray1<i64>>>,
);

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

fn check_connectivity(connectivity: u8) -> PyResult<()> {
    if !matches!(connectivity, 6 | 18 | 26) {
        return Err(err(format!(
            "connectivity must be 6, 18 or 26, got {connectivity}"
        )));
    }
    Ok(())
}

/// Borrow an optional per-voxel array as a length-checked slice.
fn per_voxel_slice<'a, T: Element>(
    name: &str,
    arr: Option<&'a PyReadonlyArray1<'_, T>>,
    n: usize,
) -> PyResult<Option<&'a [T]>> {
    match arr {
        Some(arr) => {
            let s = arr
                .as_slice()
                .map_err(|_| err(format!("{name} must be C-contiguous")))?;
            check_len(name, s.len(), n)?;
            Ok(Some(s))
        }
        None => Ok(None),
    }
}

/// Everything per-call about a field run, validated and borrowed. The
/// reusable state (`coords` + `SpatialIndex`) deliberately lives outside:
/// it comes either from a throwaway build (free function) or from a
/// `Graph` handle.
struct FieldCall<'a> {
    sources: &'a [i64],
    node_cost: Option<&'a [f32]>,
    connectivity: u8,
    aniso: [f64; 3],
    mode: CostMode,
    free_mask: Option<&'a [bool]>,
    free_eps: f64,
    min_only: bool,
    stop_mask: Option<&'a [bool]>,
    stop_count: usize,
}

/// Validate the per-call arguments shared by the free `dijkstra_field` and
/// `Graph::dijkstra_field`.
#[allow(clippy::too_many_arguments)]
fn validate_field_call<'a>(
    n: usize,
    sources: &'a PyReadonlyArray1<'_, i64>,
    node_cost: Option<&'a PyReadonlyArray1<'_, f32>>,
    connectivity: u8,
    anisotropy: (f64, f64, f64),
    cost_mode: &str,
    free_mask: Option<&'a PyReadonlyArray1<'_, bool>>,
    free_eps: f64,
    min_only: bool,
    stop_mask: Option<&'a PyReadonlyArray1<'_, bool>>,
    stop_count: usize,
) -> PyResult<FieldCall<'a>> {
    let sources = sources
        .as_slice()
        .map_err(|_| err("sources must be C-contiguous".into()))?;
    check_sources(sources, n)?;
    let node_cost = per_voxel_slice("node_cost", node_cost, n)?;
    let free_mask = per_voxel_slice("free_mask", free_mask, n)?;
    let stop_mask = per_voxel_slice("stop_mask", stop_mask, n)?;

    let mode = CostMode::parse(cost_mode).map_err(err)?;
    if mode != CostMode::Geometric && node_cost.is_none() {
        return Err(err(format!("cost_mode '{cost_mode}' requires node_cost")));
    }
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
    check_connectivity(connectivity)?;

    Ok(FieldCall {
        sources,
        node_cost,
        connectivity,
        aniso,
        mode,
        free_mask,
        free_eps,
        min_only,
        stop_mask,
        stop_count,
    })
}

/// Per-call core of `dijkstra_field`: identical code runs whether `index`
/// was just built (stateless call) or is reused from a `Graph` handle.
fn run_field(coords: &[i32], n: usize, index: &SpatialIndex, call: &FieldCall<'_>) -> FieldVecs {
    let offsets = build_offsets(call.connectivity, call.aniso);
    let problem = Problem {
        coords,
        n,
        index,
        offsets: &offsets,
        node_cost: call.node_cost,
        cost_mode: call.mode,
        free_mask: call.free_mask,
        free_eps: call.free_eps,
    };
    if call.min_only {
        let mut dist = vec![0.0f64; n];
        let mut pred = vec![0i64; n];
        let hit = dijkstra::dijkstra(
            &problem,
            call.sources,
            call.stop_mask,
            call.stop_count,
            &mut dist,
            &mut pred,
        );
        (dist, pred, vec![hit])
    } else {
        // One independent run per source, reusing the spatial index.
        let s_count = call.sources.len();
        let mut dist = vec![0.0f64; n * s_count];
        let mut pred = vec![0i64; n * s_count];
        let mut hits = Vec::with_capacity(s_count);
        for (i, &s) in call.sources.iter().enumerate() {
            hits.push(dijkstra::dijkstra(
                &problem,
                &[s],
                call.stop_mask,
                call.stop_count,
                &mut dist[i * n..(i + 1) * n],
                &mut pred[i * n..(i + 1) * n],
            ));
        }
        (dist, pred, hits)
    }
}

/// Per-call core of `index_of`: row of each query coordinate, -1 if absent.
fn probe_rows(index: &SpatialIndex, qcoords: &[i32], qn: usize) -> Vec<i64> {
    (0..qn)
        .map(|i| {
            let b = i * 3;
            index
                .get(key_of(qcoords[b], qcoords[b + 1], qcoords[b + 2]))
                .map_or(-1, |row| row as i64)
        })
        .collect()
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
    let kind = IndexKind::parse(index_kind).map_err(err)?;
    let call = validate_field_call(
        n,
        &sources,
        node_cost.as_ref(),
        connectivity,
        anisotropy,
        cost_mode,
        free_mask.as_ref(),
        free_eps,
        min_only,
        stop_mask.as_ref(),
        stop_count,
    )?;

    let (dist, pred, hits) = py
        .allow_threads(|| -> Result<FieldVecs, String> {
            let sindex = SpatialIndex::build(coords, n, kind)?;
            Ok(run_field(coords, n, &sindex, &call))
        })
        .map_err(err)?;

    Ok((
        dist.into_pyarray(py),
        pred.into_pyarray(py),
        hits.into_pyarray(py),
    ))
}

/// Connected components over the implicit sparse grid (union-find).
/// With `group`, only neighbours whose group value matches are unioned.
#[pyfunction]
#[pyo3(signature = (voxels, connectivity, group, index_kind))]
fn connected_components<'py>(
    py: Python<'py>,
    voxels: PyReadonlyArray2<'py, i32>,
    connectivity: u8,
    group: Option<PyReadonlyArray1<'py, i64>>,
    index_kind: &str,
) -> PyResult<(usize, Bound<'py, PyArray1<i32>>)> {
    let (coords, n) = coords_slice(&voxels)?;
    let kind = IndexKind::parse(index_kind).map_err(err)?;
    check_connectivity(connectivity)?;
    // Bind the borrow outside the closure: it has to outlive `allow_threads`.
    let group = per_voxel_slice("group", group.as_ref(), n)?;

    let (n_components, labels) = py
        .allow_threads(|| -> Result<(usize, Vec<i32>), String> {
            let sindex = SpatialIndex::build(coords, n, kind)?;
            let offsets = build_offsets(connectivity, [1.0; 3]);
            Ok(components::connected_components(
                coords, n, &sindex, &offsets, group,
            ))
        })
        .map_err(err)?;

    Ok((n_components, labels.into_pyarray(py)))
}

/// Pairs of distinct labels that touch, as a flat `[lo, hi, lo, hi, ...]`
/// vector (the Python wrapper reshapes it to `(K, 2)`).
#[pyfunction]
#[pyo3(signature = (voxels, labels, connectivity, index_kind))]
fn label_adjacency<'py>(
    py: Python<'py>,
    voxels: PyReadonlyArray2<'py, i32>,
    labels: PyReadonlyArray1<'py, i64>,
    connectivity: u8,
    index_kind: &str,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (coords, n) = coords_slice(&voxels)?;
    let kind = IndexKind::parse(index_kind).map_err(err)?;
    check_connectivity(connectivity)?;
    let labels = per_voxel_slice("labels", Some(&labels), n)?.unwrap();

    let out = py
        .allow_threads(|| -> Result<Vec<i64>, String> {
            let sindex = SpatialIndex::build(coords, n, kind)?;
            let offsets = build_offsets(connectivity, [1.0; 3]);
            Ok(flatten_pairs(components::label_adjacency(
                coords, n, &sindex, &offsets, labels,
            )))
        })
        .map_err(err)?;

    Ok(out.into_pyarray(py))
}

/// Per-voxel 6-bit mask of absent face-neighbours (surface extraction).
///
/// Builds the spatial index, probes the six face offsets in one pass, and
/// frees the index before returning — the memory point of the primitive: no
/// persistent handle, no six `(N, 3)` neighbour temporaries, one `(N,)`
/// `uint8` out. Bit order is `+x, -x, +y, -y, +z, -z` (see
/// `components::exposed_faces`).
#[pyfunction]
#[pyo3(signature = (voxels, index_kind))]
fn exposed_faces<'py>(
    py: Python<'py>,
    voxels: PyReadonlyArray2<'py, i32>,
    index_kind: &str,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let (coords, n) = coords_slice(&voxels)?;
    let kind = IndexKind::parse(index_kind).map_err(err)?;

    let mask = py
        .allow_threads(|| -> Result<Vec<u8>, String> {
            let sindex = SpatialIndex::build(coords, n, kind)?;
            Ok(components::exposed_faces(coords, n, &sindex))
        })
        .map_err(err)?;

    Ok(mask.into_pyarray(py))
}

/// Dense per-row labels for coordinates by exact equality (duplicates
/// collapse) — the sparse `np.unique(..., return_inverse=True)`, done as one
/// hash pass instead of a sort. Returns `(n_labels, labels, reps)`; `reps` is
/// `None` unless `return_index`. Unlike `Graph`/`index_of`/`connected_components`,
/// this *accepts* duplicated input — that is its whole reason to exist.
#[pyfunction]
#[pyo3(signature = (voxels, return_index, index_kind))]
fn factorize<'py>(
    py: Python<'py>,
    voxels: PyReadonlyArray2<'py, i32>,
    return_index: bool,
    index_kind: &str,
) -> PyResult<FactorizeArrays<'py>> {
    let (coords, n) = coords_slice(&voxels)?;
    // Validated for signature parity with the other free functions; factorize
    // is a single hash pass, so the backend does not change its result.
    IndexKind::parse(index_kind).map_err(err)?;

    let (n_labels, labels, reps) = py.allow_threads(|| dedup::factorize(coords, n, return_index));

    Ok((
        n_labels,
        labels.into_pyarray(py),
        reps.map(|r| r.into_pyarray(py)),
    ))
}

/// `[(lo, hi), ...]` -> flat `[lo, hi, lo, hi, ...]` for the NumPy boundary.
fn flatten_pairs(pairs: Vec<(i64, i64)>) -> Vec<i64> {
    let mut out = Vec::with_capacity(pairs.len() * 2);
    for (lo, hi) in pairs {
        out.push(lo);
        out.push(hi);
    }
    out
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
            Ok(probe_rows(&sindex, qcoords, qn))
        })
        .map_err(err)?;

    Ok(out.into_pyarray(py))
}

/// Reusable spatial-index handle over a fixed voxel set.
///
/// Owns a snapshot of the coordinates and the `SpatialIndex` built from
/// them, so repeated queries skip the per-call O(N) index build. Read-only
/// after construction: methods only borrow `&self` and write fresh output
/// buffers, so one handle can serve any number of queries.
#[pyclass(frozen)]
struct Graph {
    /// Owned row-major `(N, 3)` copy — decouples the handle from the
    /// caller's array (its lifetime and any later mutation).
    coords: Vec<i32>,
    n: usize,
    index: SpatialIndex,
    kind: IndexKind,
}

#[pymethods]
impl Graph {
    #[new]
    #[pyo3(signature = (voxels, index_kind="hash"))]
    fn new(py: Python<'_>, voxels: PyReadonlyArray2<'_, i32>, index_kind: &str) -> PyResult<Self> {
        let (coords, n) = coords_slice(&voxels)?;
        let kind = IndexKind::parse(index_kind).map_err(err)?;
        let coords = coords.to_vec();
        // Duplicate-coordinate rejection happens here, once per handle.
        let index = py
            .allow_threads(|| SpatialIndex::build(&coords, n, kind))
            .map_err(err)?;
        Ok(Graph {
            coords,
            n,
            index,
            kind,
        })
    }

    /// Number of voxels the handle was built over.
    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    /// Spatial-index backend fixed at construction: "sorted" or "hash".
    #[getter]
    fn index_kind(&self) -> &'static str {
        self.kind.as_str()
    }

    /// `dijkstra_field` minus `voxels`/`index_kind`: same per-call core,
    /// reusing the handle's index instead of rebuilding it.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (sources, node_cost, connectivity, anisotropy, cost_mode, free_mask, free_eps, min_only, stop_mask, stop_count))]
    fn dijkstra_field<'py>(
        &self,
        py: Python<'py>,
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
    ) -> PyResult<FieldArrays<'py>> {
        let call = validate_field_call(
            self.n,
            &sources,
            node_cost.as_ref(),
            connectivity,
            anisotropy,
            cost_mode,
            free_mask.as_ref(),
            free_eps,
            min_only,
            stop_mask.as_ref(),
            stop_count,
        )?;
        let (dist, pred, hits) =
            py.allow_threads(|| run_field(&self.coords, self.n, &self.index, &call));
        Ok((
            dist.into_pyarray(py),
            pred.into_pyarray(py),
            hits.into_pyarray(py),
        ))
    }

    /// `connected_components` minus `voxels`/`index_kind`.
    #[pyo3(signature = (connectivity, group=None))]
    fn connected_components<'py>(
        &self,
        py: Python<'py>,
        connectivity: u8,
        group: Option<PyReadonlyArray1<'py, i64>>,
    ) -> PyResult<(usize, Bound<'py, PyArray1<i32>>)> {
        check_connectivity(connectivity)?;
        let group = per_voxel_slice("group", group.as_ref(), self.n)?;
        let (n_components, labels) = py.allow_threads(|| {
            let offsets = build_offsets(connectivity, [1.0; 3]);
            components::connected_components(&self.coords, self.n, &self.index, &offsets, group)
        });
        Ok((n_components, labels.into_pyarray(py)))
    }

    /// `label_adjacency` minus `voxels`/`index_kind`.
    fn label_adjacency<'py>(
        &self,
        py: Python<'py>,
        labels: PyReadonlyArray1<'py, i64>,
        connectivity: u8,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        check_connectivity(connectivity)?;
        let labels = per_voxel_slice("labels", Some(&labels), self.n)?.unwrap();
        let out = py.allow_threads(|| {
            let offsets = build_offsets(connectivity, [1.0; 3]);
            flatten_pairs(components::label_adjacency(
                &self.coords,
                self.n,
                &self.index,
                &offsets,
                labels,
            ))
        });
        Ok(out.into_pyarray(py))
    }

    /// `exposed_faces` minus `voxels`/`index_kind`. Reuses the handle's
    /// index, so — unlike the free function — it does *not* free it after
    /// the pass; the free function is the one that delivers the memory win.
    fn exposed_faces<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let mask =
            py.allow_threads(|| components::exposed_faces(&self.coords, self.n, &self.index));
        Ok(mask.into_pyarray(py))
    }

    /// `index_of` minus `voxels`/`index_kind`.
    fn index_of<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, i32>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (qcoords, qn) = coords_slice(&queries)?;
        let out = py.allow_threads(|| probe_rows(&self.index, qcoords, qn));
        Ok(out.into_pyarray(py))
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dijkstra_field, m)?)?;
    m.add_function(wrap_pyfunction!(connected_components, m)?)?;
    m.add_function(wrap_pyfunction!(label_adjacency, m)?)?;
    m.add_function(wrap_pyfunction!(exposed_faces, m)?)?;
    m.add_function(wrap_pyfunction!(factorize, m)?)?;
    m.add_function(wrap_pyfunction!(index_of, m)?)?;
    m.add_class::<Graph>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shared core must give identical results on a reused index and a
    /// freshly built one — the `Graph` handle only changes *where* the
    /// index is built, never the output.
    #[test]
    fn run_field_on_reused_index_matches_fresh_build() {
        let coords: Vec<i32> = (0..6).flat_map(|x| [x, 0, 0]).collect();
        let n = 6;
        let geo = FieldCall {
            sources: &[0],
            node_cost: None,
            connectivity: 26,
            aniso: [1.0, 1.0, 1.0],
            mode: CostMode::Geometric,
            free_mask: None,
            free_eps: 1e-6,
            min_only: true,
            stop_mask: None,
            stop_count: 1,
        };
        let node_cost: Vec<f32> = (0..n).map(|i| 0.5 + i as f32).collect();
        let additive = FieldCall {
            sources: &[2, 5],
            node_cost: Some(&node_cost),
            connectivity: 6,
            aniso: [1.0, 2.0, 3.0],
            mode: CostMode::Additive,
            free_mask: None,
            free_eps: 1e-6,
            min_only: false,
            stop_mask: None,
            stop_count: 0,
        };

        let reused = SpatialIndex::build(&coords, n, IndexKind::Hash).unwrap();
        // two different queries back-to-back on the same index...
        let first = run_field(&coords, n, &reused, &geo);
        let second = run_field(&coords, n, &reused, &additive);
        // ...and the first repeated, to prove no state leaks between calls
        let repeat = run_field(&coords, n, &reused, &geo);
        assert_eq!(first, repeat);

        for kind in [IndexKind::Sorted, IndexKind::Hash] {
            let fresh = SpatialIndex::build(&coords, n, kind).unwrap();
            assert_eq!(first, run_field(&coords, n, &fresh, &geo));
            assert_eq!(second, run_field(&coords, n, &fresh, &additive));
        }
    }
}
