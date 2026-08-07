"""Type stubs for the Rust extension.

Arguments are positional; the Python wrapper
(``dijkstra3d_sparse/__init__.py``) supplies the defaults, except for the few
the pyo3 signature declares itself (marked ``= ...`` below).
"""

from typing import Optional, Tuple

import numpy as np

def dijkstra_field(
    voxels: np.ndarray,
    sources: np.ndarray,
    node_cost: Optional[np.ndarray],
    connectivity: int,
    anisotropy: Tuple[float, float, float],
    cost_mode: str,
    free_mask: Optional[np.ndarray],
    free_eps: float,
    min_only: bool,
    stop_mask: Optional[np.ndarray],
    stop_count: int,
    index_kind: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def connected_components(
    voxels: np.ndarray,
    connectivity: int,
    group: Optional[np.ndarray],
    index_kind: str,
) -> Tuple[int, np.ndarray]: ...
def label_adjacency(
    voxels: np.ndarray,
    labels: np.ndarray,
    connectivity: int,
    index_kind: str,
) -> np.ndarray: ...
def exposed_faces(
    voxels: np.ndarray,
    index_kind: str,
) -> np.ndarray: ...
def factorize(
    voxels: np.ndarray,
    return_index: bool,
    index_kind: str,
) -> Tuple[int, np.ndarray, Optional[np.ndarray]]: ...
def ray_exits(
    voxels: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    max_dist: np.ndarray,
    max_crossings: int,
    index_kind: str,
) -> Tuple[np.ndarray, np.ndarray]: ...
def index_of(
    voxels: np.ndarray,
    queries: np.ndarray,
    index_kind: str,
) -> np.ndarray: ...

class Graph:
    """Reusable spatial-index handle; methods mirror the free functions
    minus ``voxels``/``index_kind``, which are fixed at construction."""

    def __init__(self, voxels: np.ndarray, index_kind: str = ...) -> None: ...
    @property
    def n(self) -> int: ...
    @property
    def index_kind(self) -> str: ...
    def dijkstra_field(
        self,
        sources: np.ndarray,
        node_cost: Optional[np.ndarray],
        connectivity: int,
        anisotropy: Tuple[float, float, float],
        cost_mode: str,
        free_mask: Optional[np.ndarray],
        free_eps: float,
        min_only: bool,
        stop_mask: Optional[np.ndarray],
        stop_count: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def connected_components(
        self,
        connectivity: int,
        group: Optional[np.ndarray] = ...,
    ) -> Tuple[int, np.ndarray]: ...
    def label_adjacency(
        self,
        labels: np.ndarray,
        connectivity: int,
    ) -> np.ndarray: ...
    def exposed_faces(self) -> np.ndarray: ...
    def ray_exits(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        max_dist: np.ndarray,
        max_crossings: int,
    ) -> Tuple[np.ndarray, np.ndarray]: ...
    def index_of(self, queries: np.ndarray) -> np.ndarray: ...
