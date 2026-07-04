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
    index_kind: str,
) -> Tuple[np.ndarray, np.ndarray]: ...
def connected_components(
    voxels: np.ndarray,
    connectivity: int,
    index_kind: str,
) -> Tuple[int, np.ndarray]: ...
def index_of(
    voxels: np.ndarray,
    queries: np.ndarray,
    index_kind: str,
) -> np.ndarray: ...
