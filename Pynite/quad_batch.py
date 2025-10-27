"""
Batch computation of quadrilateral element stiffness matrices.

This module provides vectorized computation of multiple quad stiffness matrices
in a single pass, achieving significant performance improvements over individual
element computation.
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Tuple
from math import sqrt

_GAUSS_COORD = 1.0 / sqrt(3.0)


def batch_compute_quad_transformations(
    coords: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Batch compute transformation matrices for multiple quad elements.

    Parameters
    ----------
    coords : ndarray of shape (n_quads, 4, 3)
        Node coordinates for each quad [i, j, m, n] nodes with [X, Y, Z]

    Returns
    -------
    ndarray of shape (n_quads, 24, 24)
        Transformation matrices for all quads
    """
    n_quads = coords.shape[0]
    T_batch = np.zeros((n_quads, 24, 24), dtype=np.float64)

    # Extract node coordinates (vectorized)
    i_coords = coords[:, 0, :]  # shape (n_quads, 3)
    j_coords = coords[:, 1, :]
    n_coords = coords[:, 3, :]

    # Calculate local x-axis (from i to j)
    x_vec = j_coords - i_coords  # shape (n_quads, 3)
    x_mag = np.linalg.norm(x_vec, axis=1, keepdims=True)  # shape (n_quads, 1)
    x_unit = x_vec / x_mag  # normalized x direction

    # Calculate vector in plate plane (from i to n)
    xy_vec = n_coords - i_coords

    # Calculate local z-axis (perpendicular to plate)
    z_vec = np.cross(x_unit, xy_vec)
    z_mag = np.linalg.norm(z_vec, axis=1, keepdims=True)
    z_unit = z_vec / z_mag

    # Calculate local y-axis
    y_unit = np.cross(z_unit, x_unit)

    # Build transformation matrices
    # Create 3x3 rotation matrices for each quad
    for i in range(n_quads):
        dirCos = np.array([
            [x_unit[i, 0], x_unit[i, 1], x_unit[i, 2]],
            [y_unit[i, 0], y_unit[i, 1], y_unit[i, 2]],
            [z_unit[i, 0], z_unit[i, 1], z_unit[i, 2]]
        ])

        # Populate the 24x24 transformation matrix (4 nodes × 6 DOF)
        for node_idx in range(4):
            offset = node_idx * 6
            T_batch[i, offset:offset+3, offset:offset+3] = dirCos
            T_batch[i, offset+3:offset+6, offset+3:offset+6] = dirCos

    return T_batch


def batch_compute_quad_stiffness(
    quads: List,
    use_cache: bool = True,
    n_workers: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Batch compute global stiffness matrices for multiple quad elements.

    Uses multiprocessing to parallelize computation when beneficial.

    Parameters
    ----------
    quads : list of Quad3D
        List of quadrilateral elements to process
    use_cache : bool, optional
        Whether to use cached stiffness matrices where available
    n_workers : int, optional
        Number of parallel workers. If None, uses single-threaded.

    Returns
    -------
    ndarray of shape (n_quads, 24, 24)
        Global stiffness matrices for all quads
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_quads = len(quads)
    K_global = np.zeros((n_quads, 24, 24), dtype=np.float64)

    # First pass: use cached values
    need_compute = []
    for idx, quad in enumerate(quads):
        if use_cache and quad._K_global_cache is not None:
            K_global[idx] = quad._K_global_cache
        else:
            need_compute.append(idx)

    if not need_compute:
        return K_global

    # Decide whether to use parallel processing
    use_parallel = n_workers is not None and n_workers > 1 and len(need_compute) > 100

    if use_parallel:
        # Parallel computation using ThreadPoolExecutor (GIL-friendly for numpy operations)
        def compute_one(idx):
            quad = quads[idx]
            K = quad.K()  # This will compute T and k
            if use_cache:
                quad._K_global_cache = K
            return idx, K

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(compute_one, idx) for idx in need_compute]
            for future in as_completed(futures):
                idx, K = future.result()
                K_global[idx] = K
    else:
        # Single-threaded computation
        for idx in need_compute:
            quad = quads[idx]
            K = quad.K()
            K_global[idx] = K
            if use_cache:
                quad._K_global_cache = K

    return K_global


def batch_compute_quad_stiffness_parallel(
    quads: List,
    n_workers: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Parallel batch computation of quad stiffness matrices using multiprocessing.

    Parameters
    ----------
    quads : list of Quad3D
        List of quadrilateral elements to process
    n_workers : int, optional
        Number of worker processes. If None, uses number of CPU cores.

    Returns
    -------
    ndarray of shape (n_quads, 24, 24)
        Global stiffness matrices for all quads
    """
    from multiprocessing import Pool, cpu_count

    if n_workers is None:
        n_workers = cpu_count()

    # Split quads into chunks for parallel processing
    n_quads = len(quads)
    chunk_size = max(1, n_quads // n_workers)
    chunks = [quads[i:i+chunk_size] for i in range(0, n_quads, chunk_size)]

    # Process chunks in parallel
    with Pool(n_workers) as pool:
        results = pool.map(batch_compute_quad_stiffness, chunks)

    # Concatenate results
    return np.concatenate(results, axis=0)


def optimize_quad_assembly(model_quads: dict, n_workers: int = 4) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
    """
    Optimized assembly routine for quadrilateral stiffness matrices.

    Uses parallel processing to compute multiple quad stiffness matrices simultaneously.

    Parameters
    ----------
    model_quads : dict
        Dictionary of quad elements from FEModel3D
    n_workers : int, optional
        Number of parallel workers for computation

    Returns
    -------
    quad_dofs : ndarray of shape (n_quads, 24)
        DOF indices for each quad
    quad_stiffness : ndarray of shape (n_quads, 24, 24)
        Global stiffness matrices for all quads
    """
    if not model_quads:
        return np.empty((0, 24), dtype=np.int32), np.empty((0, 24, 24), dtype=np.float64)

    quads_list = list(model_quads.values())
    n_quads = len(quads_list)

    # Allocate output arrays
    quad_dofs = np.empty((n_quads, 24), dtype=np.int32)

    # Extract DOF mappings (this is still fast enough in Python)
    for idx, quad in enumerate(quads_list):
        i_id = quad.i_node.ID
        j_id = quad.j_node.ID
        m_id = quad.m_node.ID
        n_id = quad.n_node.ID
        quad_dofs[idx] = [
            i_id*6, i_id*6+1, i_id*6+2, i_id*6+3, i_id*6+4, i_id*6+5,
            j_id*6, j_id*6+1, j_id*6+2, j_id*6+3, j_id*6+4, j_id*6+5,
            m_id*6, m_id*6+1, m_id*6+2, m_id*6+3, m_id*6+4, m_id*6+5,
            n_id*6, n_id*6+1, n_id*6+2, n_id*6+3, n_id*6+4, n_id*6+5,
        ]

    # Batch compute all stiffness matrices with parallelization
    quad_stiffness = batch_compute_quad_stiffness(quads_list, use_cache=True, n_workers=n_workers)

    return quad_dofs, quad_stiffness
