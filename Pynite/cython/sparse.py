# cython: language_level=3, boundscheck=False, wraparound=False
"""
Sparse assembly helpers compiled with Cython.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def expand_stiffness_blocks(
    dofs: NDArray[np.int_],
    stiffness: NDArray[np.float_],
) -> Tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    """
    Expand local stiffness matrices into COO triplets.
    """

    if dofs.size == 0:
        empty_i = np.empty(0, dtype=np.int32)
        empty_v = np.empty(0, dtype=np.float64)
        return empty_i, empty_i, empty_v

    dofs_32 = np.asarray(dofs, dtype=np.int32)
    stiffness_64 = np.asarray(stiffness, dtype=np.float64)

    block = dofs_32.shape[1]
    if stiffness_64.shape != (dofs_32.shape[0], block, block):
        raise ValueError("stiffness array shape is inconsistent with dof array")

    rows = np.repeat(dofs_32, block, axis=1).ravel()
    cols = np.tile(dofs_32, (1, block)).ravel()
    data = stiffness_64.reshape(-1)

    return rows, cols, data


__all__ = ["expand_stiffness_blocks"]
