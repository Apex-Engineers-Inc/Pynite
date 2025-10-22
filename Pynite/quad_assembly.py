"""
High-performance helpers for quadrilateral element stiffness assembly.

This module provides a thin wrapper around the Cython extension.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from Pynite import cython as cython_kernels


def _ensure_float64(array: NDArray[np.float_]) -> NDArray[np.float64]:
    return np.ascontiguousarray(array, dtype=np.float64)


def accumulate_membrane_stiffness(
    B_stack: NDArray[np.float_],
    constitutive: NDArray[np.float_],
    det_jacobians: NDArray[np.float_],
    thickness: float,
) -> NDArray[np.float64]:
    return cython_kernels.accumulate_membrane_stiffness(
        _ensure_float64(B_stack),
        _ensure_float64(constitutive),
        _ensure_float64(det_jacobians),
        float(thickness),
    )


def accumulate_bending_shear_stiffness(
    Bb_stack: NDArray[np.float_],
    Hb: NDArray[np.float_],
    Bs_stack: NDArray[np.float_],
    Hs: NDArray[np.float_],
    det_jacobians: NDArray[np.float_],
) -> NDArray[np.float64]:
    return cython_kernels.accumulate_bending_shear_stiffness(
        _ensure_float64(Bb_stack),
        _ensure_float64(Hb),
        _ensure_float64(Bs_stack),
        _ensure_float64(Hs),
        _ensure_float64(det_jacobians),
    )


def expand_membrane_matrix(
    k_unexpanded: NDArray[np.float_],
) -> NDArray[np.float64]:
    return cython_kernels.expand_membrane_matrix(_ensure_float64(k_unexpanded))


def expand_bending_matrix(
    k_unexpanded: NDArray[np.float_],
) -> NDArray[np.float64]:
    return cython_kernels.expand_bending_matrix(_ensure_float64(k_unexpanded))


def compute_quad_local_coords(
    X1: float,
    Y1: float,
    Z1: float,
    X2: float,
    Y2: float,
    Z2: float,
    X3: float,
    Y3: float,
    Z3: float,
    X4: float,
    Y4: float,
    Z4: float,
) -> Tuple[float, float, float, float, float, float, float, float]:
    return cython_kernels.compute_quad_local_coords(
        float(X1),
        float(Y1),
        float(Z1),
        float(X2),
        float(Y2),
        float(Z2),
        float(X3),
        float(Y3),
        float(Z3),
        float(X4),
        float(Y4),
        float(Z4),
    )


def compute_quad_transformation_matrix(
    xi: float,
    yi: float,
    zi: float,
    xj: float,
    yj: float,
    zj: float,
    xn: float,
    yn: float,
    zn: float,
) -> NDArray[np.float64]:
    return cython_kernels.compute_quad_transformation_matrix(
        float(xi),
        float(yi),
        float(zi),
        float(xj),
        float(yj),
        float(zj),
        float(xn),
        float(yn),
        float(zn),
    )


__all__ = [
    "accumulate_membrane_stiffness",
    "accumulate_bending_shear_stiffness",
    "expand_membrane_matrix",
    "expand_bending_matrix",
    "compute_quad_local_coords",
    "compute_quad_transformation_matrix",
]
