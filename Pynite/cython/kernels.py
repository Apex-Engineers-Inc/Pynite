# cython: language_level=3, boundscheck=False, wraparound=False
"""
Numerical kernels compiled with Cython for PyNite.

These functions were originally implemented as Numba JIT kernels. They are now
exposed as Cython-accelerated routines using pure-Python syntax with typed
signatures so the extension can be built directly from this module.
"""

from __future__ import annotations

from math import sqrt
from typing import Final, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    double = float  # type: ignore[assignment]
else:  # pragma: no cover - executed only when Cython is available during builds
    try:
        from cython import double  # type: ignore[attr-defined]
    except ImportError:
        double = float  # type: ignore[assignment]

FLOAT = np.float64
MEMBRANE_MAP = np.array([0, 1, 6, 7, 12, 13, 18, 19], dtype=np.int64)
BENDING_MAP = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22], dtype=np.int64)


def beam_member_stiffness_matrix(
    E: double,
    G: double,
    A: double,
    Iy: double,
    Iz: double,
    J: double,
    L: double,
) -> NDArray[FLOAT]:
    """
    Returns the 12x12 local stiffness matrix for a prismatic 3D frame member.
    """

    # fmt: off
    return np.array(
        [
            [
                A * E / L, 0.0, 0.0, 0.0, 0.0, 0.0, -A * E / L, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            [
                0.0, 12 * E * Iz / L**3, 0.0, 0.0, 0.0, 6 * E * Iz / L**2, 0.0,
                -12 * E * Iz / L**3, 0.0, 0.0, 0.0, 6 * E * Iz / L**2
            ],
            [
                0.0, 0.0, 12 * E * Iy / L**3, 0.0, -6 * E * Iy / L**2, 0.0, 0.0, 0.0,
                -12 * E * Iy / L**3, 0.0, -6 * E * Iy / L**2, 0.0
            ],
            [
                0.0, 0.0, 0.0, G * J / L, 0.0, 0.0, 0.0, 0.0, 0.0, -G * J / L, 0.0, 0.0
            ],
            [
                0.0, 0.0, -6 * E * Iy / L**2, 0.0, 4 * E * Iy / L, 0.0, 0.0, 0.0,
                6 * E * Iy / L**2, 0.0, 2 * E * Iy / L, 0.0
            ],
            [
                0.0, 6 * E * Iz / L**2, 0.0, 0.0, 0.0, 4 * E * Iz / L, 0.0,
                -6 * E * Iz / L**2, 0.0, 0.0, 0.0, 2 * E * Iz / L
            ],
            [
                -A * E / L, 0.0, 0.0, 0.0, 0.0, 0.0, A * E / L, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            [
                0.0, -12 * E * Iz / L**3, 0.0, 0.0, 0.0, -6 * E * Iz / L**2, 0.0,
                12 * E * Iz / L**3, 0.0, 0.0, 0.0, -6 * E * Iz / L**2
            ],
            [
                0.0, 0.0, -12 * E * Iy / L**3, 0.0, 6 * E * Iy / L**2, 0.0, 0.0, 0.0,
                12 * E * Iy / L**3, 0.0, 6 * E * Iy / L**2, 0.0
            ],
            [
                0.0, 0.0, 0.0, -G * J / L, 0.0, 0.0, 0.0, 0.0, 0.0, G * J / L, 0.0, 0.0
            ],
            [
                0.0, 0.0, -6 * E * Iy / L**2, 0.0, 2 * E * Iy / L, 0.0, 0.0, 0.0,
                6 * E * Iy / L**2, 0.0, 4 * E * Iy / L, 0.0
            ],
            [
                0.0, 6 * E * Iz / L**2, 0.0, 0.0, 0.0, 2 * E * Iz / L, 0.0,
                -6 * E * Iz / L**2, 0.0, 0.0, 0.0, 4 * E * Iz / L
            ],
        ],
        dtype=FLOAT,
    )
    # fmt: on


def fer_point_load(P: double, x: double, L: double, axis: int) -> NDArray[FLOAT]:
    """
    Fixed-end reactions for a transverse point load.

    ``axis`` == 0 targets the local y-axis ("Fy"), while ``axis`` == 1 targets
    the local z-axis ("Fz").
    """

    b = L - x
    fer = np.zeros((12, 1), dtype=FLOAT)

    if axis == 0:
        fer[1, 0] = -P * b**2 * (L + 2 * x) / L**3
        fer[5, 0] = -P * x * b**2 / L**2
        fer[7, 0] = -P * x**2 * (L + 2 * b) / L**3
        fer[11, 0] = P * x**2 * b / L**2
    else:
        fer[2, 0] = -P * b**2 * (L + 2 * x) / L**3
        fer[4, 0] = P * x * b**2 / L**2
        fer[8, 0] = -P * x**2 * (L + 2 * b) / L**3
        fer[10, 0] = -P * x**2 * b / L**2

    return fer


def fer_moment(M: double, x: double, L: double, axis: int) -> NDArray[FLOAT]:
    """
    Fixed-end reactions for a concentrated moment.
    """

    b = L - x
    fer = np.zeros((12, 1), dtype=FLOAT)

    if axis == 0:
        fer[2, 0] = -6.0 * M * x * b / L**3
        fer[4, 0] = M * b * (2.0 * x - b) / L**2
        fer[8, 0] = 6.0 * M * x * b / L**3
        fer[10, 0] = M * x * (2.0 * b - x) / L**2
    else:
        fer[1, 0] = 6.0 * M * x * b / L**3
        fer[5, 0] = M * b * (2.0 * x - b) / L**2
        fer[7, 0] = -6.0 * M * x * b / L**3
        fer[11, 0] = M * x * (2.0 * b - x) / L**2

    return fer


def fer_linear_transverse_load(
    w1: double,
    w2: double,
    x1: double,
    x2: double,
    L: double,
    axis: int,
) -> NDArray[FLOAT]:
    """
    Fixed-end reactions for a linearly varying transverse load.
    """

    fer = np.zeros((12, 1), dtype=FLOAT)
    term_common = x1 - x2

    if axis == 0:
        fer[1, 0] = (
            term_common
            * (
                10.0 * L**3 * w1
                + 10.0 * L**3 * w2
                - 15.0 * L * w1 * x1**2
                - 10.0 * L * w1 * x1 * x2
                - 5.0 * L * w1 * x2**2
                - 5.0 * L * w2 * x1**2
                - 10.0 * L * w2 * x1 * x2
                - 15.0 * L * w2 * x2**2
                + 8.0 * w1 * x1**3
                + 6.0 * w1 * x1**2 * x2
                + 4.0 * w1 * x1 * x2**2
                + 2.0 * w1 * x2**3
                + 2.0 * w2 * x1**3
                + 4.0 * w2 * x1**2 * x2
                + 6.0 * w2 * x1 * x2**2
                + 8.0 * w2 * x2**3
            )
            / (20.0 * L**3)
        )

        fer[5, 0] = (
            term_common
            * (
                20.0 * L**2 * w1 * x1
                + 10.0 * L**2 * w1 * x2
                + 10.0 * L**2 * w2 * x1
                + 20.0 * L**2 * w2 * x2
                - 30.0 * L * w1 * x1**2
                - 20.0 * L * w1 * x1 * x2
                - 10.0 * L * w1 * x2**2
                - 10.0 * L * w2 * x1**2
                - 20.0 * L * w2 * x1 * x2
                - 30.0 * L * w2 * x2**2
                + 12.0 * w1 * x1**3
                + 9.0 * w1 * x1**2 * x2
                + 6.0 * w1 * x1 * x2**2
                + 3.0 * w1 * x2**3
                + 3.0 * w2 * x1**3
                + 6.0 * w2 * x1**2 * x2
                + 9.0 * w2 * x1 * x2**2
                + 12.0 * w2 * x2**3
            )
            / (60.0 * L**2)
        )

        fer[7, 0] = (
            -term_common
            * (
                -15.0 * L * w1 * x1**2
                - 10.0 * L * w1 * x1 * x2
                - 5.0 * L * w1 * x2**2
                - 5.0 * L * w2 * x1**2
                - 10.0 * L * w2 * x1 * x2
                - 15.0 * L * w2 * x2**2
                + 8.0 * w1 * x1**3
                + 6.0 * w1 * x1**2 * x2
                + 4.0 * w1 * x1 * x2**2
                + 2.0 * w1 * x2**3
                + 2.0 * w2 * x1**3
                + 4.0 * w2 * x1**2 * x2
                + 6.0 * w2 * x1 * x2**2
                + 8.0 * w2 * x2**3
            )
            / (20.0 * L**3)
        )

        fer[11, 0] = (
            term_common
            * (
                -15.0 * L * w1 * x1**2
                - 10.0 * L * w1 * x1 * x2
                - 5.0 * L * w1 * x2**2
                - 5.0 * L * w2 * x1**2
                - 10.0 * L * w2 * x1 * x2
                - 15.0 * L * w2 * x2**2
                + 12.0 * w1 * x1**3
                + 9.0 * w1 * x1**2 * x2
                + 6.0 * w1 * x1 * x2**2
                + 3.0 * w1 * x2**3
                + 3.0 * w2 * x1**3
                + 6.0 * w2 * x1**2 * x2
                + 9.0 * w2 * x1 * x2**2
                + 12.0 * w2 * x2**3
            )
            / (60.0 * L**2)
        )

    else:
        fer[2, 0] = (
            term_common
            * (
                10.0 * L**3 * w1
                + 10.0 * L**3 * w2
                - 15.0 * L * w1 * x1**2
                - 10.0 * L * w1 * x1 * x2
                - 5.0 * L * w1 * x2**2
                - 5.0 * L * w2 * x1**2
                - 10.0 * L * w2 * x1 * x2
                - 15.0 * L * w2 * x2**2
                + 8.0 * w1 * x1**3
                + 6.0 * w1 * x1**2 * x2
                + 4.0 * w1 * x1 * x2**2
                + 2.0 * w1 * x2**3
                + 2.0 * w2 * x1**3
                + 4.0 * w2 * x1**2 * x2
                + 6.0 * w2 * x1 * x2**2
                + 8.0 * w2 * x2**3
            )
            / (20.0 * L**3)
        )

        fer[4, 0] = (
            -term_common
            * (
                20.0 * L**2 * w1 * x1
                + 10.0 * L**2 * w1 * x2
                + 10.0 * L**2 * w2 * x1
                + 20.0 * L**2 * w2 * x2
                - 30.0 * L * w1 * x1**2
                - 20.0 * L * w1 * x1 * x2
                - 10.0 * L * w1 * x2**2
                - 10.0 * L * w2 * x1**2
                - 20.0 * L * w2 * x1 * x2
                - 30.0 * L * w2 * x2**2
                + 12.0 * w1 * x1**3
                + 9.0 * w1 * x1**2 * x2
                + 6.0 * w1 * x1 * x2**2
                + 3.0 * w1 * x2**3
                + 3.0 * w2 * x1**3
                + 6.0 * w2 * x1**2 * x2
                + 9.0 * w2 * x1 * x2**2
                + 12.0 * w2 * x2**3
            )
            / (60.0 * L**2)
        )

        fer[8, 0] = (
            -term_common
            * (
                -15.0 * L * w1 * x1**2
                - 10.0 * L * w1 * x1 * x2
                - 5.0 * L * w1 * x2**2
                - 5.0 * L * w2 * x1**2
                - 10.0 * L * w2 * x1 * x2
                - 15.0 * L * w2 * x2**2
                + 8.0 * w1 * x1**3
                + 6.0 * w1 * x1**2 * x2
                + 4.0 * w1 * x1 * x2**2
                + 2.0 * w1 * x2**3
                + 2.0 * w2 * x1**3
                + 4.0 * w2 * x1**2 * x2
                + 6.0 * w2 * x1 * x2**2
                + 8.0 * w2 * x2**3
            )
            / (20.0 * L**3)
        )

        fer[10, 0] = (
            -term_common
            * (
                -15.0 * L * w1 * x1**2
                - 10.0 * L * w1 * x1 * x2
                - 5.0 * L * w1 * x2**2
                - 5.0 * L * w2 * x1**2
                - 10.0 * L * w2 * x1 * x2
                - 15.0 * L * w2 * x2**2
                + 12.0 * w1 * x1**3
                + 9.0 * w1 * x1**2 * x2
                + 6.0 * w1 * x1 * x2**2
                + 3.0 * w1 * x2**3
                + 3.0 * w2 * x1**3
                + 6.0 * w2 * x1**2 * x2
                + 9.0 * w2 * x1 * x2**2
                + 12.0 * w2 * x2**3
            )
            / (60.0 * L**2)
        )

    return fer


def fer_axial_point_load(P: double, x: double, L: double) -> NDArray[FLOAT]:
    fer = np.zeros((12, 1), dtype=FLOAT)
    fer[0, 0] = -P * (L - x) / L
    fer[6, 0] = -P * x / L
    return fer


def fer_axial_linear_load(
    p1: double,
    p2: double,
    x1: double,
    x2: double,
    L: double,
) -> NDArray[FLOAT]:
    fer = np.zeros((12, 1), dtype=FLOAT)
    fer[0, 0] = (
        (x1 - x2)
        * (
            3.0 * L * (p1 + p2)
            - 2.0 * p1 * x1
            - p1 * x2
            - p2 * x1
            - 2.0 * p2 * x2
        )
        / (6.0 * L)
    )
    fer[6, 0] = (
        (x1 - x2)
        * (2.0 * p1 * x1 + p1 * x2 + p2 * x1 + 2.0 * p2 * x2)
        / (6.0 * L)
    )
    return fer


def fer_torque(T: double, x: double, L: double) -> NDArray[FLOAT]:
    fer = np.zeros((12, 1), dtype=FLOAT)
    fer[3, 0] = -T * (L - x) / L
    fer[9, 0] = -T * x / L
    return fer


def evaluate_polynomial(
    coeffs: NDArray[FLOAT], x_values: NDArray[FLOAT]
) -> NDArray[FLOAT]:
    n = x_values.size
    result = np.empty(n, dtype=FLOAT)

    degree = coeffs.size - 1

    for i in range(n):
        x = x_values[i]
        value = coeffs[degree]
        for j in range(degree - 1, -1, -1):
            value = value * x + coeffs[j]
        result[i] = value

    return result


def evaluate_piecewise_polynomial(
    starts: NDArray[FLOAT],
    ends: NDArray[FLOAT],
    coeff_matrix: NDArray[FLOAT],
    degrees: NDArray[np.int64],
    x_values: NDArray[FLOAT],
    tol: double = 1e-12,
) -> NDArray[FLOAT]:
    """
    Evaluate a set of Horner polynomials over contiguous piecewise intervals.
    """

    if x_values.size == 0 or starts.size == 0:
        return np.empty(x_values.size, dtype=FLOAT)

    result = np.empty(x_values.size, dtype=FLOAT)

    if starts.size > 1:
        segment_bounds = np.searchsorted(x_values, ends[:-1], side="right")
    else:
        segment_bounds = np.empty(0, dtype=np.int64)

    start_index = 0
    num_points = x_values.size

    for seg_idx in range(starts.size):
        end_index = segment_bounds[seg_idx] if seg_idx < segment_bounds.size else num_points
        if end_index <= start_index:
            continue

        start = starts[seg_idx]
        span = ends[seg_idx] - start
        degree = int(degrees[seg_idx])
        coeffs = coeff_matrix[seg_idx]

        for idx in range(start_index, end_index):
            local_x = x_values[idx] - start

            if local_x < -tol or local_x - span > tol:
                raise ValueError("x position lies outside the supplied segment interval.")

            if local_x < 0.0 and local_x > -tol:
                local_x = 0.0
            elif local_x > span and local_x - span < tol:
                local_x = span

            value = coeffs[degree]
            for power in range(degree - 1, -1, -1):
                value = value * local_x + coeffs[power]
            result[idx] = value

        start_index = end_index
        if start_index >= num_points:
            break

    return result


def compute_ring_trig(num_divisions: int) -> Tuple[NDArray[FLOAT], NDArray[FLOAT]]:
    cos_vals = np.empty(num_divisions, dtype=FLOAT)
    sin_vals = np.empty(num_divisions, dtype=FLOAT)

    theta = 2.0 * np.pi / num_divisions
    for i in range(num_divisions):
        angle = theta * i
        cos_vals[i] = np.cos(angle)
        sin_vals[i] = np.sin(angle)

    return cos_vals, sin_vals


def cross_product_3d(a: NDArray[FLOAT], b: NDArray[FLOAT]) -> NDArray[FLOAT]:
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=FLOAT,
    )


def normalize_vector_3d(v: NDArray[FLOAT]) -> NDArray[FLOAT]:
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("Cannot normalize zero-length vector")
    return v / norm


def accumulate_membrane_stiffness(
    B_stack: NDArray[FLOAT],
    constitutive: NDArray[FLOAT],
    det_jacobians: NDArray[FLOAT],
    thickness: double,
) -> NDArray[FLOAT]:
    k = np.einsum("gmi,mn,gnj,g->ij", B_stack, constitutive, B_stack, det_jacobians, optimize=True)
    return k * thickness


def accumulate_bending_shear_stiffness(
    Bb_stack: NDArray[FLOAT],
    Hb: NDArray[FLOAT],
    Bs_stack: NDArray[FLOAT],
    Hs: NDArray[FLOAT],
    det_jacobians: NDArray[FLOAT],
) -> NDArray[FLOAT]:
    bending = np.einsum("gmi,mn,gnj,g->ij", Bb_stack, Hb, Bb_stack, det_jacobians, optimize=True)
    shear = np.einsum("gmi,mn,gnj,g->ij", Bs_stack, Hs, Bs_stack, det_jacobians, optimize=True)
    return bending + shear


def _expand_matrix_24(
    k_unexpanded: NDArray[FLOAT], mapping: NDArray[np.int64]
) -> NDArray[FLOAT]:
    result = np.zeros((24, 24), dtype=FLOAT)
    size = mapping.size
    for i in range(size):
        row = mapping[i]
        for j in range(size):
            col = mapping[j]
            result[row, col] = k_unexpanded[i, j]
    return result


def expand_membrane_matrix(k_unexpanded: NDArray[FLOAT]) -> NDArray[FLOAT]:
    return _expand_matrix_24(k_unexpanded, MEMBRANE_MAP)


def expand_bending_matrix(
    k_unexpanded: NDArray[FLOAT]
) -> NDArray[FLOAT]:
    return _expand_matrix_24(k_unexpanded, BENDING_MAP)


def _quad_moment_point(
    d: NDArray[FLOAT],
    Hb: NDArray[FLOAT],
    Bb_stack: NDArray[FLOAT],
    xi: double,
    eta: double,
) -> Tuple[FLOAT, FLOAT, FLOAT]:
    gp = 1.0 / sqrt(3.0)
    xi_ex = xi / gp
    eta_ex = eta / gp

    w0 = 0.25 * (1.0 - xi_ex) * (1.0 - eta_ex)
    w1 = 0.25 * (1.0 + xi_ex) * (1.0 - eta_ex)
    w2 = 0.25 * (1.0 + xi_ex) * (1.0 + eta_ex)
    w3 = 0.25 * (1.0 - xi_ex) * (1.0 + eta_ex)

    accum0 = 0.0
    accum1 = 0.0
    accum2 = 0.0

    for gp_index in range(4):
        B = Bb_stack[gp_index]
        t0 = 0.0
        t1 = 0.0
        t2 = 0.0
        for j in range(12):
            val = d[j]
            t0 += float(B[0, j]) * val
            t1 += float(B[1, j]) * val
            t2 += float(B[2, j]) * val

        m0 = float(Hb[0, 0] * t0 + Hb[0, 1] * t1 + Hb[0, 2] * t2)
        m1 = float(Hb[1, 0] * t0 + Hb[1, 1] * t1 + Hb[1, 2] * t2)
        m2 = float(Hb[2, 0] * t0 + Hb[2, 1] * t1 + Hb[2, 2] * t2)

        if gp_index == 0:
            weight = w0
        elif gp_index == 1:
            weight = w1
        elif gp_index == 2:
            weight = w2
        else:
            weight = w3

        accum0 += weight * m0
        accum1 += weight * m1
        accum2 += weight * m2

    return FLOAT(accum0), FLOAT(accum1), FLOAT(accum2)


def quad_moment_at(
    d: NDArray[FLOAT],
    Hb: NDArray[FLOAT],
    Bb_stack: NDArray[FLOAT],
    xi: double,
    eta: double,
) -> NDArray[FLOAT]:
    m0, m1, m2 = _quad_moment_point(d, Hb, Bb_stack, xi, eta)
    result = np.empty(3, dtype=FLOAT)
    result[0] = m0
    result[1] = m1
    result[2] = m2
    return result


def quad_moment_batch(
    d: NDArray[FLOAT],
    Hb: NDArray[FLOAT],
    Bb_stack: NDArray[FLOAT],
    xi_vals: NDArray[FLOAT],
    eta_vals: NDArray[FLOAT],
) -> NDArray[FLOAT]:
    n_points = xi_vals.size
    out = np.empty((n_points, 3), dtype=FLOAT)
    for idx in range(n_points):
        m0, m1, m2 = _quad_moment_point(
            d, Hb, Bb_stack, float(xi_vals[idx]), float(eta_vals[idx])
        )
        out[idx, 0] = m0
        out[idx, 1] = m1
        out[idx, 2] = m2
    return out


def _quad_membrane_point(
    d: NDArray[FLOAT],
    Cm: NDArray[FLOAT],
    Bm_stack: NDArray[FLOAT],
    xi: double,
    eta: double,
) -> Tuple[FLOAT, FLOAT, FLOAT]:
    gp = 1.0 / sqrt(3.0)
    xi_ex = xi / gp
    eta_ex = eta / gp

    w0 = 0.25 * (1.0 - xi_ex) * (1.0 - eta_ex)
    w1 = 0.25 * (1.0 + xi_ex) * (1.0 - eta_ex)
    w2 = 0.25 * (1.0 + xi_ex) * (1.0 + eta_ex)
    w3 = 0.25 * (1.0 - xi_ex) * (1.0 + eta_ex)

    accum0 = 0.0
    accum1 = 0.0
    accum2 = 0.0

    for gp_index in range(4):
        B = Bm_stack[gp_index]
        t0 = 0.0
        t1 = 0.0
        t2 = 0.0
        for j in range(8):
            val = d[j]
            t0 += float(B[0, j]) * val
            t1 += float(B[1, j]) * val
            t2 += float(B[2, j]) * val

        s0 = float(Cm[0, 0] * t0 + Cm[0, 1] * t1 + Cm[0, 2] * t2)
        s1 = float(Cm[1, 0] * t0 + Cm[1, 1] * t1 + Cm[1, 2] * t2)
        s2 = float(Cm[2, 0] * t0 + Cm[2, 1] * t1 + Cm[2, 2] * t2)

        if gp_index == 0:
            weight = w0
        elif gp_index == 1:
            weight = w1
        elif gp_index == 2:
            weight = w2
        else:
            weight = w3

        accum0 += weight * s0
        accum1 += weight * s1
        accum2 += weight * s2

    return FLOAT(accum0), FLOAT(accum1), FLOAT(accum2)


def quad_membrane_at(
    d: NDArray[FLOAT],
    Cm: NDArray[FLOAT],
    Bm_stack: NDArray[FLOAT],
    xi: double,
    eta: double,
) -> NDArray[FLOAT]:
    s0, s1, s2 = _quad_membrane_point(d, Cm, Bm_stack, xi, eta)
    result = np.empty(3, dtype=FLOAT)
    result[0] = s0
    result[1] = s1
    result[2] = s2
    return result


def quad_membrane_batch(
    d: NDArray[FLOAT],
    Cm: NDArray[FLOAT],
    Bm_stack: NDArray[FLOAT],
    xi_vals: NDArray[FLOAT],
    eta_vals: NDArray[FLOAT],
) -> NDArray[FLOAT]:
    n_points = xi_vals.size
    out = np.empty((n_points, 3), dtype=FLOAT)
    for idx in range(n_points):
        s0, s1, s2 = _quad_membrane_point(
            d, Cm, Bm_stack, float(xi_vals[idx]), float(eta_vals[idx])
        )
        out[idx, 0] = s0
        out[idx, 1] = s1
        out[idx, 2] = s2
    return out


def compute_quad_local_coords(
    X1: double,
    Y1: double,
    Z1: double,
    X2: double,
    Y2: double,
    Z2: double,
    X3: double,
    Y3: double,
    Z3: double,
    X4: double,
    Y4: double,
    Z4: double,
) -> Tuple[double, double, double, double, double, double, double, double]:
    vector_12 = np.array([X2 - X1, Y2 - Y1, Z2 - Z1], dtype=FLOAT)
    vector_13 = np.array([X3 - X1, Y3 - Y1, Z3 - Z1], dtype=FLOAT)
    vector_14 = np.array([X4 - X1, Y4 - Y1, Z4 - Z1], dtype=FLOAT)

    x_axis = vector_12 / np.linalg.norm(vector_12)
    z_axis = np.cross(x_axis, vector_13)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    x1 = 0.0
    y1 = 0.0
    x2 = float(vector_12 @ x_axis)
    y2 = float(vector_12 @ y_axis)
    x3 = float(vector_13 @ x_axis)
    y3 = float(vector_13 @ y_axis)
    x4 = float(vector_14 @ x_axis)
    y4 = float(vector_14 @ y_axis)

    return x1, y1, x2, y2, x3, y3, x4, y4


def compute_quad_transformation_matrix(
    xi: double,
    yi: double,
    zi: double,
    xj: double,
    yj: double,
    zj: double,
    xn: double,
    yn: double,
    zn: double,
) -> NDArray[FLOAT]:
    x_vec = np.array([xj - xi, yj - yi, zj - zi], dtype=FLOAT)
    x_axis = x_vec / np.linalg.norm(x_vec)

    xy_vec = np.array([xn - xi, yn - yi, zn - zi], dtype=FLOAT)
    z_axis = np.cross(x_axis, xy_vec)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    T = np.zeros((24, 24), dtype=FLOAT)
    for block in range(8):
        offset = block * 3
        T[offset, offset : offset + 3] = x_axis
        T[offset + 1, offset : offset + 3] = y_axis
        T[offset + 2, offset : offset + 3] = z_axis
    return T


def compute_member_transformation_matrix(
    Xi: double,
    Yi: double,
    Zi: double,
    Xj: double,
    Yj: double,
    Zj: double,
    L: double,
    rotation: double,
) -> NDArray[FLOAT]:
    x = np.array([(Xj - Xi) / L, (Yj - Yi) / L, (Zj - Zi) / L], dtype=FLOAT)

    tol = 1e-9

    if abs(Xi - Xj) < tol and abs(Zi - Zj) < tol:
        if Yj > Yi:
            y = np.array([-1.0, 0.0, 0.0], dtype=FLOAT)
            z = np.array([0.0, 0.0, 1.0], dtype=FLOAT)
        else:
            y = np.array([1.0, 0.0, 0.0], dtype=FLOAT)
            z = np.array([0.0, 0.0, 1.0], dtype=FLOAT)
    elif abs(Yi - Yj) < tol:
        y_temp = np.array([0.0, 1.0, 0.0], dtype=FLOAT)
        z = cross_product_3d(x, y_temp)
        z = normalize_vector_3d(z)
        y = cross_product_3d(z, x)
        y = normalize_vector_3d(y)
    else:
        proj = np.array([Xj - Xi, 0.0, Zj - Zi], dtype=FLOAT)
        if Yj > Yi:
            z = cross_product_3d(proj, x)
        else:
            z = cross_product_3d(x, proj)
        z = normalize_vector_3d(z)
        y = cross_product_3d(z, x)
        y = normalize_vector_3d(y)

    if abs(rotation) > 1e-10:
        c = np.cos(rotation)
        s = np.sin(rotation)
        u = x
        v = y
        u_dot_v = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
        u_cross_v = cross_product_3d(u, v)
        y = v * c + u_cross_v * s + u * u_dot_v * (1.0 - c)
        y = normalize_vector_3d(y)
        z = cross_product_3d(x, y)
        z = normalize_vector_3d(z)

    T = np.zeros((12, 12), dtype=FLOAT)
    for i in range(4):
        offset = i * 3
        T[offset + 0, offset + 0] = x[0]
        T[offset + 0, offset + 1] = x[1]
        T[offset + 0, offset + 2] = x[2]
        T[offset + 1, offset + 0] = y[0]
        T[offset + 1, offset + 1] = y[1]
        T[offset + 1, offset + 2] = y[2]
        T[offset + 2, offset + 0] = z[0]
        T[offset + 2, offset + 1] = z[1]
        T[offset + 2, offset + 2] = z[2]

    return T


__all__: Final = [
    "beam_member_stiffness_matrix",
    "fer_point_load",
    "fer_moment",
    "fer_linear_transverse_load",
    "fer_axial_point_load",
    "fer_axial_linear_load",
    "fer_torque",
    "evaluate_polynomial",
    "evaluate_piecewise_polynomial",
    "compute_ring_trig",
    "quad_moment_at",
    "quad_moment_batch",
    "quad_membrane_at",
    "quad_membrane_batch",
    "accumulate_membrane_stiffness",
    "accumulate_bending_shear_stiffness",
    "expand_membrane_matrix",
    "expand_bending_matrix",
    "cross_product_3d",
    "normalize_vector_3d",
    "compute_quad_local_coords",
    "compute_quad_transformation_matrix",
    "compute_member_transformation_matrix",
]
