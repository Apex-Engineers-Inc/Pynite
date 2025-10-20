"""
Numba-accelerated numeric kernels used by high-level PyNite classes.

The functions defined here are decorated with :func:`numba.njit`. When Numba is
not available (or disabled), the decorator degrades to a lightweight no-op via
``PyNite.numba_utils`` so these kernels continue to work as regular NumPy
implementations.
"""

from __future__ import annotations

from typing import Final, Tuple

import numpy as np
from numpy.typing import NDArray

from .numba_utils import njit
from math import sqrt


FLOAT = np.float64


@njit(cache=True, fastmath=True)
def beam_member_stiffness_matrix(
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    L: float,
) -> NDArray[FLOAT]:
    """
    Returns the 12x12 local stiffness matrix for a prismatic 3D frame member.
    """

    # fmt: off
    # The coefficients below come directly from closed-form Euler-Bernoulli beam relations
    # and are arranged to match the i-node/j-node DOF ordering used throughout PyNite.
    return np.array(
        [
            # Row | DOF Idx →    0              1                   2                   3               4                   5              6               7                   8                   9               10                  11
            # ----+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            [  # 0 | i-node dx | AE/L           0                   0                   0               0                   0             -AE/L            0                   0                   0               0                   0              ],
                                A * E / L,     0.0,                0.0,                0.0,            0.0,                0.0,          -A * E / L,      0.0,                0.0,                0.0,            0.0,                0.0            ],
            [  # 1 | i-node dy | 0              12EIz/L³            0                   0               0                   6EIz/L²        0              -12EIz/L³            0                   0               0                   6EIz/L²        ],
                                0.0,           12 * E * Iz / L**3, 0.0,                0.0,            0.0,                6 * E * Iz / L**2, 0.0,        -12 * E * Iz / L**3, 0.0,                0.0,            0.0,                6 * E * Iz / L**2],
            [  # 2 | i-node dz | 0              0                   12EIy/L³            0              -6EIy/L²             0              0               0                  -12EIy/L³            0              -6EIy/L²             0              ],
                                0.0,           0.0,                12 * E * Iy / L**3, 0.0,           -6 * E * Iy / L**2,  0.0,          0.0,            0.0,               -12 * E * Iy / L**3,  0.0,           -6 * E * Iy / L**2,  0.0            ],
            [  # 3 | i-node rx | 0              0                   0                   GJ/L            0                   0              0               0                   0                  -GJ/L            0                   0              ],
                                0.0,           0.0,                0.0,                G * J / L,      0.0,                0.0,          0.0,            0.0,                0.0,               -G * J / L,      0.0,                0.0            ],
            [  # 4 | i-node ry | 0              0                  -6EIy/L²             0               4EIy/L              0              0               0                   6EIy/L²             0               2EIy/L              0              ],
                                0.0,           0.0,               -6 * E * Iy / L**2,  0.0,            4 * E * Iy / L,     0.0,          0.0,            0.0,                6 * E * Iy / L**2,   0.0,            2 * E * Iy / L,     0.0            ],
            [  # 5 | i-node rz | 0              6EIz/L²             0                   0               0                   4EIz/L         0              -6EIz/L²             0                   0               0                   2EIz/L         ],
                                0.0,           6 * E * Iz / L**2,  0.0,                0.0,            0.0,                4 * E * Iz / L, 0.0,          -6 * E * Iz / L**2,  0.0,                0.0,            0.0,                2 * E * Iz / L ],
            [  # 6 | j-node dx |-AE/L           0                   0                   0               0                   0              AE/L            0                   0                   0               0                   0              ],
                               -A * E / L,     0.0,                0.0,                0.0,            0.0,                0.0,           A * E / L,      0.0,                0.0,                0.0,            0.0,                0.0            ],
            [  # 7 | j-node dy | 0             -12EIz/L³            0                   0               0                  -6EIz/L²        0               12EIz/L³            0                   0               0                  -6EIz/L²        ],
                                0.0,          -12 * E * Iz / L**3, 0.0,                0.0,            0.0,               -6 * E * Iz / L**2, 0.0,       12 * E * Iz / L**3,  0.0,                0.0,            0.0,               -6 * E * Iz / L**2],
            [  # 8 | j-node dz | 0              0                  -12EIy/L³            0               6EIy/L²             0              0               0                   12EIy/L³            0               6EIy/L²             0              ],
                                0.0,           0.0,               -12 * E * Iy / L**3, 0.0,            6 * E * Iy / L**2,  0.0,          0.0,            0.0,                12 * E * Iy / L**3,  0.0,            6 * E * Iy / L**2,  0.0            ],
            [  # 9 | j-node rx | 0              0                   0                  -GJ/L            0                   0              0               0                   0                   GJ/L            0                   0              ],
                                0.0,           0.0,                0.0,               -G * J / L,      0.0,                0.0,          0.0,            0.0,                0.0,                G * J / L,      0.0,                0.0            ],
            [  #10 | j-node ry | 0              0                  -6EIy/L²             0               2EIy/L              0              0               0                   6EIy/L²             0               4EIy/L              0              ],
                                0.0,           0.0,               -6 * E * Iy / L**2,  0.0,            2 * E * Iy / L,     0.0,          0.0,            0.0,                6 * E * Iy / L**2,   0.0,            4 * E * Iy / L,     0.0            ],
            [  #11 | j-node rz | 0              6EIz/L²             0                   0               0                   2EIz/L         0              -6EIz/L²             0                   0               0                   4EIz/L         ],
                                0.0,           6 * E * Iz / L**2,  0.0,                0.0,            0.0,                2 * E * Iz / L, 0.0,          -6 * E * Iz / L**2,  0.0,                0.0,            0.0,                4 * E * Iz / L ],
        ],
        dtype=FLOAT,
    )
    # fmt: on


@njit(cache=True, fastmath=True)
def fer_point_load(P: float, x: float, L: float, axis: int) -> NDArray[FLOAT]:
    """
    Fixed-end reactions for a transverse point load.

    ``axis`` == 0 targets the local y-axis ("Fy"), while ``axis`` == 1 targets
    the local z-axis ("Fz").
    """

    # Split the span so we can reuse the same expressions for the opposite end reactions
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


@njit(cache=True, fastmath=True)
def fer_moment(M: float, x: float, L: float, axis: int) -> NDArray[FLOAT]:
    """
    Fixed-end reactions for a concentrated moment.

    ``axis`` == 0 corresponds to a moment about the local y-axis ("My"), while
    ``axis`` == 1 corresponds to a moment about the local z-axis ("Mz").
    """

    # As with the point load, use the distance to the remote end to keep the formulas compact
    b = L - x
    fer = np.zeros((12, 1), dtype=FLOAT)

    if axis == 1:
        fer[1, 0] = 6 * M * x * b / L**3
        fer[5, 0] = M * b * (2 * x - b) / L**2
        fer[7, 0] = -6 * M * x * b / L**3
        fer[11, 0] = M * x * (2 * b - x) / L**2
    else:
        fer[2, 0] = -6 * M * x * b / L**3
        fer[4, 0] = M * b * (2 * x - b) / L**2
        fer[8, 0] = 6 * M * x * b / L**3
        fer[10, 0] = M * x * (2 * b - x) / L**2

    return fer


@njit(cache=True, fastmath=True)
def fer_linear_transverse_load(
    w1: float,
    w2: float,
    x1: float,
    x2: float,
    L: float,
    axis: int,
) -> NDArray[FLOAT]:
    """
    Fixed-end reactions for a linearly-varying transverse load.

    ``axis`` == 0 -> local y-axis ("Fy"), ``axis`` == 1 -> local z-axis ("Fz").
    """

    fer = np.zeros((12, 1), dtype=FLOAT)

    # Cache algebraic groupings to reduce the number of floating-point ops inside the kernel
    term_common = x1 - x2

    if axis == 0:
        fer[1, 0] = (
            term_common
            * (
                10 * L**3 * w1
                + 10 * L**3 * w2
                - 15 * L * w1 * x1**2
                - 10 * L * w1 * x1 * x2
                - 5 * L * w1 * x2**2
                - 5 * L * w2 * x1**2
                - 10 * L * w2 * x1 * x2
                - 15 * L * w2 * x2**2
                + 8 * w1 * x1**3
                + 6 * w1 * x1**2 * x2
                + 4 * w1 * x1 * x2**2
                + 2 * w1 * x2**3
                + 2 * w2 * x1**3
                + 4 * w2 * x1**2 * x2
                + 6 * w2 * x1 * x2**2
                + 8 * w2 * x2**3
            )
            / (20 * L**3)
        )

        fer[5, 0] = (
            term_common
            * (
                20 * L**2 * w1 * x1
                + 10 * L**2 * w1 * x2
                + 10 * L**2 * w2 * x1
                + 20 * L**2 * w2 * x2
                - 30 * L * w1 * x1**2
                - 20 * L * w1 * x1 * x2
                - 10 * L * w1 * x2**2
                - 10 * L * w2 * x1**2
                - 20 * L * w2 * x1 * x2
                - 30 * L * w2 * x2**2
                + 12 * w1 * x1**3
                + 9 * w1 * x1**2 * x2
                + 6 * w1 * x1 * x2**2
                + 3 * w1 * x2**3
                + 3 * w2 * x1**3
                + 6 * w2 * x1**2 * x2
                + 9 * w2 * x1 * x2**2
                + 12 * w2 * x2**3
            )
            / (60 * L**2)
        )

        fer[7, 0] = (
            -term_common
            * (
                -15 * L * w1 * x1**2
                - 10 * L * w1 * x1 * x2
                - 5 * L * w1 * x2**2
                - 5 * L * w2 * x1**2
                - 10 * L * w2 * x1 * x2
                - 15 * L * w2 * x2**2
                + 8 * w1 * x1**3
                + 6 * w1 * x1**2 * x2
                + 4 * w1 * x1 * x2**2
                + 2 * w1 * x2**3
                + 2 * w2 * x1**3
                + 4 * w2 * x1**2 * x2
                + 6 * w2 * x1 * x2**2
                + 8 * w2 * x2**3
            )
            / (20 * L**3)
        )

        fer[11, 0] = (
            term_common
            * (
                -15 * L * w1 * x1**2
                - 10 * L * w1 * x1 * x2
                - 5 * L * w1 * x2**2
                - 5 * L * w2 * x1**2
                - 10 * L * w2 * x1 * x2
                - 15 * L * w2 * x2**2
                + 12 * w1 * x1**3
                + 9 * w1 * x1**2 * x2
                + 6 * w1 * x1 * x2**2
                + 3 * w1 * x2**3
                + 3 * w2 * x1**3
                + 6 * w2 * x1**2 * x2
                + 9 * w2 * x1 * x2**2
                + 12 * w2 * x2**3
            )
            / (60 * L**2)
        )

    else:
        fer[2, 0] = (
            term_common
            * (
                10 * L**3 * w1
                + 10 * L**3 * w2
                - 15 * L * w1 * x1**2
                - 10 * L * w1 * x1 * x2
                - 5 * L * w1 * x2**2
                - 5 * L * w2 * x1**2
                - 10 * L * w2 * x1 * x2
                - 15 * L * w2 * x2**2
                + 8 * w1 * x1**3
                + 6 * w1 * x1**2 * x2
                + 4 * w1 * x1 * x2**2
                + 2 * w1 * x2**3
                + 2 * w2 * x1**3
                + 4 * w2 * x1**2 * x2
                + 6 * w2 * x1 * x2**2
                + 8 * w2 * x2**3
            )
            / (20 * L**3)
        )

        fer[4, 0] = (
            -term_common
            * (
                20 * L**2 * w1 * x1
                + 10 * L**2 * w1 * x2
                + 10 * L**2 * w2 * x1
                + 20 * L**2 * w2 * x2
                - 30 * L * w1 * x1**2
                - 20 * L * w1 * x1 * x2
                - 10 * L * w1 * x2**2
                - 10 * L * w2 * x1**2
                - 20 * L * w2 * x1 * x2
                - 30 * L * w2 * x2**2
                + 12 * w1 * x1**3
                + 9 * w1 * x1**2 * x2
                + 6 * w1 * x1 * x2**2
                + 3 * w1 * x2**3
                + 3 * w2 * x1**3
                + 6 * w2 * x1**2 * x2
                + 9 * w2 * x1 * x2**2
                + 12 * w2 * x2**3
            )
            / (60 * L**2)
        )

        fer[8, 0] = (
            -term_common
            * (
                -15 * L * w1 * x1**2
                - 10 * L * w1 * x1 * x2
                - 5 * L * w1 * x2**2
                - 5 * L * w2 * x1**2
                - 10 * L * w2 * x1 * x2
                - 15 * L * w2 * x2**2
                + 8 * w1 * x1**3
                + 6 * w1 * x1**2 * x2
                + 4 * w1 * x1 * x2**2
                + 2 * w1 * x2**3
                + 2 * w2 * x1**3
                + 4 * w2 * x1**2 * x2
                + 6 * w2 * x1 * x2**2
                + 8 * w2 * x2**3
            )
            / (20 * L**3)
        )

        fer[10, 0] = (
            -term_common
            * (
                -15 * L * w1 * x1**2
                - 10 * L * w1 * x1 * x2
                - 5 * L * w1 * x2**2
                - 5 * L * w2 * x1**2
                - 10 * L * w2 * x1 * x2
                - 15 * L * w2 * x2**2
                + 12 * w1 * x1**3
                + 9 * w1 * x1**2 * x2
                + 6 * w1 * x1 * x2**2
                + 3 * w1 * x2**3
                + 3 * w2 * x1**3
                + 6 * w2 * x1**2 * x2
                + 9 * w2 * x1 * x2**2
                + 12 * w2 * x2**3
            )
            / (60 * L**2)
        )

    return fer


@njit(cache=True, fastmath=True)
def fer_axial_point_load(P: float, x: float, L: float) -> NDArray[FLOAT]:
    fer = np.zeros((12, 1), dtype=FLOAT)
    fer[0, 0] = -P * (L - x) / L
    fer[6, 0] = -P * x / L
    return fer


@njit(cache=True, fastmath=True)
def fer_axial_linear_load(
    p1: float, p2: float, x1: float, x2: float, L: float
) -> NDArray[FLOAT]:
    fer = np.zeros((12, 1), dtype=FLOAT)
    fer[0, 0] = (
        (x1 - x2)
        * (3 * L * (p1 + p2) - 2 * p1 * x1 - p1 * x2 - p2 * x1 - 2 * p2 * x2)
        / (6 * L)
    )
    fer[6, 0] = (x1 - x2) * (2 * p1 * x1 + p1 * x2 + p2 * x1 + 2 * p2 * x2) / (6 * L)
    return fer


@njit(cache=True, fastmath=True)
def fer_torque(T: float, x: float, L: float) -> NDArray[FLOAT]:
    fer = np.zeros((12, 1), dtype=FLOAT)
    fer[3, 0] = -T * (L - x) / L
    fer[9, 0] = -T * x / L
    return fer


@njit(cache=True, fastmath=True)
def accumulate_membrane_stiffness(
    B_stack: NDArray[FLOAT],
    constitutive: NDArray[FLOAT],
    det_jacobians: NDArray[FLOAT],
    thickness: float,
) -> NDArray[FLOAT]:
    """
    Assemble the unexpanded 8x8 membrane stiffness matrix.

    Parameters
    ----------
    B_stack :
        Shape ``(n_gauss, 3, 8)`` array containing the membrane ``B`` matrices
        evaluated at each Gauss point.
    constitutive :
        ``(3, 3)`` plane-stress constitutive matrix for the element material.
    det_jacobians :
        Length ``n_gauss`` array of Jacobian determinants at the integration
        points.
    thickness :
        Element thickness used to scale the stiffness terms.

    Returns
    -------
    numpy.ndarray
        The unexpanded membrane stiffness matrix in local coordinates.
    """

    n_points = det_jacobians.shape[0]
    k = np.zeros((8, 8), dtype=FLOAT)

    for gp in range(n_points):
        B = B_stack[gp]
        detJ = det_jacobians[gp]
        k += (B.T @ (constitutive @ B)) * detJ

    return k * thickness


@njit(cache=True, fastmath=True)
def accumulate_bending_shear_stiffness(
    Bb_stack: NDArray[FLOAT],
    Hb: NDArray[FLOAT],
    Bs_stack: NDArray[FLOAT],
    Hs: NDArray[FLOAT],
    det_jacobians: NDArray[FLOAT],
) -> NDArray[FLOAT]:
    """
    Assemble the unexpanded 12×12 bending + shear stiffness matrix.

    Parameters
    ----------
    Bb_stack :
        Shape ``(n_gauss, 3, 12)`` array of bending ``B`` matrices.
    Hb :
        ``(3, 3)`` bending constitutive matrix.
    Bs_stack :
        Shape ``(n_gauss, 2, 12)`` array of shear ``B`` matrices.
    Hs :
        ``(2, 2)`` shear constitutive matrix.
    det_jacobians :
        Length ``n_gauss`` array of Jacobian determinants at the integration
        points.

    Returns
    -------
    numpy.ndarray
        The unexpanded bending/shear stiffness matrix in local coordinates.
    """

    n_points = det_jacobians.shape[0]
    k = np.zeros((12, 12), dtype=FLOAT)

    for gp in range(n_points):
        detJ = det_jacobians[gp]

        Bb = Bb_stack[gp]
        k += (Bb.T @ (Hb @ Bb)) * detJ

        Bs = Bs_stack[gp]
        k += (Bs.T @ (Hs @ Bs)) * detJ

    return k


MEMBRANE_MAP = np.array([0, 1, 6, 7, 12, 13, 18, 19], dtype=np.int64)
BENDING_MAP = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22], dtype=np.int64)


@njit(cache=True)
def _expand_matrix_24(
    k_unexpanded: NDArray[FLOAT], mapping: NDArray[np.int64]
) -> NDArray[FLOAT]:
    """
    Expand a reduced stiffness matrix into a 24×24 matrix using a lookup map.

    Parameters
    ----------
    k_unexpanded :
        The compact stiffness matrix (membrane or bending/shear).
    mapping :
        Index map that relates the compact degrees of freedom to the 24-element
        ordering used by frame elements.

    Returns
    -------
    numpy.ndarray
        The expanded 24×24 stiffness matrix.
    """

    result = np.zeros((24, 24), dtype=FLOAT)
    size = mapping.size
    for i in range(size):
        row = mapping[i]
        for j in range(size):
            col = mapping[j]
            result[row, col] = k_unexpanded[i, j]
    return result


def expand_membrane_matrix(k_unexpanded: NDArray[FLOAT]) -> NDArray[FLOAT]:
    """
    Expand an 8×8 membrane stiffness matrix to the 24×24 element format.

    Parameters
    ----------
    k_unexpanded :
        Compact membrane stiffness matrix.

    Returns
    -------
    numpy.ndarray
        Expanded 24×24 stiffness matrix with translational degrees of freedom.
    """

    return _expand_matrix_24(k_unexpanded, MEMBRANE_MAP)


def expand_bending_matrix(k_unexpanded: NDArray[FLOAT]) -> NDArray[FLOAT]:
    """
    Expand a 12×12 bending/shear stiffness matrix to the 24×24 element format.

    Parameters
    ----------
    k_unexpanded :
        Compact bending/shear stiffness matrix.

    Returns
    -------
    numpy.ndarray
        Expanded 24×24 stiffness matrix including rotational degrees of freedom.
    """

    return _expand_matrix_24(k_unexpanded, BENDING_MAP)


@njit(cache=True, fastmath=True)
def evaluate_polynomial(
    coeffs: NDArray[FLOAT], x_values: NDArray[FLOAT]
) -> NDArray[FLOAT]:
    """
    Evaluates a polynomial defined by ``coeffs`` (ascending order) at the provided ``x_values``.
    """

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


@njit(cache=True, fastmath=True)
def compute_ring_trig(num_divisions: int) -> Tuple[NDArray[FLOAT], NDArray[FLOAT]]:
    """
    Pre-compute cosine and sine values for equally spaced points on a unit circle.

    Parameters
    ----------
    num_divisions :
        Number of equally spaced angles to evaluate.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        Cosine and sine arrays (length ``num_divisions``) suitable for reuse
        when generating annular meshes.
    """

    cos_vals = np.empty(num_divisions, dtype=FLOAT)
    sin_vals = np.empty(num_divisions, dtype=FLOAT)

    theta = 2.0 * np.pi / num_divisions
    for i in range(num_divisions):
        angle = theta * i
        cos_vals[i] = np.cos(angle)
        sin_vals[i] = np.sin(angle)

    return cos_vals, sin_vals


@njit(cache=True, fastmath=True, inline="always")
def _quad_moment_point(
    d: NDArray[FLOAT],
    Hb: NDArray[FLOAT],
    Bb_stack: NDArray[FLOAT],
    xi: float,
    eta: float,
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


@njit(cache=True, fastmath=True)
def quad_moment_at(
    d: NDArray[FLOAT],
    Hb: NDArray[FLOAT],
    Bb_stack: NDArray[FLOAT],
    xi: float,
    eta: float,
) -> NDArray[FLOAT]:
    """
    Evaluate bending moments (Mx, My, Mxy) for a single point in a DKMQ quad element.
    """

    m0, m1, m2 = _quad_moment_point(d, Hb, Bb_stack, xi, eta)
    result = np.empty(3, dtype=FLOAT)
    result[0] = m0
    result[1] = m1
    result[2] = m2
    return result


@njit(cache=True, fastmath=True)
def quad_moment_batch(
    d: NDArray[FLOAT],
    Hb: NDArray[FLOAT],
    Bb_stack: NDArray[FLOAT],
    xi_vals: NDArray[FLOAT],
    eta_vals: NDArray[FLOAT],
) -> NDArray[FLOAT]:
    """
    Batched evaluation of bending moments for multiple natural coordinate input points.

    Parameters
    ----------
    xi_vals, eta_vals :
        Flattened arrays (same length) containing the coordinates for each evaluation point.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, 3)`` storing ``[Mx, My, Mxy]`` per point.
    """

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


@njit(cache=True, fastmath=True, inline="always")
def _quad_membrane_point(
    d: NDArray[FLOAT],
    Cm: NDArray[FLOAT],
    Bm_stack: NDArray[FLOAT],
    xi: float,
    eta: float,
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


@njit(cache=True, fastmath=True)
def quad_membrane_at(
    d: NDArray[FLOAT],
    Cm: NDArray[FLOAT],
    Bm_stack: NDArray[FLOAT],
    xi: float,
    eta: float,
) -> NDArray[FLOAT]:
    """
    Evaluate membrane stresses at a single point in a DKMQ quad element.
    """

    s0, s1, s2 = _quad_membrane_point(d, Cm, Bm_stack, xi, eta)
    result = np.empty(3, dtype=FLOAT)
    result[0] = s0
    result[1] = s1
    result[2] = s2
    return result


@njit(cache=True, fastmath=True)
def quad_membrane_batch(
    d: NDArray[FLOAT],
    Cm: NDArray[FLOAT],
    Bm_stack: NDArray[FLOAT],
    xi_vals: NDArray[FLOAT],
    eta_vals: NDArray[FLOAT],
) -> NDArray[FLOAT]:
    """
    Batched evaluation of membrane stresses for multiple natural coordinate input points.
    """

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


@njit(cache=True, fastmath=True, inline="always")
def cross_product_3d(a: NDArray[FLOAT], b: NDArray[FLOAT]) -> NDArray[FLOAT]:
    """Fast cross product for 3D vectors."""
    result = np.empty(3, dtype=FLOAT)
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]
    return result


@njit(cache=True, fastmath=True, inline="always")
def normalize_vector_3d(v: NDArray[FLOAT]) -> NDArray[FLOAT]:
    """Fast normalization for 3D vectors."""
    norm_val = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    result = np.empty(3, dtype=FLOAT)
    result[0] = v[0] / norm_val
    result[1] = v[1] / norm_val
    result[2] = v[2] / norm_val
    return result


@njit(cache=True, fastmath=True)
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
    """
    Compute local (x, y) coordinates for a quadrilateral element.

    Returns: (x1, y1, x2, y2, x3, y3, x4, y4)
    """
    # Vectors from node 1 to other nodes
    vector_12 = np.array([X2 - X1, Y2 - Y1, Z2 - Z1], dtype=FLOAT)
    vector_13 = np.array([X3 - X1, Y3 - Y1, Z3 - Z1], dtype=FLOAT)
    vector_14 = np.array([X4 - X1, Y4 - Y1, Z4 - Z1], dtype=FLOAT)

    # Define local axes
    x_axis = vector_12
    z_axis = cross_product_3d(x_axis, vector_13)
    y_axis = cross_product_3d(z_axis, x_axis)

    # Normalize
    x_axis = normalize_vector_3d(x_axis)
    y_axis = normalize_vector_3d(y_axis)

    # Project onto local axes
    x1 = 0.0
    y1 = 0.0
    x2 = vector_12[0] * x_axis[0] + vector_12[1] * x_axis[1] + vector_12[2] * x_axis[2]
    y2 = vector_12[0] * y_axis[0] + vector_12[1] * y_axis[1] + vector_12[2] * y_axis[2]
    x3 = vector_13[0] * x_axis[0] + vector_13[1] * x_axis[1] + vector_13[2] * x_axis[2]
    y3 = vector_13[0] * y_axis[0] + vector_13[1] * y_axis[1] + vector_13[2] * y_axis[2]
    x4 = vector_14[0] * x_axis[0] + vector_14[1] * x_axis[1] + vector_14[2] * x_axis[2]
    y4 = vector_14[0] * y_axis[0] + vector_14[1] * y_axis[1] + vector_14[2] * y_axis[2]

    return (x1, y1, x2, y2, x3, y3, x4, y4)


@njit(cache=True, fastmath=True)
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
) -> NDArray[FLOAT]:
    """
    Compute the 24x24 transformation matrix for a quadrilateral element.

    Uses i, j, and n nodes to establish local coordinate system.
    """
    # Local x-axis from i to j
    x_vec = np.array([xj - xi, yj - yi, zj - zi], dtype=FLOAT)
    x_axis = normalize_vector_3d(x_vec)

    # Vector in the element plane
    xy_vec = np.array([xn - xi, yn - yi, zn - zi], dtype=FLOAT)

    # Local z-axis perpendicular to element
    z_axis = cross_product_3d(x_axis, xy_vec)
    z_axis = normalize_vector_3d(z_axis)

    # Local y-axis
    y_axis = cross_product_3d(z_axis, x_axis)
    y_axis = normalize_vector_3d(y_axis)

    # Build transformation matrix
    T = np.zeros((24, 24), dtype=FLOAT)

    # Fill 3x3 direction cosine blocks along diagonal
    for i in range(8):
        offset = i * 3
        for row in range(3):
            for col in range(3):
                if row == 0:
                    T[offset + row, offset + col] = x_axis[col]
                elif row == 1:
                    T[offset + row, offset + col] = y_axis[col]
                else:
                    T[offset + row, offset + col] = z_axis[col]

    return T


@njit(cache=True, fastmath=True)
def compute_member_transformation_matrix(
    Xi: float,
    Yi: float,
    Zi: float,
    Xj: float,
    Yj: float,
    Zj: float,
    L: float,
    rotation: float,
) -> NDArray[FLOAT]:
    """
    Compute the 12x12 transformation matrix for a frame member.

    Parameters
    ----------
    Xi, Yi, Zi : float
        Coordinates of the i-node
    Xj, Yj, Zj : float
        Coordinates of the j-node
    L : float
        Member length
    rotation : float
        Rotation angle in radians

    Returns
    -------
    NDArray[FLOAT]
        12x12 transformation matrix
    """
    # Direction cosines for local x-axis
    x = np.array([(Xj - Xi) / L, (Yj - Yi) / L, (Zj - Zi) / L], dtype=FLOAT)

    # Tolerance for isclose comparisons
    tol = 1e-9

    # Vertical members (parallel to Y axis)
    if abs(Xi - Xj) < tol and abs(Zi - Zj) < tol:
        # When the member is vertical the projection is zero, so define y/z axes manually
        if Yj > Yi:
            y = np.array([-1.0, 0.0, 0.0], dtype=FLOAT)
            z = np.array([0.0, 0.0, 1.0], dtype=FLOAT)
        else:
            y = np.array([1.0, 0.0, 0.0], dtype=FLOAT)
            z = np.array([0.0, 0.0, 1.0], dtype=FLOAT)

    # Horizontal members (parallel to XZ plane)
    elif abs(Yi - Yj) < tol:
        # Horizontal members can use the global Y axis as a starting reference
        y_temp = np.array([0.0, 1.0, 0.0], dtype=FLOAT)
        z = cross_product_3d(x, y_temp)
        z = normalize_vector_3d(z)
        y = cross_product_3d(z, x)
        y = normalize_vector_3d(y)

    # General members
    else:
        # Projection on XZ plane
        proj = np.array([Xj - Xi, 0.0, Zj - Zi], dtype=FLOAT)

        if Yj > Yi:
            # Choose a cross-product order that keeps the local axes right-handed
            z = cross_product_3d(proj, x)
        else:
            z = cross_product_3d(x, proj)

        z = normalize_vector_3d(z)
        y = cross_product_3d(z, x)
        y = normalize_vector_3d(y)

    # Apply rotation if needed
    if abs(rotation) > 1e-10:
        # Apply the user-specified roll about the member's local x-axis using Rodrigues' formula
        c = np.cos(rotation)
        s = np.sin(rotation)

        # Rodrigues rotation formula
        u = x
        v = y
        u_dot_v = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
        u_cross_v = cross_product_3d(u, v)

        y = v * c + u_cross_v * s + u * u_dot_v * (1.0 - c)
        y = normalize_vector_3d(y)

        z = cross_product_3d(x, y)
        z = normalize_vector_3d(z)

    # Construct the transformation matrix
    T = np.zeros((12, 12), dtype=FLOAT)

    # Fill in 3x3 direction cosine matrices along the diagonal (4 blocks)
    for i in range(4):
        offset = i * 3
        # Each 3x3 block maps a translational or rotational set of DOF into the global frame
        # Row 0 (x-direction)
        T[offset + 0, offset + 0] = x[0]
        T[offset + 0, offset + 1] = x[1]
        T[offset + 0, offset + 2] = x[2]
        # Row 1 (y-direction)
        T[offset + 1, offset + 0] = y[0]
        T[offset + 1, offset + 1] = y[1]
        T[offset + 1, offset + 2] = y[2]
        # Row 2 (z-direction)
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
    "accumulate_membrane_stiffness",
    "accumulate_bending_shear_stiffness",
    "expand_membrane_matrix",
    "expand_bending_matrix",
    "evaluate_polynomial",
    "compute_ring_trig",
    "quad_moment_at",
    "quad_moment_batch",
    "quad_membrane_at",
    "quad_membrane_batch",
    "cross_product_3d",
    "normalize_vector_3d",
    "compute_quad_local_coords",
    "compute_quad_transformation_matrix",
    "compute_member_transformation_matrix",
]
