# References used to derive this element:
# 1. "A Comparative Formulation of DKMQ, DSQ and MITC4 Quadrilateral Plate Elements with New Numerical Results Based on s-norm Tests", Irwan Katili, 
# 2. "Finite Element Procedures, 2nd Edition", Klaus-Jurgen Bathe
# 3. "A First Course in the Finite Element Method, 4th Edition", Daryl L. Logan
# 4. "Finite Element Analysis Fundamentals", Richard H. Gallagher

from __future__ import annotations  # Allows more recent type hints features
from typing import TYPE_CHECKING, Literal, Dict, Tuple, ClassVar

from math import sin, cos, sqrt
import numpy as np
from numpy import add
from numpy.linalg import inv, norm
import warnings

from Pynite.quad_assembly import (
    accumulate_membrane_stiffness,
    accumulate_bending_shear_stiffness,
    expand_membrane_matrix,
    expand_bending_matrix,
    compute_quad_local_coords,
    compute_quad_transformation_matrix,
)
from Pynite.cython import (
    quad_moment_at,
    quad_moment_batch,
    quad_membrane_at,
    quad_membrane_batch,
)

BENDING_SIGN_IDX = np.array([3, 9, 15, 21], dtype=np.int64)
BENDING_ORDER = np.array([2, 4, 3, 8, 10, 9, 14, 16, 15, 20, 22, 21], dtype=np.int64)
MEMBRANE_ORDER = np.array([0, 1, 6, 7, 12, 13, 18, 19], dtype=np.int64)

_GAUSS_COORD = 1.0 / sqrt(3.0)
_GAUSS_POINTS = (
    (-_GAUSS_COORD, -_GAUSS_COORD),
    (_GAUSS_COORD, -_GAUSS_COORD),
    (_GAUSS_COORD, _GAUSS_COORD),
    (-_GAUSS_COORD, _GAUSS_COORD),
)


def _hw_row(xi: float, eta: float) -> np.ndarray:
    """Precompute the shear load interpolation row for a given Gauss point."""
    return 0.25 * np.array([
        (1 - xi) * (1 - eta), 0.0, 0.0,
        (1 + xi) * (1 - eta), 0.0, 0.0,
        (1 + xi) * (1 + eta), 0.0, 0.0,
        (1 - xi) * (1 + eta), 0.0, 0.0,
    ], dtype=np.float64)


_HW_GAUSS = np.stack([_hw_row(xi, eta) for xi, eta in _GAUSS_POINTS])
_FER_EXPANSION_MAP = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22], dtype=np.int64)

if TYPE_CHECKING:
    from typing import List, Tuple, Optional
    from numpy import float64
    from numpy.typing import NDArray
    from Pynite.FEModel3D import FEModel3D
    from Pynite.Node3D import Node3D

class Quad3D():
    """
    An isoparametric general quadrilateral element, formulated by superimposing an isoparametric DKMQ bending element with an isoparametric plane stress element. Drilling stability is provided by adding a weak rotational spring stiffness at each node. Isotropic behavior is the default, but orthotropic in-plane behavior can be modeled by specifying stiffness modification factors for the element's local x and y axes.

    This element performs well for thick and thin plates, and for skewed plates. Minor errors are introduced into the solution due to the drilling approximation. Orthotropic behavior is limited to acting along the plate's local axes.
    """

    # Reuse stiffness assembly results when multiple quads share geometry/material
    _GLOBAL_STIFFNESS_CACHE: ClassVar[Dict[Tuple[float, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}

    def __init__(self, name: str, i_node: Node3D, j_node: Node3D, m_node: Node3D, n_node: Node3D, 
                 t: float, material_name: str, model: FEModel3D, kx_mod: float = 1.0,
                 ky_mod: float = 1.0):

        self.name: str = name
        self.ID: Optional[int] = None
        self.type: str = 'Quad'

        self.i_node: Node3D = i_node
        self.j_node: Node3D = j_node
        self.m_node: Node3D = m_node
        self.n_node: Node3D = n_node

        self.t: float = t
        self.kx_mod: float = kx_mod
        self.ky_mod: float = ky_mod

        self.pressures: List[Tuple[float, str]] = []  # A list of surface pressures [pressure, case='Case 1']

        # Quads need a link to the model they belong to
        self.model: FEModel3D = model

        # Get material properties for the plate from the model
        try:
            self.E: float = self.model.materials[material_name].E
            self.nu: float = self.model.materials[material_name].nu
        except:
            raise KeyError('Please define the material ' + str(material_name) + ' before assigning it to plates.')

        # Cached data for accelerated post-processing (populated when stiffness matrices are assembled)
        self._Bb_stack = None
        self._B_m_stack = None
        self._Hb_cache = None
        self._Cm_cache = None
        self._k_cache: Optional[np.ndarray] = None
        self._K_global_cache: Optional[np.ndarray] = None
        self._k_b_cache: Optional[np.ndarray] = None
        self._k_m_cache: Optional[np.ndarray] = None
        self._fer_local_cache: dict[str, np.ndarray] = {}
        self._fer_cache: dict[str, np.ndarray] = {}
        self._T_cache: Optional[np.ndarray] = None
        self._J_cache: Dict[Tuple[float, float], np.ndarray] = {}
        self._L_cache: Dict[int, float] = {}
        self._dir_cos_cache: Dict[int, Tuple[float, float]] = {}
        self._local_coords_valid: bool = False
        self._gauss_jacobians_cache: Optional[Tuple] = None
        self.x1: float = 0.0
        self.y1: float = 0.0
        self.x2: float = 0.0
        self.y2: float = 0.0
        self.x3: float = 0.0
        self.y3: float = 0.0
        self.x4: float = 0.0
        self.y4: float = 0.0

    def invalidate_cache(self) -> None:
        """
        Clears cached stiffness, transformation, and fixed-end reaction data.
        Call this whenever element properties, loads, or orientation change.
        """

        self._Bb_stack = None
        self._B_m_stack = None
        self._Hb_cache = None
        self._Cm_cache = None
        self._k_cache = None
        self._K_global_cache = None
        self._k_b_cache = None
        self._k_m_cache = None
        self._fer_local_cache.clear()
        self._fer_cache.clear()
        self._T_cache = None
        self._J_cache.clear()
        self._L_cache.clear()
        self._dir_cos_cache.clear()
        self._local_coords_valid = False
        self._gauss_jacobians_cache = None

    # def _local_coords(self):
    #     """
    #     Calculates or recalculates and stores the local (x, y) coordinates for each node of the
    #     quadrilateral.
    #     """

    #     # Get the global coordinates for each node
    #     X1, Y1, Z1 = self.i_node.X, self.i_node.Y, self.i_node.Z
    #     X2, Y2, Z2 = self.j_node.X, self.j_node.Y, self.j_node.Z
    #     X3, Y3, Z3 = self.m_node.X, self.m_node.Y, self.m_node.Z
    #     X4, Y4, Z4 = self.n_node.X, self.n_node.Y, self.n_node.Z

    #     # Node 1 will be used as the origin of the plate's local (x, y) coordinate system. Find the
    #     # vector from the origin to each node.
    #     Xi1, Yi1, Zi1 = (X1 + X4)/2, (Y1 + Y4)/2, (Z1 + Z4)/2
    #     Xi2, Yi2, Zi2 = (X2 + X3)/2, (Y2 + Y3)/2, (Z2 + Z3)/2
    #     Xo, Yo, Zo = (Xi1 + Xi2)/2, (Yi1 + Yi2)/2, (Zi1 + Zi2)/2

    #     x_axis = np.array([Xi2 - Xi1, Yi2 - Yi1, Zi2 - Zi1]).T

    #     vector_01 = np.array([X1 - Xo, Y1 - Yo, Z1 - Zo]).T
    #     vector_02 = np.array([X2 - Xo, Y2 - Yo, Z2 - Zo]).T
    #     vector_03 = np.array([X3 - Xo, Y3 - Yo, Z3 - Zo]).T
    #     vector_04 = np.array([X4 - Xo, Y4 - Yo, Z4 - Zo]).T

    #     # Define the plate's local y, and z axes
    #     vector_x3 = np.array([X3 - Xi1, Y3 - Yi1, Z3 - Zi1]).T
    #     z_axis = np.cross(x_axis, vector_x3)
    #     y_axis = np.cross(z_axis, x_axis)

    #     # Convert the x, y and z axes into unit vectors
    #     x_axis = x_axis/norm(x_axis)
    #     y_axis = y_axis/norm(y_axis)
    #     z_axis = z_axis/norm(z_axis)

    #     # Calculate the local (x, y) coordinates for each node
    #     self.x1 = np.dot(vector_01, x_axis)
    #     self.x2 = np.dot(vector_02, x_axis)
    #     self.x3 = np.dot(vector_03, x_axis)
    #     self.x4 = np.dot(vector_04, x_axis)
    #     self.y1 = np.dot(vector_01, y_axis)
    #     self.y2 = np.dot(vector_02, y_axis)
    #     self.y3 = np.dot(vector_03, y_axis)
    #     self.y4 = np.dot(vector_04, y_axis)

    def _local_coords(self):
        """
        Calculates or recalculates and stores the local (x, y) coordinates for each node of the quadrilateral.
        """

        if self._local_coords_valid:
            return

        # Get the global coordinates for each node
        X1, Y1, Z1 = self.i_node.X, self.i_node.Y, self.i_node.Z
        X2, Y2, Z2 = self.j_node.X, self.j_node.Y, self.j_node.Z
        X3, Y3, Z3 = self.m_node.X, self.m_node.Y, self.m_node.Z
        X4, Y4, Z4 = self.n_node.X, self.n_node.Y, self.n_node.Z

        # Use optimized Cython function
        self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4 = compute_quad_local_coords(
            X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4
        )

        self._J_cache.clear()
        self._L_cache.clear()
        self._dir_cos_cache.clear()
        self._local_coords_valid = True
        self._gauss_jacobians_cache = None

    def _cache_key(self) -> Tuple[float, ...]:
        self._local_coords()
        return (
            round(self.x1, 8),
            round(self.y1, 8),
            round(self.x2, 8),
            round(self.y2, 8),
            round(self.x3, 8),
            round(self.y3, 8),
            round(self.x4, 8),
            round(self.y4, 8),
            round(self.t, 8),
            round(self.E, 8),
            round(self.nu, 8),
            round(self.kx_mod, 8),
            round(self.ky_mod, 8),
        )

    def L_k(self, k: Literal[5, 6, 7, 8]) -> float:

        cached = self._L_cache.get(k)
        if cached is not None:
            return cached

        # Figures 3 and 5
        if k == 5:
            length = ((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)**0.5
        elif k == 6:
            length = ((self.x3 - self.x2)**2 + (self.y3 - self.y2)**2)**0.5
        elif k == 7:
            length = ((self.x4 - self.x3)**2 + (self.y4 - self.y3)**2)**0.5
        elif k == 8:
            length = ((self.x1 - self.x4)**2 + (self.y1 - self.y4)**2)**0.5
        else:
            raise Exception('Invalid value for k. k must be 5, 6, 7, or 8.')

        self._L_cache[k] = length
        return length
    
    def dir_cos(self, k: Literal[5, 6, 7, 8]) -> Tuple[float, float]:

        cached = self._dir_cos_cache.get(k)
        if cached is not None:
            return cached

        L_k = self.L_k(k)

        # Figures 3 and 5
        if k == 5:
            C = (self.x2 - self.x1)/L_k
            S = (self.y2 - self.y1)/L_k
        elif k == 6:
            C = (self.x3 - self.x2)/L_k
            S = (self.y3 - self.y2)/L_k
        elif k == 7:
            C = (self.x4 - self.x3)/L_k
            S = (self.y4 - self.y3)/L_k
        elif k == 8:
            C = (self.x1 - self.x4)/L_k
            S = (self.y1 - self.y4)/L_k
        else:
            raise Exception('Invalid value for k. k must be 5, 6, 7, or 8.')

        result = (C, S)
        self._dir_cos_cache[k] = result
        return result

    def phi_k(self, k: Literal[5, 6, 7, 8]) -> float:

        kappa = 5/6

        # Equation 74
        return 2/(kappa*(1-self.nu))*(self.t/self.L_k(k))**2

    def N_i(self, i: Literal[1, 2, 3, 4], xi: float, eta: float) -> float:
        """
        Returns the interpolation function for any given coordinate in the natural (xi, eta) coordinate system
        """

        if i == 1:
            return 1/4*(1 - xi)*(1 - eta)
        elif i == 2:
            return 1/4*(1 + xi)*(1 - eta)
        elif i == 3:
            return 1/4*(1 + xi)*(1 + eta)
        elif i == 4:
            return 1/4*(1 - xi)*(1 + eta)
        else:
            raise Exception('Unable to calculate interpolation function. Invalid value specifed for i.')
    
    def P_k(self, k: Literal[5, 6, 7, 8], xi: float, eta: float) -> float:

        if k == 5:
            return 1/2*(1 - xi**2)*(1 - eta)
        elif k == 6:
            return 1/2*(1 + xi)*(1 - eta**2)
        elif k == 7:
            return 1/2*(1 - xi**2)*(1 + eta)
        elif k == 8:
            return 1/2*(1 - xi)*(1 - eta**2)
        else:
            raise Exception('Unable to calculate shape function. Invalid value specified for k.')
    
    def Co(self, xi: float, eta: float) -> NDArray[float64]:
        """
        This alternate calculation of the Jacobian matrix follows "The development of DKMQ plate bending element for thick to thin shell analysis based on the Naghdi/Reissner/Mindlin shell theory" by Katili, Batoz, Maknun and Hamdouni (2015). In the reference "C^o" is used instead of "J" to refer to the Jacobian. This method does not seem to produce incorrect results, but will be kept for future reference. It is helpful for understanding this plate element and may prove a useful simplification to the code base if implemented correctly someday.
        """

        # Nodal global coordinates (i=1, j=2, m=3, n=4)
        x_1 = np.array([[self.i_node.X, self.i_node.Y, self.i_node.Z]])
        x_2 = np.array([[self.j_node.X, self.j_node.Y, self.j_node.Z]])
        x_3 = np.array([[self.m_node.X, self.m_node.Y, self.m_node.Z]])
        x_4 = np.array([[self.n_node.X, self.n_node.Y, self.n_node.Z]])

        # Derivatives of the bilinear interpolation functions
        N1_xi = 0.25*(eta - 1)
        N2_xi = -0.25*(eta - 1)
        N3_xi = 0.25*(eta + 1)
        N4_xi = -0.25*(eta + 1)
        N1_eta = 0.25*(xi - 1)
        N2_eta = -0.25*(xi + 1)
        N3_eta = 0.25*(xi + 1)
        N4_eta = -0.25*(xi - 1)

        # Equation 4 - Katili 2015
        a_1 = N1_xi*x_1 + N2_xi*x_2 + N3_xi*x_3 + N4_xi*x_4
        a_2 = N1_eta*x_1 + N2_eta*x_2 + N3_eta*x_3 + N4_eta*x_4

        # Normal vector
        n = np.cross(a_1, a_2)/np.linalg.norm(np.cross(a_1, a_2))

        # Global unit vectors
        i = np.array([[1.0, 0.0, 0.0]])
        k = np.array([[0.0, 0.0, 1.0]])

        # Equation (13)
        if np.array_equal(n, k) or np.array_equal(n, -k):
            t_1 = i
        else:
            t_1 = np.cross(n, k)

        t_2 = np.cross(n, t_1)

        # Equation (7)
        a_11 = np.dot(a_1, a_1.T)[0, 0]
        a_12 = np.dot(a_1, a_2.T)[0, 0]
        a_21 = np.dot(a_2, a_1.T)[0, 0]
        a_22 = np.dot(a_2, a_2.T)[0, 0]

        # Matrix tensor of the middle surface
        a = np.array([[a_11, a_12],
                      [a_21, a_22]])
        
        a_det = np.linalg.det(a)

        # Contravariant vectors
        a1 = 1/a_det*(a_22*a_1 - a_12*a_2)
        a2 = 1/a_det*(-a_21*a_1 + a_11*a_2)

        Co = np.array([[np.dot(a1, t_1.T)[0, 0], np.dot(a1, t_2.T)[0, 0]],
                       [np.dot(a2, t_1.T)[0, 0], np.dot(a2, t_2.T)[0, 0]]])
        
        return Co

    def _gauss_jacobians(self) -> Tuple:
        """Pre-compute Jacobian data for all 4 Gauss points."""
        if self._gauss_jacobians_cache is not None:
            return self._gauss_jacobians_cache

        gp = 1.0 / sqrt(3.0)
        gauss_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]

        results = []
        for xi, eta in gauss_pts:
            J, invJ, detJ = self._jacobian_data(xi, eta)
            results.append((J, invJ, detJ))

        self._gauss_jacobians_cache = tuple(results)
        return self._gauss_jacobians_cache

    def _jacobian_data(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, float]:
        key = (round(xi, 12), round(eta, 12))
        cached = self._J_cache.get(key)
        if cached is not None:
            return cached

        # Get the local coordinates for the element
        x1, y1, x2, y2, x3, y3, x4, y4 = self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4

        J = 0.25 * np.array([
            [x1 * (eta - 1) - x2 * (eta - 1) + x3 * (eta + 1) - x4 * (eta + 1),
             y1 * (eta - 1) - y2 * (eta - 1) + y3 * (eta + 1) - y4 * (eta + 1)],
            [x1 * (xi - 1) - x2 * (xi + 1) + x3 * (xi + 1) - x4 * (xi - 1),
             y1 * (xi - 1) - y2 * (xi + 1) + y3 * (xi + 1) - y4 * (xi - 1)],
        ])

        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(detJ) <= 1e-12:
            raise ValueError(f'Jacobian determinant is zero for quad element {self.name} at ({xi}, {eta}).')

        invJ = np.array([[J[1, 1], -J[0, 1]],
                         [-J[1, 0], J[0, 0]]], dtype=J.dtype) / detJ

        data = (J, invJ, detJ)
        self._J_cache[key] = data
        return data

    def J(self, xi: float, eta: float) -> NDArray[float64]:
        """
        Returns the Jacobian matrix for the element
        """
        return self._jacobian_data(xi, eta)[0]

    def J_inv(self, xi: float, eta: float) -> NDArray[float64]:
        """
        Returns the inverse Jacobian matrix for the element
        """
        return self._jacobian_data(xi, eta)[1]

    def J_det(self, xi: float, eta: float) -> float:
        """
        Returns the determinant of the Jacobian matrix for the element
        """
        return self._jacobian_data(xi, eta)[2]

    def N_gamma(self, xi: float, eta: float) -> NDArray[float64]:

        # Equation 44
        return np.array([[1/2*(1 - eta),       0,      1/2*(1 + eta),       0     ],
                         [     0,        1/2*(1 + xi),       0,       1/2*(1 - xi)]])

    def A_gamma(self) -> NDArray[float64]:

        L5 = self.L_k(5)
        L6 = self.L_k(6)
        L7 = self.L_k(7)
        L8 = self.L_k(8)

        # Equation 46
        return np.array([[L5/2,   0,     0,     0 ],
                         [  0,  L6/2,    0,     0 ],
                         [  0,    0,  -L7/2,    0 ],
                         [  0,    0,     0,  -L8/2]])

    def A_u(self) -> NDArray[float64]:

        # Calculate the length of each side of the quad
        L5 = self.L_k(5)
        L6 = self.L_k(6)
        L7 = self.L_k(7)
        L8 = self.L_k(8)

        # Get the direction cosines for each side of the quad
        C5, S5 = self.dir_cos(5)
        C6, S6 = self.dir_cos(6)
        C7, S7 = self.dir_cos(7)
        C8, S8 = self.dir_cos(8)

        # Return the [A_u] matrix
        return 1/2*np.array([[-2/L5, C5, S5,  2/L5, C5, S5,   0,    0,  0,   0,    0,  0],
                             [  0,    0,  0, -2/L6, C6, S6, 2/L6,  C6, S6,   0,    0,  0],
                             [  0,    0,  0,   0,    0,  0, -2/L7, C7, S7,  2/L7, C7, S7],
                             [ 2/L8, C8, S8,   0,    0,  0,   0,    0,  0, -2/L8, C8, S8]])

    def A_Delta_inv_DKMQ(self) -> NDArray[float64]:

        phi5 = self.phi_k(5)
        phi6 = self.phi_k(6)
        phi7 = self.phi_k(7)
        phi8 = self.phi_k(8)

        return -3/2*np.array([[1/(1+phi5),     0,          0,          0     ],
                              [   0,       1/(1+phi6),     0,          0     ],
                              [   0,           0,      1/(1+phi7),     0     ],
                              [   0,           0,          0,      1/(1+phi8)]])

    def A_phi_Delta(self) -> NDArray[float64]:

        phi5 = self.phi_k(5)
        phi6 = self.phi_k(6)
        phi7 = self.phi_k(7)
        phi8 = self.phi_k(8)

        return np.array([[phi5/(1+phi5),       0,             0,             0      ],
                         [      0,       phi6/(1+phi6),       0,             0      ],
                         [      0,             0,       phi7/(1+phi7),       0      ],
                         [      0,             0,             0,       phi8/(1+phi8)]])
    
    def B_b_beta(self, xi: float, eta: float) -> NDArray[float64]:

        # Get the inverse of the Jacobian matrix
        J_inv = self.J_inv(xi, eta)

        # Get the individual terms for the Jacobian inverse
        j11 = J_inv[0, 0]
        j12 = J_inv[0, 1]
        j21 = J_inv[1, 0]
        j22 = J_inv[1, 1]

        # Derivatives of the bilinear interpolation functions
        N1_xi = 0.25*(eta - 1)
        N2_xi = -0.25*(eta - 1)
        N3_xi = 0.25*(eta + 1)
        N4_xi = -0.25*(eta + 1)
        N1_eta = 0.25*(xi - 1)
        N2_eta = -0.25*(xi + 1)
        N3_eta = 0.25*(xi + 1)
        N4_eta = -0.25*(xi - 1)

        N1x = j11*N1_xi + j12*N1_eta
        N1y = j21*N1_xi + j22*N1_eta
        N2x = j11*N2_xi + j12*N2_eta
        N2y = j21*N2_xi + j22*N2_eta
        N3x = j11*N3_xi + j12*N3_eta
        N3y = j21*N3_xi + j22*N3_eta
        N4x = j11*N4_xi + j12*N4_eta
        N4y = j21*N4_xi + j22*N4_eta

        return np.array([[0, N1x,  0,  0, N2x,  0,  0, N3x,  0,  0, N4x,  0 ],
                         [0,  0,  N1y, 0,  0,  N2y, 0,  0,  N3y, 0,  0,  N4y],
                         [0, N1y, N1x, 0, N2y, N2x, 0, N3y, N3x, 0, N4y, N4x]])
    
    def B_b_Delta_beta(self, xi: float, eta: float) -> NDArray[float64]:

        # Get the inverse of the Jacobian matrix
        J_inv = self.J_inv(xi, eta)

        # Get the individual terms for the Jacobian inverse
        j11 = J_inv[0, 0]
        j12 = J_inv[0, 1]
        j21 = J_inv[1, 0]
        j22 = J_inv[1, 1]

        # Derivatives of the quadratic interpolation functions
        P5_xi = xi*(eta - 1)
        P6_xi = -0.5*(eta - 1)*(eta + 1)
        P7_xi = -xi*(eta + 1)
        P8_xi = 0.5*(eta - 1)*(eta + 1)
        P5_eta = 0.5*(xi - 1)*(xi + 1)
        P6_eta = -eta*(xi + 1)
        P7_eta = -0.5*(xi - 1)*(xi + 1)
        P8_eta = eta*(xi - 1)

        P5x = j11*P5_xi + j12*P5_eta
        P5y = j21*P5_xi + j22*P5_eta
        P6x = j11*P6_xi + j12*P6_eta
        P6y = j21*P6_xi + j22*P6_eta
        P7x = j11*P7_xi + j12*P7_eta
        P7y = j21*P7_xi + j22*P7_eta
        P8x = j11*P8_xi + j12*P8_eta
        P8y = j21*P8_xi + j22*P8_eta

        C5, S5 = self.dir_cos(5)
        C6, S6 = self.dir_cos(6)
        C7, S7 = self.dir_cos(7)
        C8, S8 = self.dir_cos(8)

        return np.array([[    P5x*C5,          P6x*C6,          P7x*C7,          P8x*C8     ],
                         [    P5y*S5,          P6y*S6,          P7y*S7,          P8y*S8,    ],
                         [P5y*C5 + P5x*S5, P6y*C6 + P6x*S6, P7y*C7 + P7x*S7, P8y*C8 + P8x*S8]])
    
    def B_b(self, xi: float, eta: float) -> NDArray[float64]:
        """
        Returns the [B_b] matrix for bending
        """

        # Return the [B] matrix for bending
        return add(self.B_b_beta(xi, eta), self.B_b_Delta_beta(xi, eta) @ self.A_Delta_inv_DKMQ() @ self.A_u())

    def B_s(self, xi: float, eta: float) -> NDArray[float64]:
        """
        Returns the [B_s] matrix for shear
        """
        
        # Return the [B] matrix for shear
        return self.J_inv(xi, eta) @ self.N_gamma(xi, eta) @ self.A_gamma() @ self.A_phi_Delta() @ self.A_u()

    def B_s_gamma(self, xi:float , eta: float) -> None:
        """Returns the [B_s_gamma] matrix for shear (Equation 39 in Reference 1)

        :param xi: _description_
        :type xi: _type_
        :param eta: _description_
        :type eta: _type_
        """
        raise NotImplementedError('This function is not implemented yet. It is not needed for the current implementation of the Quad3D element.')

    def B_m(self, xi: float, eta: float) -> NDArray[float64]:

        # Differentiate the interpolation functions
        # Row 1 = interpolation functions differentiated with respect to x
        # Row 2 = interpolation functions differentiated with respect to y
        # Note that the inverse of the Jacobian converts from derivatives with
        # respect to xi and eta to derivatives with respect to x and y
        dH = np.matmul(self.J_inv(xi, eta), 1/4*np.array([[eta - 1, -eta + 1, eta + 1, -eta - 1],                 
                                                           [xi - 1,  -xi - 1,  xi + 1,  -xi + 1 ]]))

        # Reference 2, Example 5.5 (page 353)
        B_m = np.array([[dH[0, 0],    0,     dH[0, 1],    0,     dH[0, 2],    0,     dH[0, 3],    0    ],
                        [   0,     dH[1, 0],    0,     dH[1, 1],    0,     dH[1, 2],    0,     dH[1, 3]],
                        [dH[1, 0], dH[0, 0], dH[1, 1], dH[0, 1], dH[1, 2], dH[0, 2], dH[1, 3], dH[0, 3]]])

        return B_m

    def Hb(self) -> NDArray[float64]:
        '''
        Returns the stress-strain matrix for plate bending.
        '''

        # Referemce 1, Table 4.3, page 194
        nu = self.nu
        E = self.E
        h = self.t

        Hb = E*h**3/(12*(1 - nu**2))*np.array([[1,  nu,      0    ],
                                               [nu, 1,       0    ],
                                               [0,  0,  (1 - nu)/2]])
        
        return Hb

    def Hs(self) -> NDArray[float64]:
        '''
        Returns the stress-strain matrix for shear.
        '''
        # Reference 2, Equations (5.97), page 422
        k = 5/6
        E = self.E
        h = self.t
        nu = self.nu

        Hs = E*h*k/(2*(1 + nu))*np.array([[1, 0],
                                          [0, 1]])

        return Hs

    def Cm(self) -> NDArray[float64]:
        """
        Returns the stress-strain matrix for an isotropic or orthotropic plane stress element
        """
        
        # Apply the stiffness modification factors for each direction to obtain orthotropic
        # behavior. Stiffness modification factors of 1.0 in each direction (the default) will
        # model isotropic behavior. Orthotropic behavior is limited to the element's local
        # coordinate system.
        Ex = self.E*self.kx_mod
        Ey = self.E*self.ky_mod
        nu_xy = self.nu
        nu_yx = self.nu

        # The shear modulus will be unafected by orthotropic behavior
        # Logan, Appendix C.3, page 750
        G = self.E/(2*(1 + self.nu))

        # Gallagher, Equation 9.3, page 251
        Cm = 1/(1 - nu_xy*nu_yx)*np.array([[   Ex,    nu_yx*Ex,           0         ],
                                           [nu_xy*Ey,    Ey,              0         ],
                                           [    0,        0,     (1 - nu_xy*nu_yx)*G]])

        return Cm

    def k_b(self) -> NDArray[float64]:
        '''
        Returns the local stiffness matrix for bending and shear stresses
        '''

        if self._k_b_cache is not None:
            return self._k_b_cache

        Hb = self.Hb()
        Hs = self.Hs()

        # Define the gauss point for numerical integration
        gp = 1/3**0.5

        dets = np.array([
            self.J_det(-gp, -gp),
            self.J_det(gp, -gp),
            self.J_det(gp, gp),
            self.J_det(-gp, gp)
        ])

        Bb_stack = np.array([
            self.B_b(-gp, -gp),
            self.B_b(gp, -gp),
            self.B_b(gp, gp),
            self.B_b(-gp, gp)
        ])

        Bs_stack = np.array([
            self.B_s(-gp, -gp),
            self.B_s(gp, -gp),
            self.B_s(gp, gp),
            self.B_s(-gp, gp)
        ])

        k_unexpanded = accumulate_bending_shear_stiffness(Bb_stack, Hb, Bs_stack, Hs, dets)

        self._Bb_stack = np.ascontiguousarray(Bb_stack)
        self._Hb_cache = np.ascontiguousarray(Hb)

        # Following Bathe's recommendation for the drilling degree of freedom
        # from Example 4.19 in "Finite Element Procedures, 2nd Ed.", calculate
        # the drilling stiffness as 1/1000 of the smallest diagonal term in
        # the element's stiffness matrix. This is not theoretically correct,
        # but it allows the model to solve without singularities, and should
        # have a minimal effect on the final solution. Bathe recommends 1/1000
        # as a value that is weak enough but not so small that it affect the
        # results. Bathe recommends looking at all the diagonals in the
        # combined bending plus membrane stiffness matrix. Some of those terms
        # relate to translational stiffness. It seems more rational to only
        # look at the terms relating to rotational stiffness. That will be
        # Pynite's approach.
        k_rz = min(abs(k_unexpanded[1, 1]), abs(k_unexpanded[2, 2]), abs(k_unexpanded[4, 4]), abs(k_unexpanded[5, 5]),
                   abs(k_unexpanded[7, 7]), abs(k_unexpanded[8, 8]), abs(k_unexpanded[10, 10]), abs(k_unexpanded[11, 11])
                   )/1000

        k_exp = expand_bending_matrix(k_unexpanded)

        # Add the drilling degree of freedom's weak spring
        k_exp[5, 5] = k_rz
        k_exp[11, 11] = k_rz
        k_exp[17, 17] = k_rz
        k_exp[23, 23] = k_rz

        # Invert the local +y bending sign convention to match Pynite's
        k_exp[[4, 10, 16, 22], :] *= -1
        k_exp[:, [4, 10, 16, 22]] *= -1

        # The way the DKMQ element was derived, the positions relating to x
        # and y in the element's stiffness matrix are swapped from Pynite's.
        # Swap them to match Pynite.
        k_exp[[3, 4, 9, 10, 15, 16, 21, 22], :] = k_exp[[4, 3, 10, 9, 16, 15, 22, 21], :]
        k_exp[:, [3, 4, 9, 10, 15, 16, 21, 22]] = k_exp[:, [4, 3, 10, 9, 16, 15, 22, 21]]

        self._k_b_cache = k_exp
        return k_exp

    def k_m(self) -> NDArray[float64]:
        '''
        Returns the local stiffness matrix for membrane (in-plane) stresses.

        Plane stress is assumed
        '''

        if self._k_m_cache is not None:
            return self._k_m_cache

        t = self.t
        Cm = self.Cm()

        # Define the gauss point for numerical integration
        gp = 1/3**0.5

        B_stack = np.array([
            self.B_m(-gp, -gp),
            self.B_m(gp, -gp),
            self.B_m(gp, gp),
            self.B_m(-gp, gp)
        ])

        dets = np.array([
            self.J_det(-gp, -gp),
            self.J_det(gp, -gp),
            self.J_det(gp, gp),
            self.J_det(-gp, gp)
        ])

        if np.any(dets <= 0):
            warnings.warn(f'The Jacobian matrix for quad element {self.name} is less than or equal to zero, indicating the element is invalid or badly distorted.')

        k_unexpanded = accumulate_membrane_stiffness(B_stack, Cm, dets, t)

        self._B_m_stack = np.ascontiguousarray(B_stack)
        self._Cm_cache = np.ascontiguousarray(Cm)
        self._k_m_cache = expand_membrane_matrix(k_unexpanded)
        return self._k_m_cache

    def k(self) -> NDArray[float64]:
        '''
        Returns the quad element's local stiffness matrix.
        '''

        if self._k_cache is not None:
            return self._k_cache

        key = self._cache_key()
        cached = Quad3D._GLOBAL_STIFFNESS_CACHE.get(key)
        if cached is not None:
            (k_b_cached, Bb_cached, Hb_cached, k_m_cached, Bm_cached, Cm_cached) = cached
            self._k_b_cache = k_b_cached.copy()
            self._Bb_stack = Bb_cached.copy() if Bb_cached is not None else None
            self._Hb_cache = Hb_cached.copy() if Hb_cached is not None else None
            self._k_m_cache = k_m_cached.copy()
            self._B_m_stack = Bm_cached.copy() if Bm_cached is not None else None
            self._Cm_cache = Cm_cached.copy() if Cm_cached is not None else None
            self._k_cache = np.add(self._k_b_cache, self._k_m_cache)
            return self._k_cache

        # Sum the bending and membrane stiffness matrices using freshly computed values
        k_b_local = self.k_b()
        k_m_local = self.k_m()
        self._k_cache = np.add(k_b_local, k_m_local)
        Quad3D._GLOBAL_STIFFNESS_CACHE[key] = (
            k_b_local.copy(),
            self._Bb_stack.copy() if self._Bb_stack is not None else None,
            self._Hb_cache.copy() if self._Hb_cache is not None else None,
            k_m_local.copy(),
            self._B_m_stack.copy() if self._B_m_stack is not None else None,
            self._Cm_cache.copy() if self._Cm_cache is not None else None,
        )
        return self._k_cache

    def f(self, combo_name: str='Combo 1') -> NDArray[float64]:
        """
        Returns the quad element's local end force vector
        """

        # Calculate and return the plate's local end force vector
        return np.add(self.k() @ self.d(combo_name), self.fer(combo_name))

    def fer(self, combo_name: str='Combo 1') -> NDArray[float64]:
        """
        Returns the quadrilateral's local fixed end reaction vector.

        Parameters
        ----------
        combo_name : string
            The name of the load combination to get the consistent load vector for.
        """

        cached = self._fer_local_cache.get(combo_name)
        if cached is not None:
            return cached

        # Update the local coordinate system
        self._local_coords()

        # Initialize the fixed end reaction vector
        fer_local = None

        # Get the requested load combination
        combo = self.model.load_combos[combo_name]

        # Initialize the element's surface pressure to zero
        p = 0.0

        # Loop through each load case and factor in the load combination
        for case, factor in combo.factors.items():

            # Sum the pressures
            for pressure in self.pressures:

                # Check if the current pressure corresponds to the current load case
                if pressure[1] == case:

                    # Sum the pressures
                    p -= factor*pressure[0]

        if abs(p) < 1e-16:
            fer_exp = np.zeros((24, 1))
            self._fer_local_cache[combo_name] = fer_exp
            return fer_exp

        dets = np.array([self.J_det(xi, eta) for xi, eta in _GAUSS_POINTS], dtype=np.float64)
        fer_local = (_HW_GAUSS * dets[:, None]).sum(axis=0) * p

        # Initialize the expanded vector to all zeros
        fer_exp = np.zeros(24, dtype=np.float64)
        fer_exp[_FER_EXPANSION_MAP] = fer_local
        fer_exp = fer_exp.reshape(24, 1)

        self._fer_local_cache[combo_name] = fer_exp
        return fer_exp

    def d(self, combo_name='Combo 1') -> NDArray[float64]:
       """
       Returns the quad element's local displacement vector
       """

       # Calculate and return the local displacement vector
       return self.T() @ self.D(combo_name)

    def F(self, combo_name: str='Combo 1') -> NDArray[float64]:
        """
        Returns the quad element's global force vector

        Parameters
        ----------
        combo_name : string
            The load combination to get results for.
        """

        # Calculate and return the global force vector
        return inv(self.T()) @ self.f(combo_name)

    def D(self, combo_name:str='Combo 1') -> NDArray[float64]:
        '''
        Returns the quad element's global displacement vector for the given
        load combination.

        Parameters
        ----------
        combo_name : string
            The name of the load combination to get the displacement vector
            for (not the load combination itself).
        '''
        
        # Initialize the displacement vector
        D = np.zeros((24, 1))
        
        # Read in the global displacements from the nodes
        D[0, 0] = self.i_node.DX[combo_name]
        D[1, 0] = self.i_node.DY[combo_name]
        D[2, 0] = self.i_node.DZ[combo_name]
        D[3, 0] = self.i_node.RX[combo_name]
        D[4, 0] = self.i_node.RY[combo_name]
        D[5, 0] = self.i_node.RZ[combo_name]

        D[6, 0] = self.j_node.DX[combo_name]
        D[7, 0] = self.j_node.DY[combo_name]
        D[8, 0] = self.j_node.DZ[combo_name]
        D[9, 0] = self.j_node.RX[combo_name]
        D[10, 0] = self.j_node.RY[combo_name]
        D[11, 0] = self.j_node.RZ[combo_name]

        D[12, 0] = self.m_node.DX[combo_name]
        D[13, 0] = self.m_node.DY[combo_name]
        D[14, 0] = self.m_node.DZ[combo_name]
        D[15, 0] = self.m_node.RX[combo_name]
        D[16, 0] = self.m_node.RY[combo_name]
        D[17, 0] = self.m_node.RZ[combo_name]

        D[18, 0] = self.n_node.DX[combo_name]
        D[19, 0] = self.n_node.DY[combo_name]
        D[20, 0] = self.n_node.DZ[combo_name]
        D[21, 0] = self.n_node.RX[combo_name]
        D[22, 0] = self.n_node.RY[combo_name]
        D[23, 0] = self.n_node.RZ[combo_name]

        # Return the global displacement vector
        return D

    def K(self) -> NDArray[float64]:
        '''
        Returns the quad element's global stiffness matrix
        '''

        if self._K_global_cache is not None:
            return self._K_global_cache

        # Get the transformation matrix
        T = self.T()

        # Calculate and return the stiffness matrix in global coordinates
        self._K_global_cache = T.T @ self.k() @ T
        return self._K_global_cache

    # Global fixed end reaction vector
    def FER(self, combo_name:str='Combo 1') -> NDArray[float64]:
        '''
        Returns the global fixed end reaction vector.

        Parameters
        ----------
        combo_name : string
            The name of the load combination to calculate the fixed end
            reaction vector for (not the load combination itself).
        '''
        cached = self._fer_cache.get(combo_name)
        if cached is not None:
            return cached

        result = self.T().T @ self.fer(combo_name)
        self._fer_cache[combo_name] = result
        return result
  
    def T(self) -> NDArray[float64]:
        """
        Returns the coordinate transformation matrix for the quad element.
        """

        if self._T_cache is not None:
            return self._T_cache

        # Use optimized Cython function
        self._T_cache = compute_quad_transformation_matrix(
            self.i_node.X, self.i_node.Y, self.i_node.Z,
            self.j_node.X, self.j_node.Y, self.j_node.Z,
            self.n_node.X, self.n_node.Y, self.n_node.Z,
        )

        return self._T_cache
    
    # def T(self):
    #     """
    #     Returns the coordinate transformation matrix for the quad element.
    #     """

    #     xi = self.i_node.X
    #     xj = self.j_node.X
    #     yi = self.i_node.Y
    #     yj = self.j_node.Y
    #     zi = self.i_node.Z
    #     zj = self.j_node.Z

    #     # Calculate the direction cosines for the local x-axis.The local x-axis will run from
    #     # the i-node to the j-node
    #     x = [xj - xi, yj - yi, zj - zi]

    #     # Divide the vector by its magnitude to produce a unit x-vector of
    #     # direction cosines
    #     mag = (x[0]**2 + x[1]**2 + x[2]**2)**0.5
    #     x = [x[0]/mag, x[1]/mag, x[2]/mag]
        
    #     # The local y-axis will be in the plane of the plate. Find a vector in
    #     # the plate's local xy plane.
    #     xn = self.n_node.X
    #     yn = self.n_node.Y
    #     zn = self.n_node.Z
    #     xy = [xn - xi, yn - yi, zn - zi]

    #     # Find a vector perpendicular to the plate surface to get the
    #     # orientation of the local z-axis.
    #     z = np.cross(x, xy)
        
    #     # Divide the z-vector by its magnitude to produce a unit z-vector of
    #     # direction cosines.
    #     mag = (z[0]**2 + z[1]**2 + z[2]**2)**0.5
    #     z = [z[0]/mag, z[1]/mag, z[2]/mag]

    #     # Calculate the local y-axis as a vector perpendicular to the local z
    #     # and x-axes.
    #     y = np.cross(z, x)
        
    #     # Divide the y-vector by its magnitude to produce a unit vector of
    #     # direction cosines.
    #     mag = (y[0]**2 + y[1]**2 + y[2]**2)**0.5
    #     y = [y[0]/mag, y[1]/mag, y[2]/mag]

    #     # Create the direction cosines matrix.
    #     dir_cos = np.array([x,
    #                         y,
    #                         z])
        
    #     # Build the transformation matrix.
    #     T = np.zeros((24, 24))
    #     T[0:3, 0:3] = dir_cos
    #     T[3:6, 3:6] = dir_cos
    #     T[6:9, 6:9] = dir_cos
    #     T[9:12, 9:12] = dir_cos
    #     T[12:15, 12:15] = dir_cos
    #     T[15:18, 15:18] = dir_cos
    #     T[18:21, 18:21] = dir_cos
    #     T[21:24, 21:24] = dir_cos
        
    #     # Return the transformation matrix.
    #     return T
    
    def shear(self, xi:float=0.0, eta:float=0.0, local:bool=True, combo_name:str='Combo 1') -> NDArray[float64]:
        """
        Returns the interal shears at any point in the quad element.

        Internal shears are reported as a 2D array [[Qx], [Qy]] at the
        specified location in the (xi, eta) natural coordinate system.

        Parameters
        ----------
        xi : float
            The xi-coordinate. Default is 0.
        eta : float
            The eta-coordinate. Default is 0.
        
        Returns
        -------
        Internal shear force per unit length of the quad element: [[Qx], [Qy]]
        """

        # Get the plate's local displacement vector
        d = self.d(combo_name)
                
        # Correct the sign convention for x-axis rotation - note that +x bending and +x rotation are opposite in the DKMQ derivation. Hence when correcting d we correct the x terms, but when correcting k we correct the y terms
        d[[3, 9, 15, 21], :] *= -1

        # Slice out terms not related to plate bending, and swap the local x and y to match the DKMQ derivation
        d = d[[2, 4, 3, 8, 10, 9, 14, 16, 15, 20, 22, 21], :]

        # Define the gauss point used for numerical integration
        gp = 1/3**0.5

        # Define extrapolated r and s points
        xi_ex = xi/gp
        eta_ex = eta/gp

        # Define the interpolation functions
        H = 1/4*np.array([(1 - xi_ex)*(1 - eta_ex), (1 + xi_ex)*(1 - eta_ex), (1 + xi_ex)*(1 + eta_ex), (1 - xi_ex)*(1 + eta_ex)])

        # Get the stress-strain matrix
        Hs = self.Hs()

        # Calculate the internal shears [Qx, Qy] at each gauss point
        q1 = np.matmul(Hs, np.matmul(self.B_s(-gp, -gp), d))
        q2 = np.matmul(Hs, np.matmul(self.B_s(gp, -gp), d))
        q3 = np.matmul(Hs, np.matmul(self.B_s(gp, gp), d))
        q4 = np.matmul(Hs, np.matmul(self.B_s(-gp, gp), d))

        # Extrapolate to get the value at the requested location
        Qx = H[0]*q1[0] + H[1]*q2[0] + H[2]*q3[0] + H[3]*q4[0]
        Qy = H[0]*q1[1] + H[1]*q2[1] + H[2]*q3[1] + H[3]*q4[1]

        if local:
            
            return np.array([Qx,
                             Qy])
        
        else:
            
            # Get the direction cosines for the plate's local coordinate system
            dir_cos = self.T()[:3, :3]

            # Transform the results to a global vector
            Qx = float(Qx)
            Qy = float(Qy)
            Q_global = np.matmul(dir_cos.T, np.array([[Qx, 0,  0],
                                                      [0,  Qy, 0],
                                                      [0,  0,  0]]))

            # Extract results acting along each global axis
            Qx_global = Q_global[0, 0]
            Qy_global = Q_global[1, 1]
            Qz_global = Q_global[2, 1]

            # Return the results as a 2D matrix
            return np.array([[Qx_global],
                             [Qy_global],
                             [Qz_global]])

    def moment(self, xi:float=0.0, eta:float=0.0, local:bool=True, combo_name:str='Combo 1') -> NDArray[float64]:
        """
        Returns the interal moments at any point in the quad element.

        Internal moments are reported as a 2D array [[Mx], [My], [Mxy]] at the
        specified location in the (xi, eta) natural coordinate system.

        Parameters
        ----------
        xi : float
            The xi-coordinate. Default is 0.
        eta : float
            The eta-coordinate. Default is 0.

        Returns
        -------
        Internal moment per unit length of the quad element: [[Mx], [My], [Mxy]]
        """

        d_full = np.asarray(self.d(combo_name), dtype=float).reshape(-1).copy()
        d_full[BENDING_SIGN_IDX] *= -1
        d_local = np.ascontiguousarray(d_full[BENDING_ORDER])

        if self._Bb_stack is None or self._Hb_cache is None:
            self.k_b()

        xi_arr = np.asarray(xi)
        eta_arr = np.asarray(eta)

        if xi_arr.ndim == 0 and eta_arr.ndim == 0:
            xi_scalar = float(xi_arr)
            eta_scalar = float(eta_arr)
            results = quad_moment_at(d_local, self._Hb_cache, self._Bb_stack, xi_scalar, eta_scalar)
            Mx = results[0]
            My = results[1]
            Mxy = results[2]

            if local:
                return np.array([[Mx],
                                 [My],
                                 [Mxy]])

            dir_cos = self.T()[:3, :3]
            M_local = np.array([
                [Mx, Mxy, 0.0],
                [Mxy, My, 0.0],
                [0.0, 0.0, 0.0]
            ])

            M_global_tensor = dir_cos @ M_local @ dir_cos.T

            return np.array([[M_global_tensor[0, 0]],
                             [M_global_tensor[1, 1]],
                             [M_global_tensor[0, 1]]])

        xi_b, eta_b = np.broadcast_arrays(xi_arr, eta_arr)
        xi_flat = xi_b.astype(float).ravel()
        eta_flat = eta_b.astype(float).ravel()

        batch = quad_moment_batch(d_local, self._Hb_cache, self._Bb_stack, xi_flat, eta_flat)
        Mx = batch[:, 0].reshape(xi_b.shape)
        My = batch[:, 1].reshape(xi_b.shape)
        Mxy = batch[:, 2].reshape(xi_b.shape)

        if local:
            return np.stack((Mx, My, Mxy), axis=0)

        dir_cos = self.T()[:3, :3]
        Mx_g = np.empty_like(Mx)
        My_g = np.empty_like(My)
        Mxy_g = np.empty_like(Mxy)

        flat_Mx = Mx.ravel()
        flat_My = My.ravel()
        flat_Mxy = Mxy.ravel()

        for idx in range(flat_Mx.size):
            M_local = np.array([
                [flat_Mx[idx], flat_Mxy[idx], 0.0],
                [flat_Mxy[idx], flat_My[idx], 0.0],
                [0.0, 0.0, 0.0]
            ])
            M_global_tensor = dir_cos @ M_local @ dir_cos.T
            Mx_g.ravel()[idx] = M_global_tensor[0, 0]
            My_g.ravel()[idx] = M_global_tensor[1, 1]
            Mxy_g.ravel()[idx] = M_global_tensor[0, 1]

        return np.stack((Mx_g, My_g, Mxy_g), axis=0)


    def membrane(self, xi:float=0, eta: float=0, local:bool=True, combo_name:str='Combo 1') -> NDArray[float64]:

        d_full = np.asarray(self.d(combo_name), dtype=float).reshape(-1)
        d_local = np.ascontiguousarray(d_full[MEMBRANE_ORDER])

        if self._B_m_stack is None or self._Cm_cache is None:
            self.k_m()

        xi_arr = np.asarray(xi)
        eta_arr = np.asarray(eta)

        if xi_arr.ndim == 0 and eta_arr.ndim == 0:
            xi_scalar = float(xi_arr)
            eta_scalar = float(eta_arr)
            stresses = quad_membrane_at(d_local, self._Cm_cache, self._B_m_stack, xi_scalar, eta_scalar)
            Sx = stresses[0]
            Sy = stresses[1]
            Txy = stresses[2]

            if local:
                return np.array([[Sx],
                                 [Sy],
                                 [Txy]])

            dir_cos = self.T()[:3, :3]
            S_local = np.array([
                [Sx, Txy, 0.0],
                [Txy, Sy, 0.0],
                [0.0, 0.0, 0.0]
            ])

            S_global_tensor = dir_cos @ S_local @ dir_cos.T

            return np.array([[S_global_tensor[0, 0]],
                             [S_global_tensor[1, 1]],
                             [S_global_tensor[0, 1]]])

        xi_b, eta_b = np.broadcast_arrays(xi_arr, eta_arr)
        xi_flat = xi_b.astype(float).ravel()
        eta_flat = eta_b.astype(float).ravel()

        batch = quad_membrane_batch(d_local, self._Cm_cache, self._B_m_stack, xi_flat, eta_flat)
        Sx = batch[:, 0].reshape(xi_b.shape)
        Sy = batch[:, 1].reshape(xi_b.shape)
        Txy = batch[:, 2].reshape(xi_b.shape)

        if local:
            return np.stack((Sx, Sy, Txy), axis=0)

        dir_cos = self.T()[:3, :3]
        Sx_g = np.empty_like(Sx)
        Sy_g = np.empty_like(Sy)
        Sxy_g = np.empty_like(Txy)

        flat_Sx = Sx.ravel()
        flat_Sy = Sy.ravel()
        flat_Txy = Txy.ravel()

        for idx in range(flat_Sx.size):
            S_local = np.array([
                [flat_Sx[idx], flat_Txy[idx], 0.0],
                [flat_Txy[idx], flat_Sy[idx], 0.0],
                [0.0, 0.0, 0.0]
            ])
            S_global_tensor = dir_cos @ S_local @ dir_cos.T
            Sx_g.ravel()[idx] = S_global_tensor[0, 0]
            Sy_g.ravel()[idx] = S_global_tensor[1, 1]
            Sxy_g.ravel()[idx] = S_global_tensor[0, 1]

        return np.stack((Sx_g, Sy_g, Sxy_g), axis=0)
