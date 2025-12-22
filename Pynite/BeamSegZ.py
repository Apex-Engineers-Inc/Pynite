# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:52:31 2017

@author: D. Craig Brinck, SE

Optimized for performance with __slots__ and vectorized numpy operations.
"""
from __future__ import annotations  # Allows more recent type hints features
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy import full, asarray, ndarray

if TYPE_CHECKING:
    from typing import Any, List
    from numpy.typing import NDArray


# %%
# A mathematically continuous beam segment
class BeamSegZ():
    """
    A mathematically continuous beam segment with vectorized operations.

    Properties
    ----------
    x1 : number
      The starting location of the segment relative to the start of the beam
    x2 : number
      The ending location of the segment relative to the start of the beam
    w1 : number
      The distributed load magnitude at the start of the segment
    w2 : number
      The distributed load magnitude at the end of the segment
    p1 : number
      The distributed axial load magnitude at the start of the segment
    p2 : number
      The distributed axial load magnitude at the end of the segment
    V1 : number
      The internal shear force at the start of the segment
    M1 : number
      The internal moment at the start of the segment
    P1 : number
      The internal axial force at the start of the segment
    T1 : number
      Torsional moment at start of segment
    theta1: number
      The slope (radians) at the start of the segment
    delta1: number
      The transverse displacement at the start of the segment
    delta_x1 : number
      The axial displacement at the start of the segment
    EI : number
      The flexural stiffness of the segment
    EA : number
      The axial stiffness of the segment

    Notes
    -----
    Any unit system may be used as long as the units are consistent with each other.
    All computational methods are vectorized to accept numpy arrays for x.
    """

    # Use __slots__ for memory efficiency and faster attribute access
    __slots__ = ('x1', 'x2', 'w1', 'w2', 'p1', 'p2', 'V1', 'M1', 'P1', 'T1',
                 'theta1', 'delta1', 'delta_x1', 'EI', 'EA', '_length')

    def __init__(self) -> None:
        """
        Constructor
        """
        self.x1: float = 0.0  # Start location of beam segment (relative to start of beam)
        self.x2: float = 0.0  # End location of beam segment (relative to start of beam)
        self.w1: float = 0.0  # Linear distributed transverse load at start of segment
        self.w2: float = 0.0  # Linear distributed transverse load at end of segment
        self.p1: float = 0.0  # Linear distributed axial load at start of segment
        self.p2: float = 0.0  # Linear distributed axial load at end of segment
        self.V1: float = 0.0  # Internal shear force at start of segment
        self.M1: float = 0.0  # Internal moment at start of segment
        self.P1: float = 0.0  # Internal axial force at start of segment
        self.T1: float = 0.0  # Torsional moment at start of segment
        self.theta1: float = 0.0  # Slope at start of beam segment
        self.delta1: float = 0.0  # Displacement at start of beam segment
        self.delta_x1: float = 0.0  # Axial displacement at start of beam segment
        self.EI: float = 0.0  # Flexural stiffness of the beam segment
        self.EA: float = 0.0  # Axial stiffness of the beam segment
        self._length: float = 0.0  # Cached length

    def Length(self) -> float:
        """
        Returns the length of the segment (cached for performance).
        """
        if self._length == 0.0:
            self._length = self.x2 - self.x1
        return self._length

    def Shear(self, x: Union[float, ndarray]) -> Union[float, ndarray]:
        """
        Returns the shear force at location(s) 'x' on the segment.
        Vectorized: accepts scalar or numpy array.

        Parameters
        ----------
        x : float or ndarray
            Location(s) relative to start of segment

        Returns
        -------
        float or ndarray
            Shear force(s) at the location(s)
        """
        V1 = self.V1
        w1 = self.w1
        w2 = self.w2
        L = self.Length()

        # V(x) = V1 + w1*x + x^2*(w2-w1)/(2L)
        return V1 + w1*x + x*x*(-w1 + w2)/(2*L)

    def moment(self, x: Union[float, ndarray], P_delta: bool = False) -> Union[float, ndarray]:
        """
        Returns the moment at location(s) on the segment.
        Vectorized: accepts scalar or numpy array.

        Parameters
        ----------
        x : float or ndarray
            Location(s) relative to start of segment
        P_delta : bool
            Include P-Delta effects

        Returns
        -------
        float or ndarray
            Moment(s) at the location(s)
        """
        V1 = self.V1
        M1 = self.M1
        w1 = self.w1
        w2 = self.w2
        L = self.Length()

        # M(x) = M1 - V1*x - w1*x^2/2 - x^3*(-w1+w2)/(6L)
        x2 = x * x
        x3 = x2 * x
        M = M1 - V1*x - w1*x2/2 - x3*(-w1 + w2)/(6*L)

        if P_delta:
            P1 = self.P1
            delta_1 = self.delta1
            delta_x = self.deflection(x, P_delta=False)  # Avoid recursion
            M = M + P1*(delta_x - delta_1)

        return M

    def axial(self, x: Union[float, ndarray]) -> Union[float, ndarray]:
        """
        Returns the axial force at location(s) on the segment.
        Vectorized: accepts scalar or numpy array.

        Parameters
        ----------
        x : float or ndarray
            Location(s) relative to start of segment

        Returns
        -------
        float or ndarray
            Axial force(s) at the location(s)
        """
        P1 = self.P1
        p1 = self.p1
        p2 = self.p2
        L = self.Length()

        # P(x) = P1 + (p2-p1)/(2L)*x^2 + p1*x
        return P1 + (p2 - p1)/(2*L)*x*x + p1*x

    def Torsion(self, x: Union[float, ndarray] = 0) -> Union[float, ndarray]:
        """
        Returns the torsional moment in the segment.
        Vectorized: accepts scalar or numpy array.

        Parameters
        ----------
        x : float or ndarray
            Location(s) relative to start of segment (unused, for interface consistency)

        Returns
        -------
        float or ndarray
            Torsional moment(s) - constant along segment
        """
        # The torsional moment is constant across the segment
        if isinstance(x, ndarray):
            return full(x.shape, self.T1)
        return self.T1

    def slope(self, x: Union[float, ndarray], P_delta: bool = False) -> Union[float, ndarray]:
        """
        Returns the slope of the elastic curve at location(s) along the segment.
        Vectorized: accepts scalar or numpy array.

        Parameters
        ----------
        x : float or ndarray
            Location(s) relative to start of segment where slope is to be calculated.
        P_delta : bool, optional
            Include P-little-delta effects. Defaults to False.

        Returns
        -------
        float or ndarray
            Slope(s) of the elastic curve (radians) at location(s).
        """
        V1 = self.V1
        M1 = self.M1
        w1 = self.w1
        w2 = self.w2
        theta_1 = self.theta1
        L = self.Length()
        EI = self.EI

        x2 = x * x
        x3 = x2 * x
        x4 = x3 * x

        if P_delta:
            P1 = self.P1
            delta_1 = self.delta1
            delta_x = self.deflection(x, P_delta)
            theta_x = theta_1 - (-V1*x2/2 - w1*x3/6 + x*(M1 - P1*delta_1 + P1*delta_x) + x4*(w1 - w2)/(24*L))/EI
        else:
            theta_x = theta_1 - (-V1*x2/2 - w1*x3/6 + x*M1 + x4*(w1 - w2)/(24*L))/EI

        return theta_x

    def deflection(self, x: Union[float, ndarray], P_delta: bool = False) -> Union[float, ndarray]:
        """
        Returns the deflection at location(s) on the segment.
        Vectorized: accepts scalar or numpy array.

        Parameters
        ----------
        x : float or ndarray
            Location(s) relative to start of segment
        P_delta : bool
            Include P-Delta effects

        Returns
        -------
        float or ndarray
            Deflection(s) at the location(s)
        """
        V1 = self.V1
        M1 = self.M1
        P1 = self.P1
        w1 = self.w1
        w2 = self.w2
        theta_1 = self.theta1
        delta_1 = self.delta1
        L = self.Length()
        EI = self.EI

        x2 = x * x
        x3 = x2 * x
        x4 = x3 * x
        x5 = x4 * x

        if P_delta:
            # Return the calculated deflection, amplified for P-delta effects
            numerator = delta_1 + theta_1*x + V1*x3/(6*EI) + w1*x4/(24*EI) + x2*(-M1 + P1*delta_1)/(2*EI) + x5*(-w1 + w2)/(120*EI*L)
            denominator = 1 + P1*x2/(2*EI)
            return numerator / denominator
        else:
            # Return the calculated deflection
            return delta_1 + theta_1*x + V1*x3/(6*EI) + w1*x4/(24*EI) + x2*(-M1)/(2*EI) + x5*(-w1 + w2)/(120*EI*L)

    def axial_deflection(self, x: Union[float, ndarray]) -> Union[float, ndarray]:
        """
        Returns the axial deflection at location(s) on the segment.
        Vectorized: accepts scalar or numpy array.

        Parameters
        ----------
        x : float or ndarray
            Location(s) relative to start of segment

        Returns
        -------
        float or ndarray
            Axial deflection(s) at the location(s)
        """
        delta_x1 = self.delta_x1
        P1 = self.P1
        p1 = self.p1
        p2 = self.p2
        L = self.Length()
        EA = self.EA

        x2 = x * x
        x3 = x2 * x

        return delta_x1 - 1/EA*(P1*x + p1*x2/2 + (p2 - p1)*x3/(6*L))

    def max_shear(self) -> float:
        """
        Returns the maximum shear in the segment.
        """
        w1 = self.w1
        w2 = self.w2
        L = self.Length()

        # Determine possible locations of maximum shear
        if w1 - w2 == 0:
            x1 = 0.0
        else:
            x1 = w1*L/(w1-w2)

        if x1 < 0 or x1 > L:
            x1 = 0.0

        # Evaluate at critical points
        x_vals = np.array([x1, 0.0, L])
        V_vals = self.Shear(x_vals)
        return float(np.max(V_vals))

    def min_shear(self) -> float:
        """
        Returns the minimum shear in the segment.
        """
        w1 = self.w1
        w2 = self.w2
        L = self.Length()

        # Determine possible locations of minimum shear
        if w1 - w2 == 0:
            x1 = 0.0
        else:
            x1 = w1*L/(w1-w2)

        if x1 < 0 or x1 > L:
            x1 = 0.0

        # Evaluate at critical points
        x_vals = np.array([x1, 0.0, L])
        V_vals = self.Shear(x_vals)
        return float(np.min(V_vals))

    def max_moment(self, P_delta: bool = False) -> float:
        """
        Returns the maximum moment in the segment.
        """
        w1 = self.w1
        w2 = self.w2
        V1 = self.V1
        L = self.Length()

        # Find the quadratic equation parameters
        a = -(w2-w1)/(2*L)
        b = -w1
        c = -V1

        # Determine possible locations of maximum moment
        x_list = [0.0, L]

        if a == 0:
            if b != 0:
                x_crit = -c/b
                if 0 <= x_crit <= L:
                    x_list.append(x_crit)
        else:
            discriminant = b*b - 4*a*c
            if discriminant >= 0:
                sqrt_disc = discriminant**0.5
                x1 = (-b + sqrt_disc)/(2*a)
                x2 = (-b - sqrt_disc)/(2*a)
                if 0 <= x1 <= L:
                    x_list.append(x1)
                if 0 <= x2 <= L:
                    x_list.append(x2)

        # Evaluate at critical points
        x_vals = np.array(x_list)
        M_vals = self.moment(x_vals, P_delta)
        return float(np.max(M_vals))

    def min_moment(self, P_delta: bool = False) -> float:
        """
        Returns the minimum moment in the segment.
        """
        w1 = self.w1
        w2 = self.w2
        V1 = self.V1
        L = self.Length()

        # Find the quadratic equation parameters
        a = -(w2-w1)/(2*L)
        b = -w1
        c = -V1

        # Determine possible locations of minimum moment
        x_list = [0.0, L]

        if a == 0:
            if b != 0:
                x_crit = -c/b
                if 0 <= x_crit <= L:
                    x_list.append(x_crit)
        else:
            discriminant = b*b - 4*a*c
            if discriminant >= 0:
                sqrt_disc = discriminant**0.5
                x1 = (-b + sqrt_disc)/(2*a)
                x2 = (-b - sqrt_disc)/(2*a)
                if 0 <= x1 <= L:
                    x_list.append(x1)
                if 0 <= x2 <= L:
                    x_list.append(x2)

        # Evaluate at critical points
        x_vals = np.array(x_list)
        M_vals = self.moment(x_vals, P_delta)
        return float(np.min(M_vals))

    def max_axial(self) -> float:
        """
        Returns the maximum axial force in the segment.
        """
        p1 = self.p1
        p2 = self.p2
        L = self.Length()

        # Determine possible locations of maximum axial force
        x_list = [0.0, L]

        if p1 - p2 != 0:
            x1 = L*p1/(p1-p2)
            if 0 <= x1 <= L:
                x_list.append(x1)

        # Evaluate at critical points
        x_vals = np.array(x_list)
        P_vals = self.axial(x_vals)
        return float(np.max(P_vals))

    def min_axial(self) -> float:
        """
        Returns the minimum axial force in the segment.
        """
        p1 = self.p1
        p2 = self.p2
        L = self.Length()

        # Determine possible locations of minimum axial force
        x_list = [0.0, L]

        if p1 - p2 != 0:
            x1 = L*p1/(p1-p2)
            if 0 <= x1 <= L:
                x_list.append(x1)

        # Evaluate at critical points
        x_vals = np.array(x_list)
        P_vals = self.axial(x_vals)
        return float(np.min(P_vals))

    def MaxTorsion(self) -> float:
        """
        Returns the maximum torsional moment in the segment.
        """
        return self.T1

    def MinTorsion(self) -> float:
        """
        Returns the minimum torsional moment in the segment.
        """
        return self.T1
