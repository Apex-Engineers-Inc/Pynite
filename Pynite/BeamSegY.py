# -*- coding: utf-8 -*-
"""
BeamSegY - Y-direction beam segment with vectorized operations.

Inherits from BeamSegZ but overrides moment, slope, and deflection methods
for the opposite sign convention used in the local y-axis bending.
"""
from __future__ import annotations
from typing import Union

import numpy as np
from numpy import ndarray

from Pynite.BeamSegZ import BeamSegZ


# %%
class BeamSegY(BeamSegZ):
    """
    Beam segment for y-direction bending with vectorized operations.
    Inherits __slots__ and base methods from BeamSegZ.
    """

    def moment(self, x: Union[float, ndarray], P_delta: bool = False) -> Union[float, ndarray]:
        """
        Returns the moment at location(s) on the segment.
        Vectorized: accepts scalar or numpy array.

        Parameters
        ----------
        x : float or ndarray
            Location(s) relative to start of segment where moment is to be calculated
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

        # M = -M1 - V1*x - w1*x^2/2 - x^3*(-w1+w2)/(6L)
        x2 = x * x
        x3 = x2 * x
        M = -M1 - V1*x - w1*x2/2 - x3*(-w1 + w2)/(6*L)

        if P_delta:
            P1 = self.P1
            delta1 = self.delta1
            delta = self.deflection(x, P_delta=False)  # Avoid recursion
            M = M + P1*(delta - delta1)

        return M

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
            delta_x = self.deflection(x, P_delta)
            delta_1 = self.delta1
            return theta_1 + (-V1*x2/2 - w1*x3/6 + x*(-M1 - P1*delta_1 + P1*delta_x) + x4*(w1 - w2)/(24*L))/EI
        else:
            return theta_1 + (-V1*x2/2 - w1*x3/6 + x*(-M1) + x4*(w1 - w2)/(24*L))/EI

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
            numerator = delta_1 - theta_1*x + V1*x3/(6*EI) + w1*x4/(24*EI) - x2*(-M1 - P1*delta_1)/(2*EI) - x5*(w1 - w2)/(120*EI*L)
            denominator = 1 + P1*x2/(2*EI)
            return numerator / denominator
        else:
            # Return the calculated deflection
            return delta_1 - theta_1*x + V1*x3/(6*EI) + w1*x4/(24*EI) - x2*(-M1)/(2*EI) - x5*(w1 - w2)/(120*EI*L)

    def max_moment(self, P_delta: bool = False) -> float:
        """
        Returns the maximum moment in the segment.
        """
        w1 = self.w1
        w2 = self.w2
        V1 = self.V1
        L = self.Length()

        # Find the quadratic equation parameters (opposite sign for y-bending)
        a = (w2-w1)/(2*L)
        b = w1
        c = V1

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

        # Find the quadratic equation parameters (opposite sign for y-bending)
        a = (w2-w1)/(2*L)
        b = w1
        c = V1

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
