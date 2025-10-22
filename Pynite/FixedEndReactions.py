# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:58:03 2017

@author: D. Craig Brinck, SE
"""
# %%
from __future__ import annotations # Allows more recent type hints features
from typing import TYPE_CHECKING

from Pynite.cython import (
    fer_axial_linear_load,
    fer_axial_point_load,
    fer_linear_transverse_load,
    fer_moment,
    fer_point_load,
    fer_torque,
)

if TYPE_CHECKING:
    from typing import Literal
    from numpy import float64
    from numpy.typing import NDArray


# %%
def FER_PtLoad(P: float, x: float, L: float, Direction: Literal["Fy", "Fz"]) -> NDArray[float64]:
    """
    Returns the fixed end reaction vector for a point load

    Parameters:
    -----------
    P : float
        The magnitude of the point load
    x : float
        The location of the point load relative to the start of the member
    L : float
        The length of the member
    Direction : Literal["Fy", "Fz"]
        The direction of the point load. Must be one of the following:
            "Fy" = Force on the member's local y-axis
            "Fz" = Force on the member's local z-axis
    """
    # Translate the friendly axis label into the integer that the kernel expects
    if Direction == "Fy":
        axis = 0
    elif Direction == "Fz":
        axis = 1
    else:
        raise ValueError(f"Direction must be 'Fy' or 'Fz'. Received '{Direction}'.")

    return fer_point_load(P, x, L, axis)


def FER_Moment(M: float, x: float, L: float, Direction: Literal["My", "Mz"]) -> NDArray[float64]:
    """
    Returns the fixed end reaction vector for a concentrated moment

    Parameters
    ----------
    M : float
        The magnitude of the moment
    x : float
        The location of the moment relative to the start of the member
    L : float
        The length of the member
    Direction : Literal["My", "Mz"]
        The direction of the moment. Must be one of the following:
            "My" = Moment applied about the local y-axis
            "Mz" = Moment applied about the local z-axis
    """

    # Convert rotational axis strings into the kernel's 0/1 index
    if Direction == "My":
        axis = 0
    elif Direction == "Mz":
        axis = 1
    else:
        raise ValueError(f"Direction must be 'My' or 'Mz'. Received '{Direction}'.")

    return fer_moment(M, x, L, axis)


# Returns the fixed end reaction vector for a linear distributed load
def FER_LinLoad(w1: float, w2: float, x1: float, x2: float, L: float, Direction: Literal["Fy", "Fz"]) -> NDArray[float64]:
    """
    Returns the fixed end reaction vector for a linear distributed load

    Parameters
    ----------
    w1 : float
        The load magnitude at the start location
    w2 : float
        The load magnitude at the end location
    x1 : float
        The start location of the distributed load
    x2 : float
        The end location of the distributed load
    L : float
        The length of the member
    Direction : Literal["Fy", "Fz"]
        The direction of the distributed load. Must be one of the following:
            "Fy" = Force on the member's local y-axis
            "Fz" = Force on the member's local z-axis
    """

    # Direction label determines whether the transverse load acts about local y or z
    if Direction == "Fy":
        axis = 0
    elif Direction == "Fz":
        axis = 1
    else:
        raise ValueError(f"Direction must be 'Fy' or 'Fz'. Received '{Direction}'.")

    return fer_linear_transverse_load(w1, w2, x1, x2, L, axis)


# Returns the fixed end reaction vector for an axial point load
def FER_AxialPtLoad(P: float, x: float, L: float) -> NDArray[float64]:
    """
    Returns the fixed end reaction vector for an axial point load

    Parameters
    ----------
    P : float
        The magnitude of the axial point load
    x : float
        The location of the axial point load relative to the start of the member
    L : float
        The length of the member
    """

    return fer_axial_point_load(P, x, L)


# Returns the fixed end reaction vector for a distributed axial load
def FER_AxialLinLoad(p1: float, p2: float, x1: float, x2: float, L: float) -> NDArray[float64]:
    """
    Returns the fixed end reaction vector for a distributed axial load

    Parameters
    ----------
    p1 : float
        The axial load magnitude at the start location
    p2 : float
        The axial load magnitude at the end location
    x1 : float
        The start location of the distributed axial load
    x2 : float
        The end location of the distributed axial load
    L : float
        The length of the member
    """

    return fer_axial_linear_load(p1, p2, x1, x2, L)


def FER_Torque(T: float, x: float, L: float) -> NDArray[float64]:
    """
    Returns the fixed end reaction vector for a concentrated torque

    Parameters
    ----------
    T : float
        The magnitude of the torque
    x : float
        The location of the torque relative to the start of the member
    L : float
        The length of the member
    """

    return fer_torque(T, x, L)
