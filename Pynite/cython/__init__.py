"""
Typed wrappers around the accelerated numerical kernels.

The package exposes pure-Python fallbacks that Cython can compile into
extension modules. Regular Python imports continue to work, but Cythonized
builds will seamlessly replace the pure implementations with optimized
versions.
"""

from __future__ import annotations

from .kernels import (
    accumulate_bending_shear_stiffness,
    accumulate_membrane_stiffness,
    beam_member_stiffness_matrix,
    compute_member_transformation_matrix,
    compute_ring_trig,
    compute_quad_local_coords,
    compute_quad_transformation_matrix,
    cross_product_3d,
    evaluate_polynomial,
    evaluate_piecewise_polynomial,
    expand_bending_matrix,
    expand_membrane_matrix,
    fer_axial_linear_load,
    fer_axial_point_load,
    fer_linear_transverse_load,
    fer_moment,
    fer_point_load,
    fer_torque,
    normalize_vector_3d,
    quad_membrane_at,
    quad_membrane_batch,
    quad_moment_at,
    quad_moment_batch,
)
from .sparse import expand_stiffness_blocks

__all__ = [
    "accumulate_bending_shear_stiffness",
    "accumulate_membrane_stiffness",
    "beam_member_stiffness_matrix",
    "compute_member_transformation_matrix",
    "compute_quad_local_coords",
    "compute_quad_transformation_matrix",
    "compute_ring_trig",
    "cross_product_3d",
    "evaluate_polynomial",
    "evaluate_piecewise_polynomial",
    "expand_bending_matrix",
    "expand_membrane_matrix",
    "expand_stiffness_blocks",
    "fer_axial_linear_load",
    "fer_axial_point_load",
    "fer_linear_transverse_load",
    "fer_moment",
    "fer_point_load",
    "fer_torque",
    "normalize_vector_3d",
    "quad_membrane_at",
    "quad_membrane_batch",
    "quad_moment_at",
    "quad_moment_batch",
]
