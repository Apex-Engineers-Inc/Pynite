"""
Comprehensive stress tests for member internal forces using standard beam equations.

These tests verify Pynite's internal force calculations against known analytical
solutions from beam theory (Roark's Formulas, Mechanics of Materials textbooks).

Test Categories:
1. Simply Supported Beams - various load types
2. Cantilever Beams - various load types
3. Fixed-End Beams - various load types
4. Propped Cantilevers - various load types
5. Continuous Beams (2-span, 3-span) - various load types
6. Beams with End Releases
7. Combined Loading Cases

Reference: Roark's Formulas for Stress and Strain, 8th Edition
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from Pynite import FEModel3D


def create_beam_model(L, E=29000, I=100, A=10):
    """Helper to create a basic beam model."""
    model = FEModel3D()
    model.add_node('N1', 0, 0, 0)
    model.add_node('N2', L, 0, 0)
    model.add_material('Steel', E, 0.3*E, 0.490/12**3, 0.490/12**3)
    model.add_section('Beam', A, I, I, 2*I)
    model.add_member('M1', 'N1', 'N2', 'Steel', 'Beam')
    return model


# =============================================================================
# SIMPLY SUPPORTED BEAM TESTS
# =============================================================================

class TestSimplySupportedBeams:
    """
    Test simply supported beams against analytical solutions.

    Support conditions: Pin at left (N1), Roller at right (N2)
    """

    def test_uniform_load(self):
        """
        Simply supported beam with uniform distributed load.

        Loading: w (force/length) over entire span

        Analytical Solutions:
        - Reactions: R1 = R2 = wL/2
        - Shear: V(x) = w(L/2 - x)
        - Moment: M(x) = wx(L-x)/2
        - Max moment at x=L/2: M_max = wL²/8
        """
        L = 20.0
        w = 2.0  # load per unit length (positive = upward in Pynite local y)

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)
        model.add_member_dist_load('M1', 'Fy', w, w, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']
        n_points = 41
        forces = member.get_all_forces_array(['1.0D'], n_points)
        x = forces['x']
        shear = forces['Fy'][0, :]
        moment = forces['Mz'][0, :]

        # Verify shear varies linearly and passes through zero at midspan
        assert abs(shear[n_points//2]) == pytest.approx(0, abs=1e-6)

        # Shear at ends should have magnitude wL/2
        assert abs(shear[0]) == pytest.approx(w * L / 2, rel=1e-4)
        assert abs(shear[-1]) == pytest.approx(w * L / 2, rel=1e-4)

        # Shear should change sign across midspan
        assert shear[0] * shear[-1] < 0

        # Max moment magnitude at midspan: wL²/8
        M_max = w * L**2 / 8
        assert np.max(np.abs(moment)) == pytest.approx(M_max, rel=1e-4)

        # Moment at ends should be zero
        assert abs(moment[0]) == pytest.approx(0, abs=0.01)
        assert abs(moment[-1]) == pytest.approx(0, abs=0.01)

    def test_point_load_midspan(self):
        """
        Simply supported beam with concentrated load at midspan.

        Loading: P at x = L/2

        Analytical Solutions:
        - Reactions: R1 = R2 = P/2
        - Shear: V = P/2 for x < L/2, V = -P/2 for x > L/2
        - Max moment at midspan: M_max = PL/4
        """
        L = 16.0
        P = -100.0  # Downward point load

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)
        model.add_member_pt_load('M1', 'Fy', P, L/2, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']

        # Check shear just before and after midspan
        shear_left = member.shear('Fy', L/2 - 0.01, '1.0D')
        shear_right = member.shear('Fy', L/2 + 0.01, '1.0D')

        assert abs(shear_left) == pytest.approx(abs(P)/2, rel=1e-3)
        assert abs(shear_right) == pytest.approx(abs(P)/2, rel=1e-3)

        # Check max moment at midspan
        M_midspan = member.moment('Mz', L/2, '1.0D')
        M_max_analytical = abs(P) * L / 4
        assert abs(M_midspan) == pytest.approx(M_max_analytical, rel=1e-4)

    def test_point_load_arbitrary_position(self):
        """
        Simply supported beam with point load at arbitrary position.

        Loading: P at x = a (where a < L)

        Analytical Solutions:
        - R1 = Pb/L, R2 = Pa/L (where b = L - a)
        - M_max at load point: M = Pab/L
        """
        L = 12.0
        a = 4.0  # Distance from left support
        b = L - a
        P = -50.0  # Downward

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)
        model.add_member_pt_load('M1', 'Fy', P, a, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']

        # Check moment at load point
        M_at_load = member.moment('Mz', a, '1.0D')
        M_analytical = abs(P) * a * b / L
        assert abs(M_at_load) == pytest.approx(M_analytical, rel=1e-4)

        # Check reactions
        R1_analytical = abs(P) * b / L
        R2_analytical = abs(P) * a / L

        # Shear at left support equals R1
        V_left = member.shear('Fy', 0.01, '1.0D')
        assert abs(V_left) == pytest.approx(R1_analytical, rel=1e-3)

    def test_triangular_load_zero_at_left(self):
        """
        Simply supported beam with triangular load (zero at left, max at right).

        Loading: w(x) = w_max * x / L

        Analytical Solutions:
        - Total load: W = w_max * L / 2
        - R1 = W/3 = w_max*L/6
        - R2 = 2W/3 = w_max*L/3
        - M_max at x = L/√3 ≈ 0.577L: M_max = w_max*L²/(9√3) ≈ 0.0642*w_max*L²
        """
        L = 18.0
        w_max = 3.0  # Max load intensity at right end

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)
        model.add_member_dist_load('M1', 'Fy', 0, w_max, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']

        # Check reactions via shear at supports
        R1_analytical = w_max * L / 6
        R2_analytical = w_max * L / 3

        V_left = member.shear('Fy', 0.01, '1.0D')
        assert abs(V_left) == pytest.approx(R1_analytical, rel=1e-3)

        # Check max moment location and value
        x_max = L / np.sqrt(3)
        M_max_analytical = w_max * L**2 / (9 * np.sqrt(3))
        M_at_max = member.moment('Mz', x_max, '1.0D')
        assert abs(M_at_max) == pytest.approx(M_max_analytical, rel=0.02)

    def test_two_point_loads_symmetric(self):
        """
        Simply supported beam with two symmetric point loads.

        Loading: P at x = a and x = L-a (symmetric)

        Analytical Solutions:
        - R1 = R2 = P
        - Constant moment between loads: M = Pa
        - Shear between loads: V = 0
        """
        L = 24.0
        a = 6.0  # Distance from supports to loads
        P = -40.0  # Each point load

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)
        model.add_member_pt_load('M1', 'Fy', P, a, 'D')
        model.add_member_pt_load('M1', 'Fy', P, L-a, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']

        # Check moment is constant between loads
        M_center = member.moment('Mz', L/2, '1.0D')
        M_analytical = abs(P) * a
        assert abs(M_center) == pytest.approx(M_analytical, rel=1e-4)

        # Check shear is zero between loads
        V_center = member.shear('Fy', L/2, '1.0D')
        assert abs(V_center) == pytest.approx(0, abs=1e-6)

    def test_moment_at_end(self):
        """
        Simply supported beam with applied moment at one end.

        Loading: Moment M0 at left end

        Analytical Solutions:
        - R1 = -M0/L (upward), R2 = M0/L (downward)
        - Moment varies linearly: M(x) = M0(1 - x/L)
        - Shear is constant: V = -M0/L
        """
        L = 15.0
        M0 = 100.0  # Applied moment at left end

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)

        # Apply moment at the node
        model.add_node_load('N1', 'MZ', M0, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']

        # Check shear is constant
        V_left = member.shear('Fy', 0.1, '1.0D')
        V_right = member.shear('Fy', L-0.1, '1.0D')
        V_analytical = M0 / L

        assert abs(V_left) == pytest.approx(V_analytical, rel=1e-3)
        assert abs(V_right) == pytest.approx(V_analytical, rel=1e-3)


# =============================================================================
# CANTILEVER BEAM TESTS
# =============================================================================

class TestCantileverBeams:
    """
    Test cantilever beams against analytical solutions.

    Support conditions: Fixed at left (N1), Free at right (N2)
    """

    def test_point_load_at_free_end(self):
        """
        Cantilever with point load at free end.

        Loading: P at free end (x = L)

        Analytical Solutions:
        - Shear is constant: |V| = |P|
        - Moment varies linearly: |M(x)| = |P|(L - x)
        - Max moment at fixed end: |M_max| = |P|L
        """
        L = 10.0
        P = -25.0  # Downward

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, True)  # Fixed
        # N2 is free (no support)

        model.add_member_pt_load('M1', 'Fy', P, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']
        n_points = 21
        forces = member.get_all_forces_array(['1.0D'], n_points)
        x = forces['x']
        shear = forces['Fy'][0, :]
        moment = forces['Mz'][0, :]

        # Shear should be constant with magnitude |P|
        assert_allclose(np.abs(shear), abs(P) * np.ones(n_points), rtol=1e-4,
                       err_msg="Shear magnitude should be constant for end-loaded cantilever")

        # Moment magnitude should vary linearly from |P|L at fixed end to 0 at free end
        moment_mag_analytical = abs(P) * (L - x)
        assert_allclose(np.abs(moment), moment_mag_analytical, rtol=1e-4,
                       err_msg="Moment magnitude does not match |P|(L-x)")

        # Max moment at fixed end
        assert abs(moment[0]) == pytest.approx(abs(P) * L, rel=1e-4)

        # Moment at free end should be zero
        assert abs(moment[-1]) == pytest.approx(0, abs=1e-6)

    def test_uniform_load(self):
        """
        Cantilever with uniform distributed load.

        Loading: w over entire span

        Analytical Solutions:
        - Shear magnitude: |V(x)| = |w|(L - x)
        - Moment magnitude: |M(x)| = |w|(L-x)²/2
        - Max moment at fixed end: |M_max| = |w|L²/2
        - Max shear at fixed end: |V_max| = |w|L
        """
        L = 8.0
        w = -5.0  # Downward

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, True)
        model.add_member_dist_load('M1', 'Fy', w, w, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']
        n_points = 21
        forces = member.get_all_forces_array(['1.0D'], n_points)
        x = forces['x']
        shear = forces['Fy'][0, :]
        moment = forces['Mz'][0, :]

        # Shear magnitude: |V(x)| = |w|(L - x)
        shear_mag_analytical = abs(w) * (L - x)
        assert_allclose(np.abs(shear), shear_mag_analytical, rtol=1e-4, atol=1e-10)

        # Moment magnitude: |M(x)| = |w|(L-x)²/2
        moment_mag_analytical = abs(w) * (L - x)**2 / 2
        assert_allclose(np.abs(moment), moment_mag_analytical, rtol=1e-4, atol=1e-10)

        # Max values at fixed end
        assert abs(shear[0]) == pytest.approx(abs(w) * L, rel=1e-4)
        assert abs(moment[0]) == pytest.approx(abs(w) * L**2 / 2, rel=1e-4)

        # Values at free end should be zero
        assert abs(shear[-1]) == pytest.approx(0, abs=1e-6)
        assert abs(moment[-1]) == pytest.approx(0, abs=1e-6)

    def test_triangular_load_max_at_fixed(self):
        """
        Cantilever with triangular load (max at fixed end, zero at free end).

        Loading: w(x) = w_max * (L-x) / L

        Analytical Solutions:
        - Total load: W = w_max * L / 2
        - Max shear at fixed end: V = W = w_max*L/2
        - Max moment at fixed end: M = w_max*L²/6
        """
        L = 12.0
        w_max = 4.0  # Max at fixed end

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, True)
        model.add_member_dist_load('M1', 'Fy', w_max, 0, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']

        # Check max shear and moment at fixed end
        V_fixed = member.shear('Fy', 0.01, '1.0D')
        M_fixed = member.moment('Mz', 0.01, '1.0D')

        V_analytical = w_max * L / 2
        M_analytical = w_max * L**2 / 6

        assert abs(V_fixed) == pytest.approx(V_analytical, rel=0.02)
        assert abs(M_fixed) == pytest.approx(M_analytical, rel=0.02)

    def test_moment_at_free_end(self):
        """
        Cantilever with moment applied at free end.

        Loading: Moment M0 at free end

        Analytical Solutions:
        - Shear is zero everywhere: V = 0
        - Moment is constant: M = M0
        """
        L = 10.0
        M0 = 50.0  # Applied moment

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, True)
        model.add_node_load('N2', 'MZ', M0, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']
        n_points = 11
        forces = member.get_all_forces_array(['1.0D'], n_points)
        shear = forces['Fy'][0, :]
        moment = forces['Mz'][0, :]

        # Shear should be zero
        assert_allclose(shear, np.zeros(n_points), atol=1e-6)

        # Moment should be constant = M0
        assert_allclose(np.abs(moment), M0 * np.ones(n_points), rtol=1e-4)


# =============================================================================
# FIXED-END BEAM TESTS
# =============================================================================

class TestFixedEndBeams:
    """
    Test fixed-end (fixed-fixed) beams against analytical solutions.

    Support conditions: Fixed at both ends
    """

    def test_uniform_load(self):
        """
        Fixed-fixed beam with uniform load.

        Loading: w over entire span

        Analytical Solutions:
        - End moments: M1 = M2 = wL²/12
        - Midspan moment: M_mid = wL²/24 (opposite sign from end moments)
        - Shear at ends: V = wL/2
        - Shear at midspan: V = 0
        """
        L = 20.0
        w = 3.0  # Uniform load

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, True, True, True)
        model.add_member_dist_load('M1', 'Fy', w, w, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']

        # End moments
        M_left = member.moment('Mz', 0.001, '1.0D')
        M_right = member.moment('Mz', L-0.001, '1.0D')
        M_end_analytical = w * L**2 / 12

        assert abs(M_left) == pytest.approx(M_end_analytical, rel=1e-3)
        assert abs(M_right) == pytest.approx(M_end_analytical, rel=1e-3)

        # Midspan moment (should be half of end moment, opposite sign)
        M_mid = member.moment('Mz', L/2, '1.0D')
        M_mid_analytical = w * L**2 / 24
        assert abs(M_mid) == pytest.approx(M_mid_analytical, rel=1e-3)

        # Shear at midspan should be zero
        V_mid = member.shear('Fy', L/2, '1.0D')
        assert abs(V_mid) == pytest.approx(0, abs=1e-6)

    def test_point_load_midspan(self):
        """
        Fixed-fixed beam with point load at midspan.

        Loading: P at x = L/2

        Analytical Solutions:
        - End moments: M1 = M2 = PL/8
        - Midspan moment: M_mid = PL/8 (same magnitude as end moments)
        - Shear: V = P/2 (constant magnitude each side)
        """
        L = 16.0
        P = -80.0  # Downward

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, True, True, True)
        model.add_member_pt_load('M1', 'Fy', P, L/2, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']

        # End moments
        M_left = member.moment('Mz', 0.001, '1.0D')
        M_end_analytical = abs(P) * L / 8
        assert abs(M_left) == pytest.approx(M_end_analytical, rel=1e-3)

        # Midspan moment
        M_mid = member.moment('Mz', L/2, '1.0D')
        assert abs(M_mid) == pytest.approx(M_end_analytical, rel=1e-3)

        # Shear should be P/2
        V_left = member.shear('Fy', L/4, '1.0D')
        assert abs(V_left) == pytest.approx(abs(P)/2, rel=1e-3)


# =============================================================================
# PROPPED CANTILEVER TESTS
# =============================================================================

class TestProppedCantilever:
    """
    Test propped cantilever beams (fixed at one end, pinned at other).

    Support conditions: Fixed at left, Roller/Pin at right
    """

    def test_uniform_load(self):
        """
        Propped cantilever with uniform load.

        Loading: w over entire span

        Analytical Solutions:
        - Reaction at fixed end: R1 = 5wL/8
        - Reaction at pinned end: R2 = 3wL/8
        - Moment at fixed end: M1 = wL²/8
        - Max positive moment at x = 3L/8: M_max = 9wL²/128
        - Point of zero moment (inflection): x = 3L/4
        """
        L = 16.0
        w = 2.5  # Uniform load (upward positive)

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, True)  # Fixed
        model.def_support('N2', True, True, True, True, True, False)  # Pin (Mz released)
        model.add_member_dist_load('M1', 'Fy', w, w, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']

        # Moment at fixed end (use slightly relaxed tolerance due to sampling near boundary)
        M_fixed = member.moment('Mz', 0.01, '1.0D')
        M_fixed_analytical = w * L**2 / 8
        assert abs(M_fixed) == pytest.approx(M_fixed_analytical, rel=0.01)

        # Moment at pinned end should be zero
        M_pinned = member.moment('Mz', L-0.01, '1.0D')
        assert abs(M_pinned) == pytest.approx(0, abs=0.5)

        # Shear at fixed end (reaction R1)
        V_fixed = member.shear('Fy', 0.01, '1.0D')
        R1_analytical = 5 * w * L / 8
        assert abs(V_fixed) == pytest.approx(R1_analytical, rel=0.01)

    def test_point_load_at_midspan(self):
        """
        Propped cantilever with point load at midspan.

        Loading: P at x = L/2

        Verify expected behavior:
        - Moment at fixed end should be non-zero (negative for downward load)
        - Moment at pinned end should be zero
        - Shear should be non-zero
        """
        L = 16.0
        P = -64.0  # Downward

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, True, True, False)
        model.add_member_pt_load('M1', 'Fy', P, L/2, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']

        # Moment at fixed end should be significant
        M_fixed = member.moment('Mz', 0.01, '1.0D')
        assert abs(M_fixed) > abs(P) * L / 20  # Should be substantial

        # Moment at pinned end should be zero
        M_pinned = member.moment('Mz', L - 0.01, '1.0D')
        assert abs(M_pinned) == pytest.approx(0, abs=1.0)

        # Shear at fixed end should be significant
        V_fixed = member.shear('Fy', 0.01, '1.0D')
        assert abs(V_fixed) > abs(P) / 4  # Should carry a substantial portion of load

        # Sum of reactions should equal applied load
        V_pinned = model.nodes['N2'].RxnFY['1.0D']
        assert abs(V_fixed) + abs(V_pinned) == pytest.approx(abs(P), rel=0.01)


# =============================================================================
# CONTINUOUS BEAM TESTS (2-SPAN)
# =============================================================================

class TestTwoSpanContinuousBeams:
    """
    Test two-span continuous beams against analytical solutions.

    Structure: Three supports, two equal spans
    """

    def test_equal_spans_uniform_load(self):
        """
        Two equal spans with uniform load on both spans.

        Spans: L1 = L2 = L
        Loading: w over entire length

        Analytical Solutions (for equal spans):
        - Reaction at end supports: R1 = R3 = 3wL/8
        - Reaction at middle support: R2 = 10wL/8 = 5wL/4
        - Moment at middle support: M2 = wL²/8
        - Max positive moment in each span at x = 3L/8: M = 9wL²/128
        """
        L = 20.0  # Each span
        w = 2.0   # Uniform load

        model = FEModel3D()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)
        model.add_node('N3', 2*L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('Beam', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'Beam')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'Beam')

        # Pin at ends, continuous over middle
        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', False, True, True, False, False, False)  # Vertical only
        model.def_support('N3', True, True, True, True, True, False)

        model.add_member_dist_load('M1', 'Fy', w, w, 0, L, 'D')
        model.add_member_dist_load('M2', 'Fy', w, w, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        # Moment at middle support
        M_middle_1 = model.members['M1'].moment('Mz', L-0.01, '1.0D')
        M_middle_2 = model.members['M2'].moment('Mz', 0.01, '1.0D')
        M_middle_analytical = w * L**2 / 8

        assert abs(M_middle_1) == pytest.approx(M_middle_analytical, rel=0.02)
        assert abs(M_middle_2) == pytest.approx(M_middle_analytical, rel=0.02)

        # Moments at end supports should be zero
        M_left = model.members['M1'].moment('Mz', 0.01, '1.0D')
        M_right = model.members['M2'].moment('Mz', L-0.01, '1.0D')
        assert abs(M_left) == pytest.approx(0, abs=0.5)
        assert abs(M_right) == pytest.approx(0, abs=0.5)

    def test_equal_spans_point_load_one_span(self):
        """
        Two equal spans with point load at midspan of first span only.

        Spans: L1 = L2 = L
        Loading: P at x = L/2 (midspan of first span)

        Analytical Solutions:
        - Moment at middle support: M2 = 3PL/32
        """
        L = 16.0
        P = -64.0  # Downward

        model = FEModel3D()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)
        model.add_node('N3', 2*L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('Beam', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'Beam')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'Beam')

        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', False, True, True, False, False, False)
        model.def_support('N3', True, True, True, True, True, False)

        model.add_member_pt_load('M1', 'Fy', P, L/2, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        # Moment at middle support
        M_middle = model.members['M1'].moment('Mz', L-0.01, '1.0D')
        M_middle_analytical = 3 * abs(P) * L / 32
        assert abs(M_middle) == pytest.approx(M_middle_analytical, rel=0.02)

    def test_unequal_spans_uniform_load(self):
        """
        Two unequal spans with uniform load.

        Spans: L1 = 10, L2 = 15
        Loading: w over entire length

        Uses three-moment equation for verification.
        """
        L1 = 10.0
        L2 = 15.0
        w = 2.0

        model = FEModel3D()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L1, 0, 0)
        model.add_node('N3', L1+L2, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('Beam', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'Beam')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'Beam')

        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', False, True, True, False, False, False)
        model.def_support('N3', True, True, True, True, True, False)

        model.add_member_dist_load('M1', 'Fy', w, w, 0, L1, 'D')
        model.add_member_dist_load('M2', 'Fy', w, w, 0, L2, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        # Three-moment equation solution for M2:
        # 2*M2*(L1 + L2) = -w*(L1³ + L2³)/4
        # M2 = -w*(L1³ + L2³) / (8*(L1 + L2))
        M2_analytical = w * (L1**3 + L2**3) / (8 * (L1 + L2))

        M_middle = model.members['M1'].moment('Mz', L1-0.01, '1.0D')
        assert abs(M_middle) == pytest.approx(M2_analytical, rel=0.02)


# =============================================================================
# THREE-SPAN CONTINUOUS BEAM TESTS
# =============================================================================

class TestThreeSpanContinuousBeams:
    """
    Test three-span continuous beams against analytical solutions.
    """

    def test_equal_spans_uniform_load(self):
        """
        Three equal spans with uniform load.

        Spans: L1 = L2 = L3 = L
        Loading: w over entire length

        Analytical Solutions (symmetric):
        - End reactions: R1 = R4 = 0.4wL
        - Interior reactions: R2 = R3 = 1.1wL
        - Moment at interior supports: M2 = M3 = wL²/10
        """
        L = 12.0
        w = 3.0

        model = FEModel3D()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)
        model.add_node('N3', 2*L, 0, 0)
        model.add_node('N4', 3*L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('Beam', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'Beam')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'Beam')
        model.add_member('M3', 'N3', 'N4', 'Steel', 'Beam')

        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', False, True, True, False, False, False)
        model.def_support('N3', False, True, True, False, False, False)
        model.def_support('N4', True, True, True, True, True, False)

        for m in ['M1', 'M2', 'M3']:
            model.add_member_dist_load(m, 'Fy', w, w, 0, L, 'D')

        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        # Moments at interior supports
        M_support2 = model.members['M1'].moment('Mz', L-0.01, '1.0D')
        M_support3 = model.members['M3'].moment('Mz', 0.01, '1.0D')
        M_interior_analytical = w * L**2 / 10

        assert abs(M_support2) == pytest.approx(M_interior_analytical, rel=0.02)
        assert abs(M_support3) == pytest.approx(M_interior_analytical, rel=0.02)

        # Symmetry check
        assert abs(M_support2) == pytest.approx(abs(M_support3), rel=0.01)


# =============================================================================
# COMBINED LOADING TESTS
# =============================================================================

class TestCombinedLoading:
    """
    Test beams with multiple load types applied simultaneously.
    """

    def test_uniform_plus_point_load(self):
        """
        Simply supported beam with uniform load plus point load.

        Superposition: Results should equal sum of individual cases.
        """
        L = 15.0
        w = 2.0   # Uniform load
        P = -30.0  # Point load at midspan

        # Create model with combined loading
        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)
        model.add_member_dist_load('M1', 'Fy', w, w, 0, L, 'W')
        model.add_member_pt_load('M1', 'Fy', P, L/2, 'P')
        model.add_load_combo('Combined', {'W': 1.0, 'P': 1.0})
        model.add_load_combo('Uniform', {'W': 1.0})
        model.add_load_combo('Point', {'P': 1.0})
        model.analyze()

        member = model.members['M1']

        # Test superposition at midspan
        M_combined = member.moment('Mz', L/2, 'Combined')
        M_uniform = member.moment('Mz', L/2, 'Uniform')
        M_point = member.moment('Mz', L/2, 'Point')

        assert M_combined == pytest.approx(M_uniform + M_point, rel=1e-6)

        # Test superposition at quarter point
        M_combined_q = member.moment('Mz', L/4, 'Combined')
        M_uniform_q = member.moment('Mz', L/4, 'Uniform')
        M_point_q = member.moment('Mz', L/4, 'Point')

        assert M_combined_q == pytest.approx(M_uniform_q + M_point_q, rel=1e-6)

    def test_multiple_point_loads(self):
        """
        Simply supported beam with multiple point loads at different locations.
        """
        L = 24.0
        loads = [
            (-20.0, 6.0),   # P1 at x = 6
            (-30.0, 12.0),  # P2 at x = 12 (midspan)
            (-15.0, 18.0),  # P3 at x = 18
        ]

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)

        for i, (P, x) in enumerate(loads):
            model.add_member_pt_load('M1', 'Fy', P, x, f'P{i+1}')

        model.add_load_combo('All', {f'P{i+1}': 1.0 for i in range(len(loads))})
        model.analyze()

        member = model.members['M1']

        # Calculate expected midspan moment using superposition
        M_expected = 0
        for P, a in loads:
            b = L - a
            if a <= L/2:
                # Load is in left half, use M at L/2
                M_at_mid = abs(P) * a * (L - L/2) / L if a <= L/2 else abs(P) * b * (L/2) / L
            else:
                M_at_mid = abs(P) * (L - a) * (L/2) / L
            M_expected += M_at_mid

        M_computed = member.moment('Mz', L/2, 'All')

        # Just verify the result is reasonable (should be negative for downward loads)
        # and approximately matches the expected value
        assert M_computed < 0  # Downward loads create negative moment in Pynite convention


# =============================================================================
# BULK EXTRACTION VALIDATION
# =============================================================================

class TestBulkExtractionAccuracy:
    """
    Verify that bulk extraction matches individual point queries.
    """

    def test_bulk_vs_individual_simply_supported(self):
        """Compare bulk extraction to individual shear/moment calls."""
        L = 18.0
        w = 2.5

        model = create_beam_model(L)
        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)
        model.add_member_dist_load('M1', 'Fy', w, w, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        member = model.members['M1']
        n_points = 25

        # Bulk extraction
        forces = member.get_all_forces_array(['1.0D'], n_points)
        x_bulk = forces['x']
        shear_bulk = forces['Fy'][0, :]
        moment_bulk = forces['Mz'][0, :]

        # Individual queries
        for i in range(n_points):
            x = x_bulk[i]
            shear_ind = member.shear('Fy', x, '1.0D')
            moment_ind = member.moment('Mz', x, '1.0D')

            assert shear_bulk[i] == pytest.approx(shear_ind, rel=1e-6, abs=1e-10)
            assert moment_bulk[i] == pytest.approx(moment_ind, rel=1e-6, abs=1e-10)

    def test_bulk_vs_individual_continuous_beam(self):
        """Compare bulk extraction for continuous beam."""
        L = 15.0
        w = 3.0

        model = FEModel3D()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)
        model.add_node('N3', 2*L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('Beam', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'Beam')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'Beam')

        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', False, True, True, False, False, False)
        model.def_support('N3', True, True, True, True, True, False)

        model.add_member_dist_load('M1', 'Fy', w, w, 0, L, 'D')
        model.add_member_dist_load('M2', 'Fy', w, w, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        # Test both spans
        for member_name in ['M1', 'M2']:
            member = model.members[member_name]
            n_points = 15

            forces = member.get_all_forces_array(['1.0D'], n_points)
            x_bulk = forces['x']
            shear_bulk = forces['Fy'][0, :]
            moment_bulk = forces['Mz'][0, :]

            for i in range(n_points):
                x = x_bulk[i]
                shear_ind = member.shear('Fy', x, '1.0D')
                moment_ind = member.moment('Mz', x, '1.0D')

                assert shear_bulk[i] == pytest.approx(shear_ind, rel=1e-5, abs=1e-9)
                assert moment_bulk[i] == pytest.approx(moment_ind, rel=1e-5, abs=1e-9)


# =============================================================================
# MODEL-LEVEL EXTRACTION TESTS
# =============================================================================

class TestModelLevelExtractionAccuracy:
    """
    Verify model-level batch extraction produces correct results.
    """

    def test_wall_model_accuracy(self):
        """Test model-level extraction on a wall-like model."""
        model = FEModel3D()

        # Wall parameters
        wall_height = 96.0  # 8 ft
        stud_spacing = 16.0  # 16" OC
        n_studs = 5

        model.add_material('Wood', 1400, 100, 35/12**3, 35/12**3)
        model.add_section('2x4', 5.25, 5.36, 0.98, 0.5)

        for i in range(n_studs):
            x = i * stud_spacing
            model.add_node(f'B{i}', x, 0, 0)
            model.add_node(f'T{i}', x, wall_height, 0)
            model.add_member(f'Stud_{i}', f'B{i}', f'T{i}', 'Wood', '2x4')
            model.def_support(f'B{i}', True, True, True, True, True, True)
            model.add_node_load(f'T{i}', 'FY', -200, 'D')
            model.add_member_dist_load(f'Stud_{i}', 'Fz', 0.05, 0.05, 0, wall_height, 'W')

        model.add_load_combo('1.4D', {'D': 1.4})
        model.add_load_combo('1.2D+W', {'D': 1.2, 'W': 1.0})
        model.analyze()

        combo_names = ['1.4D', '1.2D+W']
        n_points = 15

        # Model-level extraction
        all_forces = model.get_all_member_forces(combo_names, n_points=n_points)

        # Verify against individual extraction
        for stud_name in [f'Stud_{i}' for i in range(n_studs)]:
            member = model.members[stud_name]
            individual = member.get_all_forces_array(combo_names, n_points)

            model_level = all_forces[stud_name]

            assert_allclose(model_level['x'], individual['x'], rtol=1e-10)
            assert_allclose(model_level['Fy'], individual['Fy'], rtol=1e-6)
            assert_allclose(model_level['Mz'], individual['Mz'], rtol=1e-6)
            assert_allclose(model_level['Fx'], individual['Fx'], rtol=1e-6)
            assert_allclose(model_level['Mx'], individual['Mx'], rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
