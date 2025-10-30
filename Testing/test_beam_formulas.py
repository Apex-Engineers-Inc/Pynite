"""
Test internal member forces and deflections against known beam formulas.
Reference: AWC Design Aid No. 6 - Beam Formulas with Shear and Moment Diagrams

All tests use consistent units:
- Forces: kips
- Lengths: feet
- Moments: kip-ft
- Stress: ksf (kips per square foot)
"""

from Pynite import FEModel3D
import pytest


def test_simple_beam_uniformly_distributed_load():
    """
    Figure 1: Simple Beam - Uniformly Distributed Load

    Tests:
    - R = V = wℓ/2
    - Vx = w(ℓ/2 - x)
    - Mmax (at center) = wℓ²/8
    - Mx = (wx/2)(ℓ - x)
    - Δmax (at center) = 5wℓ⁴/(384EI)
    """
    beam = FEModel3D()

    # Beam properties
    L = 10.0  # feet
    w = 0.5   # kips/ft

    # Define nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)

    # Simply supported
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', False, True, True, True, False, False)

    # Material properties
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    # Section properties
    I = 200/12**4  # ft^4
    beam.add_section('Section', 12/144, I, I, 400/12**4)

    # Create member and load
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')
    beam.add_member_dist_load('M1', 'FY', -w, -w, case='D')
    beam.add_load_combo('D', {'D': 1.0})

    beam.analyze_linear()

    # Test reactions: R = wℓ/2
    R_expected = w * L / 2
    assert beam.nodes['N1'].RxnFY['D'] == pytest.approx(R_expected, rel=0.01), \
        f'R1 = {beam.nodes["N1"].RxnFY["D"]}, expected {R_expected}'
    assert beam.nodes['N2'].RxnFY['D'] == pytest.approx(R_expected, rel=0.01), \
        f'R2 = {beam.nodes["N2"].RxnFY["D"]}, expected {R_expected}'

    # Test shear at multiple points: Vx = w(ℓ/2 - x)
    test_points = [0, L/8, L/4, 3*L/8, L/2, 5*L/8, 3*L/4, 7*L/8, L]
    for x in test_points:
        V_expected = w * (L/2 - x)
        V_actual = beam.members['M1'].shear('Fy', x, 'D')
        assert V_actual == pytest.approx(V_expected, abs=0.01), \
            f'Shear at x={x}: {V_actual}, expected {V_expected}'

    # Test moment at multiple points: Mx = (wx/2)(ℓ - x)
    moment_test_points = [0, L/8, L/4, 3*L/8, L/2, 5*L/8, 3*L/4, 7*L/8, L]
    for x in moment_test_points:
        M_expected = -(w * x / 2) * (L - x)  # Negative for downward load
        M_actual = beam.members['M1'].moment('Mz', x, 'D')
        assert M_actual == pytest.approx(M_expected, rel=0.01), \
            f'Moment at x={x}: {M_actual}, expected {M_expected}'

    # Verify max moment is at center: Mmax = wℓ²/8
    M_max_expected = w * L**2 / 8
    M_at_center = beam.members['M1'].moment('Mz', L/2, 'D')
    assert abs(M_at_center) == pytest.approx(M_max_expected, rel=0.01), \
        f'Max moment at center: {M_at_center}, expected {-M_max_expected}'

    # Test deflection at multiple points using exact beam formula
    # For simply supported beam with UDL: δ(x) = (wx/24EI)(ℓ³ - 2ℓx² + x³)
    deflection_test_points = [L/8, L/4, 3*L/8, L/2, 5*L/8, 3*L/4, 7*L/8]
    for x in deflection_test_points:
        delta_expected = -(w * x / (24 * E * I)) * (L**3 - 2*L*x**2 + x**3)
        delta_actual = beam.members['M1'].deflection('dy', x, 'D')
        assert delta_actual == pytest.approx(delta_expected, rel=0.01), \
            f'Deflection at x={x}: {delta_actual}, expected {delta_expected}'

    # Verify max deflection at center: Δmax = 5wℓ⁴/(384EI)
    delta_max_expected = 5 * w * L**4 / (384 * E * I)
    delta_at_center = beam.members['M1'].deflection('dy', L/2, 'D')
    assert delta_at_center == pytest.approx(-delta_max_expected, rel=0.01), \
        f'Max deflection at center: {delta_at_center}, expected {-delta_max_expected}'


def test_simple_beam_concentrated_load_at_center():
    """
    Figure 7: Simple Beam - Concentrated Load at Center

    Tests:
    - R = V = P/2
    - Mmax (at point of load) = Pℓ/4
    - Mx (when x < ℓ/2) = Px/2
    - Δmax (at point of load) = Pℓ³/(48EI)
    - Δx (when x < ℓ/2) = (Px/48EI)(3ℓ² - 4x²)
    """
    beam = FEModel3D()

    # Beam properties
    L = 12.0  # feet
    P = 10.0  # kips

    # Define nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)

    # Simply supported
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', False, True, True, True, False, False)

    # Material properties
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    # Section properties
    I = 300/12**4  # ft^4
    beam.add_section('Section', 15/144, I, I, 600/12**4)

    # Create member and load
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')
    beam.add_member_pt_load('M1', 'FY', -P, L/2, case='L')
    beam.add_load_combo('L', {'L': 1.0})

    beam.analyze_linear()

    # Test reactions: R = P/2
    R_expected = P / 2
    assert beam.nodes['N1'].RxnFY['L'] == pytest.approx(R_expected, rel=0.01), \
        f'R1 = {beam.nodes["N1"].RxnFY["L"]}, expected {R_expected}'
    assert beam.nodes['N2'].RxnFY['L'] == pytest.approx(R_expected, rel=0.01), \
        f'R2 = {beam.nodes["N2"].RxnFY["L"]}, expected {R_expected}'

    # Test shear at multiple points before and after load
    # For x < ℓ/2: V = P/2 (constant)
    # For x > ℓ/2: V = -P/2 (constant)
    shear_test_points_before = [0, L/8, L/4, 3*L/8]
    for x in shear_test_points_before:
        V_expected = P / 2
        V_actual = beam.members['M1'].shear('Fy', x, 'L')
        assert V_actual == pytest.approx(V_expected, abs=0.01), \
            f'Shear at x={x}: {V_actual}, expected {V_expected}'

    shear_test_points_after = [5*L/8, 3*L/4, 7*L/8, L - 0.01]
    for x in shear_test_points_after:
        V_expected = -P / 2
        V_actual = beam.members['M1'].shear('Fy', x, 'L')
        assert V_actual == pytest.approx(V_expected, abs=0.01), \
            f'Shear at x={x}: {V_actual}, expected {V_expected}'

    # Test moment at multiple points: Mx = Px/2 for x < ℓ/2
    moment_test_points = [L/8, L/4, 3*L/8]
    for x in moment_test_points:
        M_expected = -P * x / 2  # Negative for downward load
        M_actual = beam.members['M1'].moment('Mz', x, 'L')
        assert M_actual == pytest.approx(M_expected, rel=0.01), \
            f'Moment at x={x}: {M_actual}, expected {M_expected}'

    # Test moment at symmetrical points: Mx = P(ℓ-x)/2 for x > ℓ/2
    moment_test_points_right = [5*L/8, 3*L/4, 7*L/8]
    for x in moment_test_points_right:
        M_expected = -P * (L - x) / 2  # By symmetry
        M_actual = beam.members['M1'].moment('Mz', x, 'L')
        assert M_actual == pytest.approx(M_expected, rel=0.01), \
            f'Moment at x={x}: {M_actual}, expected {M_expected}'

    # Verify max moment at center: Mmax = Pℓ/4
    M_max_expected = P * L / 4
    M_at_center = beam.members['M1'].moment('Mz', L/2, 'L')
    assert abs(M_at_center) == pytest.approx(M_max_expected, rel=0.01), \
        f'Max moment at center: {M_at_center}, expected {-M_max_expected}'

    # Test deflection at multiple points: Δx = (Px/48EI)(3ℓ² - 4x²) for x < ℓ/2
    deflection_test_points = [L/8, L/4, 3*L/8]
    for x in deflection_test_points:
        delta_expected = -(P * x / (48 * E * I)) * (3 * L**2 - 4 * x**2)
        delta_actual = beam.members['M1'].deflection('dy', x, 'L')
        assert delta_actual == pytest.approx(delta_expected, rel=0.01), \
            f'Deflection at x={x}: {delta_actual}, expected {delta_expected}'

    # Test deflection at symmetrical points (right side)
    deflection_test_points_right = [5*L/8, 3*L/4, 7*L/8]
    for x in deflection_test_points_right:
        x_mirror = L - x
        delta_expected = -(P * x_mirror / (48 * E * I)) * (3 * L**2 - 4 * x_mirror**2)
        delta_actual = beam.members['M1'].deflection('dy', x, 'L')
        assert delta_actual == pytest.approx(delta_expected, rel=0.01), \
            f'Deflection at x={x}: {delta_actual}, expected {delta_expected}'

    # Verify max deflection at center: Δmax = Pℓ³/(48EI)
    delta_max_expected = P * L**3 / (48 * E * I)
    delta_at_center = beam.members['M1'].deflection('dy', L/2, 'L')
    assert delta_at_center == pytest.approx(-delta_max_expected, rel=0.01), \
        f'Max deflection at center: {delta_at_center}, expected {-delta_max_expected}'


def test_simple_beam_concentrated_load_at_any_point():
    """
    Figure 8: Simple Beam - Concentrated Load at Any Point

    For a load at distance 'a' from left support:
    - R1 = Pb/ℓ (max when a < b)
    - R2 = Pa/ℓ (max when a > b)
    - Mmax (at point of load) = Pab/ℓ
    - Mx (when x < a) = Pbx/ℓ
    """
    beam = FEModel3D()

    # Beam properties
    L = 15.0  # feet
    P = 8.0   # kips
    a = 6.0   # feet from left support
    b = L - a # feet from right support

    # Define nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)

    # Simply supported
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', False, True, True, True, False, False)

    # Material properties
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    # Section properties
    I = 250/12**4  # ft^4
    beam.add_section('Section', 12/144, I, I, 500/12**4)

    # Create member and load
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')
    beam.add_member_pt_load('M1', 'FY', -P, a, case='D')
    beam.add_load_combo('D', {'D': 1.0})

    beam.analyze_linear()

    # Test reactions
    R1_expected = P * b / L
    R2_expected = P * a / L
    assert beam.nodes['N1'].RxnFY['D'] == pytest.approx(R1_expected, rel=0.01), \
        f'R1 = {beam.nodes["N1"].RxnFY["D"]}, expected {R1_expected}'
    assert beam.nodes['N2'].RxnFY['D'] == pytest.approx(R2_expected, rel=0.01), \
        f'R2 = {beam.nodes["N2"].RxnFY["D"]}, expected {R2_expected}'

    # Test shear at multiple points before load: V = R1 = Pb/ℓ (constant for x < a)
    shear_test_points_before = [0, a/4, a/2, 3*a/4]
    for x in shear_test_points_before:
        V_expected = R1_expected
        V_actual = beam.members['M1'].shear('Fy', x, 'D')
        assert V_actual == pytest.approx(V_expected, abs=0.01), \
            f'Shear at x={x}: {V_actual}, expected {V_expected}'

    # Test shear at multiple points after load: V = -R2 = -Pa/ℓ (constant for x > a)
    shear_test_points_after = [a + b/4, a + b/2, a + 3*b/4]
    for x in shear_test_points_after:
        V_expected = -R2_expected
        V_actual = beam.members['M1'].shear('Fy', x, 'D')
        assert V_actual == pytest.approx(V_expected, abs=0.01), \
            f'Shear at x={x}: {V_actual}, expected {V_expected}'

    # Test moment at multiple points before load: Mx = Pbx/ℓ for x < a
    moment_test_points_before = [a/4, a/2, 3*a/4]
    for x in moment_test_points_before:
        M_expected = -P * b * x / L  # Negative for downward load
        M_actual = beam.members['M1'].moment('Mz', x, 'D')
        assert M_actual == pytest.approx(M_expected, rel=0.01), \
            f'Moment at x={x} (before load): {M_actual}, expected {M_expected}'

    # Test moment at multiple points after load: Mx = P(L-x)a/ℓ for x > a
    moment_test_points_after = [a + b/4, a + b/2, a + 3*b/4]
    for x in moment_test_points_after:
        M_expected = -P * a * (L - x) / L  # Negative for downward load
        M_actual = beam.members['M1'].moment('Mz', x, 'D')
        assert M_actual == pytest.approx(M_expected, rel=0.01), \
            f'Moment at x={x} (after load): {M_actual}, expected {M_expected}'

    # Verify max moment at load point: Mmax = Pab/ℓ
    M_max_expected = P * a * b / L
    M_at_load = beam.members['M1'].moment('Mz', a, 'D')
    assert abs(M_at_load) == pytest.approx(M_max_expected, rel=0.01), \
        f'Max moment at load point: {M_at_load}, expected {-M_max_expected}'


def test_simple_beam_two_equal_concentrated_loads_symmetrical():
    """
    Figure 9: Simple Beam - Two Equal Concentrated Loads Symmetrically Placed

    Tests:
    - R = V = P (total load is 2P)
    - Mmax (between loads) = Pa
    - Mx (when x < a) = Px
    - Δmax (at center) = Pa/(24EI)(3ℓ² - 4a²)
    - Δx (when x < a) = Px/(6EI)(3ℓa - 3a² - x²)
    """
    beam = FEModel3D()

    # Beam properties
    L = 20.0  # feet
    P = 5.0   # kips (per load)
    a = 5.0   # feet from each support to load

    # Define nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)

    # Simply supported
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', False, True, True, True, False, False)

    # Material properties
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    # Section properties
    I = 350/12**4  # ft^4
    beam.add_section('Section', 18/144, I, I, 700/12**4)

    # Create member and loads
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')
    beam.add_member_pt_load('M1', 'FY', -P, a, case='L')
    beam.add_member_pt_load('M1', 'FY', -P, L-a, case='L')
    beam.add_load_combo('L', {'L': 1.0})

    beam.analyze_linear()

    # Test reactions: R = P (each reaction carries one load)
    R_expected = P
    assert beam.nodes['N1'].RxnFY['L'] == pytest.approx(R_expected, rel=0.01), \
        f'R1 = {beam.nodes["N1"].RxnFY["L"]}, expected {R_expected}'
    assert beam.nodes['N2'].RxnFY['L'] == pytest.approx(R_expected, rel=0.01), \
        f'R2 = {beam.nodes["N2"].RxnFY["L"]}, expected {R_expected}'

    # Test max moment between loads: Mmax = Pa
    M_max_expected = P * a
    M_actual = beam.members['M1'].moment('Mz', L/2, 'L')
    assert abs(M_actual) == pytest.approx(M_max_expected, rel=0.01), \
        f'Max moment at center: {M_actual}, expected {-M_max_expected}'

    # Test moment at first load point should also be Pa
    M_at_load = beam.members['M1'].moment('Mz', a, 'L')
    assert abs(M_at_load) == pytest.approx(M_max_expected, rel=0.01), \
        f'Moment at load point: {M_at_load}, expected {-M_max_expected}'

    # Test moment at x < a: Mx = Px
    x = a / 2
    M_expected = -P * x  # Negative for downward load
    M_actual = beam.members['M1'].moment('Mz', x, 'L')
    assert M_actual == pytest.approx(M_expected, rel=0.01), \
        f'Moment at x={x}: {M_actual}, expected {M_expected}'

    # Test max deflection at center: Δmax = Pa/(24EI)(3ℓ² - 4a²)
    delta_max_expected = (P * a / (24 * E * I)) * (3 * L**2 - 4 * a**2)
    delta_actual = beam.members['M1'].deflection('dy', L/2, 'L')
    assert delta_actual == pytest.approx(-delta_max_expected, rel=0.01), \
        f'Max deflection: {delta_actual}, expected {-delta_max_expected}'


def test_cantilever_beam_uniformly_distributed_load():
    """
    Figure 12: Cantilever Beam - Uniformly Distributed Load

    Tests:
    - R = V = wℓ
    - Vx = wx
    - Mmax (at fixed end) = wℓ²/2
    - Mx = wx²/2
    - Δmax (at free end) = wℓ⁴/(8EI)
    - Δx = w/(24EI)(x⁴ - 4ℓ³x + 3ℓ⁴)
    """
    beam = FEModel3D()

    # Beam properties
    L = 8.0   # feet
    w = 1.0   # kips/ft

    # Define nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)

    # Fixed at left, free at right
    beam.def_support('N1', True, True, True, True, True, True)
    beam.def_support('N2', False, False, False, False, False, False)

    # Material properties
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    # Section properties
    I = 180/12**4  # ft^4
    beam.add_section('Section', 10/144, I, I, 360/12**4)

    # Create member and load
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')
    beam.add_member_dist_load('M1', 'FY', -w, -w, case='D')
    beam.add_load_combo('D', {'D': 1.0})

    beam.analyze_linear()

    # Test reaction: R = wℓ
    R_expected = w * L
    assert beam.nodes['N1'].RxnFY['D'] == pytest.approx(R_expected, rel=0.01), \
        f'Reaction: {beam.nodes["N1"].RxnFY["D"]}, expected {R_expected}'

    # Test shear at multiple points: V(x) = w(ℓ - x)
    # Shear decreases linearly from R at fixed end to 0 at free end
    shear_test_points = [0, L/8, L/4, 3*L/8, L/2, 5*L/8, 3*L/4, 7*L/8, L]
    for x in shear_test_points:
        V_expected = w * (L - x)
        V_actual = beam.members['M1'].shear('Fy', x, 'D')
        assert V_actual == pytest.approx(V_expected, abs=0.01), \
            f'Shear at x={x}: {V_actual}, expected {V_expected}'

    # Test moment at multiple points: M(x) = w(L-x)²/2
    # Moment increases parabolically from 0 at free end to max at fixed end
    moment_test_points = [L/8, L/4, 3*L/8, L/2, 5*L/8, 3*L/4, 7*L/8, L]
    for x in moment_test_points:
        M_expected = w * (L - x)**2 / 2
        M_actual = beam.members['M1'].moment('Mz', x, 'D')
        assert abs(M_actual) == pytest.approx(M_expected, rel=0.01), \
            f'Moment at x={x}: {abs(M_actual)}, expected {M_expected}'

    # Verify max moment at fixed end: Mmax = wℓ²/2
    M_max_expected = w * L**2 / 2
    M_at_fixed = beam.members['M1'].moment('Mz', 0, 'D')
    assert abs(M_at_fixed) == pytest.approx(M_max_expected, rel=0.01), \
        f'Max moment at fixed end: {abs(M_at_fixed)}, expected {M_max_expected}'

    # Test deflection at multiple points: δ(ξ) = w/(24EI)(ξ⁴ - 4ℓ³ξ + 3ℓ⁴)
    # where ξ is measured from the FREE end, so ξ = ℓ - x
    deflection_test_points = [L/8, L/4, 3*L/8, L/2, 5*L/8, 3*L/4, 7*L/8]
    for x in deflection_test_points:
        xi = L - x  # Distance from free end
        delta_expected = -(w / (24 * E * I)) * (xi**4 - 4*L**3*xi + 3*L**4)
        delta_actual = beam.members['M1'].deflection('dy', x, 'D')
        assert delta_actual == pytest.approx(delta_expected, rel=0.01), \
            f'Deflection at x={x}: {delta_actual}, expected {delta_expected}'

    # Verify max deflection at free end: Δmax = wℓ⁴/(8EI)
    delta_max_expected = w * L**4 / (8 * E * I)
    delta_at_free = beam.members['M1'].deflection('dy', L, 'D')
    assert delta_at_free == pytest.approx(-delta_max_expected, rel=0.01), \
        f'Max deflection at free end: {delta_at_free}, expected {-delta_max_expected}'


def test_cantilever_beam_concentrated_load_at_free_end():
    """
    Figure 13: Cantilever Beam - Concentrated Load at Free End

    Tests:
    - R = V = P
    - Mmax (at fixed end) = Pℓ
    - Mx = Px
    - Δmax (at free end) = Pℓ³/(3EI)
    - Δx = P/(6EI)(2ℓ³ - 3ℓ²x + x³)
    """
    beam = FEModel3D()

    # Beam properties
    L = 10.0  # feet
    P = 6.0   # kips

    # Define nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)

    # Fixed at left, free at right
    beam.def_support('N1', True, True, True, True, True, True)
    beam.def_support('N2', False, False, False, False, False, False)

    # Material properties
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    # Section properties
    I = 220/12**4  # ft^4
    beam.add_section('Section', 12/144, I, I, 440/12**4)

    # Create member and load
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')
    beam.add_member_pt_load('M1', 'FY', -P, L, case='L')
    beam.add_load_combo('L', {'L': 1.0})

    beam.analyze_linear()

    # Test reaction: R = P
    assert beam.nodes['N1'].RxnFY['L'] == pytest.approx(P, rel=0.01), \
        f'Reaction: {beam.nodes["N1"].RxnFY["L"]}, expected {P}'

    # Test shear at multiple points: V = P everywhere (constant)
    # For cantilever with end load, shear is constant throughout
    shear_test_points = [0, L/8, L/4, 3*L/8, L/2, 5*L/8, 3*L/4, 7*L/8]
    for x in shear_test_points:
        V_actual = beam.members['M1'].shear('Fy', x, 'L')
        assert abs(V_actual) == pytest.approx(P, abs=0.01), \
            f'Shear at x={x}: {abs(V_actual)}, expected {P}'

    # Test moment at multiple points: M(x) = P(L - x)
    # Moment increases linearly from 0 at free end to max at fixed end
    moment_test_points = [L/8, L/4, 3*L/8, L/2, 5*L/8, 3*L/4, 7*L/8, L]
    for x in moment_test_points:
        M_expected = P * (L - x)
        M_actual = beam.members['M1'].moment('Mz', x, 'L')
        assert abs(M_actual) == pytest.approx(M_expected, rel=0.01), \
            f'Moment at x={x}: {abs(M_actual)}, expected {M_expected}'

    # Verify max moment at fixed end: Mmax = Pℓ
    M_max_expected = P * L
    M_at_fixed = beam.members['M1'].moment('Mz', 0, 'L')
    assert abs(M_at_fixed) == pytest.approx(M_max_expected, rel=0.01), \
        f'Max moment at fixed end: {abs(M_at_fixed)}, expected {M_max_expected}'

    # Test deflection at multiple points: δ(ξ) = P/(6EI)(2ℓ³ - 3ℓ²ξ + ξ³)
    # where ξ is measured from the FREE end, so ξ = ℓ - x
    deflection_test_points = [L/8, L/4, 3*L/8, L/2, 5*L/8, 3*L/4, 7*L/8]
    for x in deflection_test_points:
        xi = L - x  # Distance from free end
        delta_expected = -(P / (6 * E * I)) * (2*L**3 - 3*L**2*xi + xi**3)
        delta_actual = beam.members['M1'].deflection('dy', x, 'L')
        assert delta_actual == pytest.approx(delta_expected, rel=0.01), \
            f'Deflection at x={x}: {delta_actual}, expected {delta_expected}'

    # Verify max deflection at free end: Δmax = Pℓ³/(3EI)
    delta_max_expected = P * L**3 / (3 * E * I)
    delta_at_free = beam.members['M1'].deflection('dy', L, 'L')
    assert delta_at_free == pytest.approx(-delta_max_expected, rel=0.01), \
        f'Max deflection at free end: {delta_at_free}, expected {-delta_max_expected}'


def test_beam_overhanging_one_support_uniform_load():
    """
    Figure 18: Beam Overhanging One Support - Uniformly Distributed Load

    For overhang length 'a':
    - R1 = w/(2ℓ)(ℓ² - a²)
    - R2 = w/(2ℓ)(ℓ + a)²
    - M1 (at x = ℓ - a²/ℓ²) = w/(8ℓ²)(ℓ + a)²(ℓ - a)²
    - M2 (at R2) = wa²/2
    """
    beam = FEModel3D()

    # Beam properties
    L = 12.0  # feet (main span)
    a = 3.0   # feet (overhang)
    w = 0.8   # kips/ft

    # Define nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)
    beam.add_node('N3', L + a, 0, 0)

    # Simply supported at N1 and N2
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', False, True, True, True, False, False)

    # Material properties
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    # Section properties
    I = 280/12**4  # ft^4
    beam.add_section('Section', 14/144, I, I, 560/12**4)

    # Create members
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')
    beam.add_member('M2', 'N2', 'N3', 'Steel', 'Section')

    # Add uniform load over entire length
    beam.add_member_dist_load('M1', 'FY', -w, -w, case='D')
    beam.add_member_dist_load('M2', 'FY', -w, -w, case='D')
    beam.add_load_combo('D', {'D': 1.0})

    beam.analyze_linear()

    # Test reactions
    R1_expected = w * (L**2 - a**2) / (2 * L)
    R2_expected = w * (L + a)**2 / (2 * L)

    assert beam.nodes['N1'].RxnFY['D'] == pytest.approx(R1_expected, rel=0.02), \
        f'R1 = {beam.nodes["N1"].RxnFY["D"]}, expected {R1_expected}'
    assert beam.nodes['N2'].RxnFY['D'] == pytest.approx(R2_expected, rel=0.02), \
        f'R2 = {beam.nodes["N2"].RxnFY["D"]}, expected {R2_expected}'

    # Test moment at support R2: M2 = wa²/2 (magnitude)
    # The moment will be opposite sign convention due to overhang
    M2_expected = w * a**2 / 2
    M_at_R2 = beam.members['M2'].moment('Mz', 0, 'D')  # At start of overhang member
    assert abs(M_at_R2) == pytest.approx(M2_expected, abs=0.5), \
        f'Moment at R2: {abs(M_at_R2)}, expected {M2_expected}'


def test_fixed_beam_uniformly_distributed_load():
    """
    Figure 23: Beam Fixed at Both Ends - Uniformly Distributed Load

    Tests:
    - R = V = wℓ/2
    - Vx = w(ℓ/2 - x)
    - Mmax (at ends) = wℓ²/12
    - M1 (at center) = wℓ²/24
    - Mx = w/12(6ℓx - ℓ² - 6x²)
    - Δmax (at center) = wℓ⁴/(384EI)
    """
    beam = FEModel3D()

    # Beam properties
    L = 14.0  # feet
    w = 1.2   # kips/ft

    # Define nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)

    # Fixed at both ends
    beam.def_support('N1', True, True, True, True, True, True)
    beam.def_support('N2', True, True, True, True, True, True)

    # Material properties
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    # Section properties
    I = 400/12**4  # ft^4
    beam.add_section('Section', 20/144, I, I, 800/12**4)

    # Create member and load
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')
    beam.add_member_dist_load('M1', 'FY', -w, -w, case='D')
    beam.add_load_combo('D', {'D': 1.0})

    beam.analyze_linear()

    # Test reactions: R = wℓ/2
    R_expected = w * L / 2
    assert beam.nodes['N1'].RxnFY['D'] == pytest.approx(R_expected, rel=0.01), \
        f'R1 = {beam.nodes["N1"].RxnFY["D"]}, expected {R_expected}'
    assert beam.nodes['N2'].RxnFY['D'] == pytest.approx(R_expected, rel=0.01), \
        f'R2 = {beam.nodes["N2"].RxnFY["D"]}, expected {R_expected}'

    # Test shear at multiple points: V(x) = w(ℓ/2 - x)
    # Same as simply supported, but with different moment distribution
    shear_test_points = [L/8, L/4, 3*L/8, L/2, 5*L/8, 3*L/4, 7*L/8]
    for x in shear_test_points:
        V_expected = w * (L/2 - x)
        V_actual = beam.members['M1'].shear('Fy', x, 'D')
        assert V_actual == pytest.approx(V_expected, abs=0.01), \
            f'Shear at x={x}: {V_actual}, expected {V_expected}'

    # Test moment at multiple points: M(x) = w/12(6ℓx - ℓ² - 6x²)
    # For fixed-fixed beam with UDL
    moment_test_points = [L/8, L/4, 3*L/8, L/2, 5*L/8, 3*L/4, 7*L/8]
    for x in moment_test_points:
        M_expected = -(w / 12) * (6*L*x - L**2 - 6*x**2)
        M_actual = beam.members['M1'].moment('Mz', x, 'D')
        assert M_actual == pytest.approx(M_expected, rel=0.02), \
            f'Moment at x={x}: {M_actual}, expected {M_expected}'

    # Verify moment at fixed ends: Mmax = wℓ²/12 (hogging at supports)
    M_end_expected = w * L**2 / 12
    M_at_start = beam.members['M1'].moment('Mz', 0.01, 'D')
    M_at_end = beam.members['M1'].moment('Mz', L - 0.01, 'D')
    assert abs(M_at_start) == pytest.approx(M_end_expected, rel=0.05), \
        f'Moment at start: {abs(M_at_start)}, expected {M_end_expected}'
    assert abs(M_at_end) == pytest.approx(M_end_expected, rel=0.05), \
        f'Moment at end: {abs(M_at_end)}, expected {M_end_expected}'

    # Verify moment at center: M_center = wℓ²/24 (sagging)
    M_center_expected = w * L**2 / 24
    M_at_center = beam.members['M1'].moment('Mz', L/2, 'D')
    assert abs(M_at_center) == pytest.approx(M_center_expected, rel=0.05), \
        f'Moment at center: {abs(M_at_center)}, expected {M_center_expected}'

    # Test deflection at multiple points
    # For fixed-fixed beam: δ(x) = wx²(ℓ-x)²/(24EI)
    deflection_test_points = [L/8, L/4, 3*L/8, L/2, 5*L/8, 3*L/4, 7*L/8]
    for x in deflection_test_points:
        delta_expected = -(w * x**2 * (L - x)**2) / (24 * E * I)
        delta_actual = beam.members['M1'].deflection('dy', x, 'D')
        assert delta_actual == pytest.approx(delta_expected, rel=0.02), \
            f'Deflection at x={x}: {delta_actual}, expected {delta_expected}'

    # Verify max deflection at center: Δmax = wℓ⁴/(384EI)
    delta_max_expected = w * L**4 / (384 * E * I)
    delta_at_center = beam.members['M1'].deflection('dy', L/2, 'D')
    assert delta_at_center == pytest.approx(-delta_max_expected, rel=0.01), \
        f'Max deflection at center: {delta_at_center}, expected {-delta_max_expected}'


def test_fixed_beam_concentrated_load_at_center():
    """
    Figure 24: Beam Fixed at Both Ends - Concentrated Load at Center

    Tests:
    - R = V = P/2
    - Mmax (at center and ends) = Pℓ/8
    - Mx (when x < ℓ/2) = P/8(4x - ℓ)
    - Δmax (at center) = Pℓ³/(192EI)
    - Δx (when x < ℓ/2) = Px²/(48EI)(3ℓ - 4x)
    """
    beam = FEModel3D()

    # Beam properties
    L = 16.0  # feet
    P = 12.0  # kips

    # Define nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)

    # Fixed at both ends
    beam.def_support('N1', True, True, True, True, True, True)
    beam.def_support('N2', True, True, True, True, True, True)

    # Material properties
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    # Section properties
    I = 450/12**4  # ft^4
    beam.add_section('Section', 22/144, I, I, 900/12**4)

    # Create member and load
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')
    beam.add_member_pt_load('M1', 'FY', -P, L/2, case='L')
    beam.add_load_combo('L', {'L': 1.0})

    beam.analyze_linear()

    # Test reactions: R = P/2
    R_expected = P / 2
    assert beam.nodes['N1'].RxnFY['L'] == pytest.approx(R_expected, rel=0.01), \
        f'R1 = {beam.nodes["N1"].RxnFY["L"]}, expected {R_expected}'
    assert beam.nodes['N2'].RxnFY['L'] == pytest.approx(R_expected, rel=0.01), \
        f'R2 = {beam.nodes["N2"].RxnFY["L"]}, expected {R_expected}'

    # Test shear at multiple points before and after load
    # For x < ℓ/2: V = P/2 (constant)
    # For x > ℓ/2: V = -P/2 (constant)
    shear_test_points_before = [L/8, L/4, 3*L/8]
    for x in shear_test_points_before:
        V_expected = P / 2
        V_actual = beam.members['M1'].shear('Fy', x, 'L')
        assert V_actual == pytest.approx(V_expected, abs=0.01), \
            f'Shear at x={x}: {V_actual}, expected {V_expected}'

    shear_test_points_after = [5*L/8, 3*L/4, 7*L/8]
    for x in shear_test_points_after:
        V_expected = -P / 2
        V_actual = beam.members['M1'].shear('Fy', x, 'L')
        assert V_actual == pytest.approx(V_expected, abs=0.01), \
            f'Shear at x={x}: {V_actual}, expected {V_expected}'

    # Test moment at multiple points: M(x) = P/8(4x - ℓ) for x < ℓ/2
    moment_test_points_before = [L/8, L/4, 3*L/8]
    for x in moment_test_points_before:
        M_expected = -(P / 8) * (4*x - L)
        M_actual = beam.members['M1'].moment('Mz', x, 'L')
        assert M_actual == pytest.approx(M_expected, rel=0.02), \
            f'Moment at x={x}: {M_actual}, expected {M_expected}'

    # Test moment at symmetrical points: M(x) = P/8(3ℓ - 4x) for x > ℓ/2
    moment_test_points_after = [5*L/8, 3*L/4, 7*L/8]
    for x in moment_test_points_after:
        M_expected = -(P / 8) * (3*L - 4*x)
        M_actual = beam.members['M1'].moment('Mz', x, 'L')
        assert M_actual == pytest.approx(M_expected, rel=0.02), \
            f'Moment at x={x}: {M_actual}, expected {M_expected}'

    # Verify moment at ends and center: Mmax = Pℓ/8
    M_expected = P * L / 8
    M_at_center = beam.members['M1'].moment('Mz', L/2, 'L')
    assert abs(M_at_center) == pytest.approx(M_expected, rel=0.05), \
        f'Moment at center: {abs(M_at_center)}, expected {M_expected}'

    # Test deflection at multiple points: δ(x) = Px²/(48EI)(3ℓ - 4x) for x < ℓ/2
    deflection_test_points_before = [L/8, L/4, 3*L/8]
    for x in deflection_test_points_before:
        delta_expected = -(P * x**2 / (48 * E * I)) * (3*L - 4*x)
        delta_actual = beam.members['M1'].deflection('dy', x, 'L')
        assert delta_actual == pytest.approx(delta_expected, rel=0.02), \
            f'Deflection at x={x}: {delta_actual}, expected {delta_expected}'

    # Test deflection at symmetrical points (right side)
    deflection_test_points_after = [5*L/8, 3*L/4, 7*L/8]
    for x in deflection_test_points_after:
        x_mirror = L - x
        delta_expected = -(P * x_mirror**2 / (48 * E * I)) * (3*L - 4*x_mirror)
        delta_actual = beam.members['M1'].deflection('dy', x, 'L')
        assert delta_actual == pytest.approx(delta_expected, rel=0.02), \
            f'Deflection at x={x}: {delta_actual}, expected {delta_expected}'

    # Verify max deflection at center: Δmax = Pℓ³/(192EI)
    delta_max_expected = P * L**3 / (192 * E * I)
    delta_at_center = beam.members['M1'].deflection('dy', L/2, 'L')
    assert delta_at_center == pytest.approx(-delta_max_expected, rel=0.01), \
        f'Max deflection at center: {delta_at_center}, expected {-delta_max_expected}'


def test_uniform_load_partially_distributed():
    """
    Figure 2: Simple Beam - Uniform Load Partially Distributed

    For a uniform load w over length b, starting at distance a from left support:
    - R1 = wb/(2ℓ)(2c + b)
    - R2 = wb/(2ℓ)(2a + b)
    - Mmax (at x = a + R1/w) = R1(a + R1/(2w))
    """
    beam = FEModel3D()

    # Beam properties
    L = 18.0  # feet
    w = 0.6   # kips/ft
    a = 4.0   # feet from left to start of load
    b = 8.0   # feet length of distributed load
    c = L - a - b  # feet from end of load to right support

    # Define nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)

    # Simply supported
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', False, True, True, True, False, False)

    # Material properties
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    # Section properties
    I = 320/12**4  # ft^4
    beam.add_section('Section', 16/144, I, I, 640/12**4)

    # Create member and load
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')
    beam.add_member_dist_load('M1', 'FY', -w, -w, a, a + b, case='D')
    beam.add_load_combo('D', {'D': 1.0})

    beam.analyze_linear()

    # Test reactions
    R1_expected = w * b * (2 * c + b) / (2 * L)
    R2_expected = w * b * (2 * a + b) / (2 * L)

    assert beam.nodes['N1'].RxnFY['D'] == pytest.approx(R1_expected, rel=0.02), \
        f'R1 = {beam.nodes["N1"].RxnFY["D"]}, expected {R1_expected}'
    assert beam.nodes['N2'].RxnFY['D'] == pytest.approx(R2_expected, rel=0.02), \
        f'R2 = {beam.nodes["N2"].RxnFY["D"]}, expected {R2_expected}'

    # Verify total load is carried
    total_load = w * b
    total_reaction = beam.nodes['N1'].RxnFY['D'] + beam.nodes['N2'].RxnFY['D']
    assert total_reaction == pytest.approx(total_load, rel=0.01), \
        f'Total reaction {total_reaction} should equal total load {total_load}'


def test_load_increasing_uniformly_to_one_end():
    """
    Figure 5: Simple Beam - Load Increasing Uniformly to One End

    For a triangular load with max intensity W at right end:
    - R1 = W/3
    - R2 = 2W/3
    - Mmax (at x = ℓ/√3 = 0.5774ℓ) = 2Wℓ/(9√3) = 0.1283Wℓ
    """
    beam = FEModel3D()

    # Beam properties
    L = 10.0  # feet
    # W is the total load from triangular distribution: W = (1/2) * w_max * L
    # where w_max is the intensity at the right end
    # For formulas: W represents total load, w represents load intensity per unit length
    w_max = 1.5  # kips/ft (intensity at right end)
    W = 0.5 * w_max * L  # Total load from triangle

    # Define nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)

    # Simply supported
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', False, True, True, True, False, False)

    # Material properties
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    # Section properties
    I = 240/12**4  # ft^4
    beam.add_section('Section', 13/144, I, I, 480/12**4)

    # Create member and load (triangular: 0 at start, w_max at end)
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')
    beam.add_member_dist_load('M1', 'FY', 0, -w_max, case='D')
    beam.add_load_combo('D', {'D': 1.0})

    beam.analyze_linear()

    # Test reactions
    R1_expected = W / 3
    R2_expected = 2 * W / 3

    assert beam.nodes['N1'].RxnFY['D'] == pytest.approx(R1_expected, rel=0.02), \
        f'R1 = {beam.nodes["N1"].RxnFY["D"]}, expected {R1_expected}'
    assert beam.nodes['N2'].RxnFY['D'] == pytest.approx(R2_expected, rel=0.02), \
        f'R2 = {beam.nodes["N2"].RxnFY["D"]}, expected {R2_expected}'

    # Test approximate location and magnitude of max moment
    # According to formula, max moment occurs at x = 0.5774ℓ
    x_max = 0.5774 * L
    M_max_expected = -0.1283 * W * L  # Negative for downward load
    M_actual = beam.members['M1'].moment('Mz', x_max, 'D')

    assert M_actual == pytest.approx(M_max_expected, rel=0.05), \
        f'Max moment at x={x_max}: {M_actual}, expected {M_max_expected}'


def test_figure_29_two_equal_spans_udl():
    """
    AWC Figure 29: Continuous Beam - Two Equal Spans - Uniformly Distributed Load

    Configuration: |----ℓ----•----ℓ----|
                   w        w

    Formulas from AWC Design Aid No. 6:
    - R₁ = R₃ = 3wℓ/8
    - R₂ = 10wℓ/8 = 5wℓ/4
    - V₂ = Vₘₐₓ = 5wℓ/8
    - M₁ (at interior support R₂) = -wℓ²/8
    - M₂ (at 3ℓ/8 from R₁ and R₃) = 9wℓ²/128
    - Δₘₐₓ (at 0.4215ℓ from R₁ and R₃) = wℓ⁴/(185EI)
    """
    beam = FEModel3D()

    # Parameters
    L = 10.0  # feet per span (ℓ in AWC notation)
    w = 1.0   # kips/ft

    # Nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)
    beam.add_node('N3', 2*L, 0, 0)

    # Supports - simply supported at all three points
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', False, True, True, True, False, False)
    beam.def_support('N3', False, True, True, True, False, False)

    # Material and section
    E = 29000 * 144  # ksf (steel)
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    I = 200/12**4  # ft⁴
    A = 12/144     # ft²
    J = 400/12**4  # ft⁴
    beam.add_section('Section', A, I, I, J)

    # Member with UDL on both spans
    beam.add_member('M1', 'N1', 'N3', 'Steel', 'Section')
    beam.add_member_dist_load('M1', 'FY', -w, -w, case='D')
    beam.add_load_combo('D', {'D': 1.0})

    beam.analyze_linear()

    # Test reactions (AWC formulas)
    R1_expected = 3 * w * L / 8
    R2_expected = 10 * w * L / 8  # or 5*w*L/4
    R3_expected = 3 * w * L / 8

    assert beam.nodes['N1'].RxnFY['D'] == pytest.approx(R1_expected, rel=0.01), \
        f'R1 = {beam.nodes["N1"].RxnFY["D"]}, expected {R1_expected}'
    assert beam.nodes['N2'].RxnFY['D'] == pytest.approx(R2_expected, rel=0.01), \
        f'R2 = {beam.nodes["N2"].RxnFY["D"]}, expected {R2_expected}'
    assert beam.nodes['N3'].RxnFY['D'] == pytest.approx(R3_expected, rel=0.01), \
        f'R3 = {beam.nodes["N3"].RxnFY["D"]}, expected {R3_expected}'

    # Test moment at interior support (positive in Pynite = hogging)
    M1_expected = w * L**2 / 8
    M1_actual = beam.members['M1'].moment('Mz', L, 'D')
    assert M1_actual == pytest.approx(M1_expected, rel=0.01), \
        f'M at interior support: {M1_actual}, expected {M1_expected}'

    # Test max positive moment in spans (at 3ℓ/8 from end supports)
    # Note: Pynite uses negative for sagging moments (downward loads)
    # AWC Figure 29 shows "3ℓ/8" measured from R₁ and R₃ (the END supports)
    # Left span: 3ℓ/8 from N1 (R₁)
    # Right span: 3ℓ/8 from N3 (R₃), which is at 2L - 3L/8 = 13L/8 from N1
    x_max_left = 3 * L / 8
    x_max_right = 2*L - 3 * L / 8  # Measured from right end support
    M2_expected = 9 * w * L**2 / 128
    M2_actual_left = beam.members['M1'].moment('Mz', x_max_left, 'D')
    M2_actual_right = beam.members['M1'].moment('Mz', x_max_right, 'D')

    assert abs(M2_actual_left) == pytest.approx(M2_expected, rel=0.01), \
        f'M at 3ℓ/8 in left span: {abs(M2_actual_left)}, expected {M2_expected}'
    assert abs(M2_actual_right) == pytest.approx(M2_expected, rel=0.01), \
        f'M at 3ℓ/8 in right span: {abs(M2_actual_right)}, expected {M2_expected}'

    # Test shear at multiple locations
    # AWC Mₓ formula for when x < ℓ: Mₓ = (wx/16)(7ℓ - 8x)
    # Taking derivative: Vₓ = R₁ - wx
    test_x = L / 4
    V_expected = R1_expected - w * test_x
    V_actual = beam.members['M1'].shear('Fy', test_x, 'D')
    assert V_actual == pytest.approx(V_expected, abs=0.01), \
        f'Shear at x=ℓ/4: {V_actual}, expected {V_expected}'


def test_figure_30_two_equal_spans_two_point_loads():
    """
    AWC Figure 30: Continuous Beam - Two Equal Spans - Two Equal Concentrated Loads
                    Symmetrically Placed

    Configuration: |----ℓ----•----ℓ----|
                       P         P
                   (at a from each end)

    AWC diagram shows loads at distance 'a' from outer supports R₁ and R₃.
    The specific formulas given correspond to a = ℓ/2:
    - R₁ = R₃ = 5P/16
    - R₂ = 11P/8
    - V₂ = P - R₁ = 11P/16
    - M₁ (at interior support) = -3Pℓ/16
    - M₂ (at point of load) = 5Pℓ/32
    - Mₓ (when x < a) = R₁x
    """
    beam = FEModel3D()

    # Parameters
    L = 12.0  # feet per span (ℓ in AWC notation)
    P = 10.0  # kips
    a = L / 2  # Distance from outer supports to loads (AWC formulas use a = ℓ/2)

    # Nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)
    beam.add_node('N3', 2*L, 0, 0)

    # Supports
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', False, True, True, True, False, False)
    beam.def_support('N3', False, True, True, True, False, False)

    # Material and section
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    I = 350/12**4  # ft⁴
    beam.add_section('Section', 18/144, I, I, 700/12**4)

    # Member with point loads symmetrically placed
    # AWC Figure 30: loads are at distance 'a' from the outer supports R₁ and R₃
    # With a = ℓ/2, loads are at the quarter-points of the total length (2ℓ)
    # With L=12, a=6: loads at x=6 and x=18 from left
    beam.add_member('M1', 'N1', 'N3', 'Steel', 'Section')
    beam.add_member_pt_load('M1', 'FY', -P, a, case='L')            # At 'a' from R₁
    beam.add_member_pt_load('M1', 'FY', -P, 2*L - a, case='L')      # At 'a' from R₃
    beam.add_load_combo('L', {'L': 1.0})

    beam.analyze_linear()

    # Test reactions (AWC formulas for symmetrically placed loads)
    R1_expected = 5 * P / 16
    R2_expected = 11 * P / 8
    R3_expected = 5 * P / 16

    assert beam.nodes['N1'].RxnFY['L'] == pytest.approx(R1_expected, rel=0.02), \
        f'R1 = {beam.nodes["N1"].RxnFY["L"]}, expected {R1_expected}'
    assert beam.nodes['N2'].RxnFY['L'] == pytest.approx(R2_expected, rel=0.02), \
        f'R2 = {beam.nodes["N2"].RxnFY["L"]}, expected {R2_expected}'
    assert beam.nodes['N3'].RxnFY['L'] == pytest.approx(R3_expected, rel=0.02), \
        f'R3 = {beam.nodes["N3"].RxnFY["L"]}, expected {R3_expected}'

    # Test moment at interior support
    M1_expected = 3 * P * L / 16
    M1_actual = beam.members['M1'].moment('Mz', L, 'L')
    assert M1_actual == pytest.approx(M1_expected, rel=0.02), \
        f'M at interior support: {M1_actual}, expected {M1_expected}'

    # Test moment at point of load (max positive moment)
    # Note: Pynite uses negative for sagging moments
    M2_expected = 5 * P * L / 32
    M2_actual_left = beam.members['M1'].moment('Mz', a, 'L')
    M2_actual_right = beam.members['M1'].moment('Mz', 2*L - a, 'L')

    assert abs(M2_actual_left) == pytest.approx(M2_expected, rel=0.02), \
        f'M at load in left span: {abs(M2_actual_left)}, expected {M2_expected}'
    assert abs(M2_actual_right) == pytest.approx(M2_expected, rel=0.02), \
        f'M at load in right span: {abs(M2_actual_right)}, expected {M2_expected}'

    # Test shear before first load: V = R₁
    V_before = beam.members['M1'].shear('Fy', a/2, 'L')
    assert V_before == pytest.approx(R1_expected, abs=0.01), \
        f'Shear before load: {V_before}, expected {R1_expected}'

    # Test shear after first load: V = R₁ - P
    V_after = beam.members['M1'].shear('Fy', (a + L)/2, 'L')
    V_expected_after = R1_expected - P
    assert V_after == pytest.approx(V_expected_after, abs=0.01), \
        f'Shear after load: {V_after}, expected {V_expected_after}'


def test_figure_26_two_equal_spans_udl_on_one_span():
    """
    AWC Figure 26: Continuous Beam - Two Equal Spans - Uniform Load on One Span

    Configuration: |----ℓ----•----ℓ----|
                   w

    Load only on first span.

    Formulas from AWC Design Aid No. 6:
    - R₁ = 7wℓ/16
    - R₂ = 5wℓ/8
    - R₃ = -wℓ/16 (upward reaction)
    - M₁ (at support R₂) = -wℓ²/16
    - Mₘₐₓ (at x = 7ℓ/16) = 49wℓ²/512
    - Mₓ (when x < ℓ) = (wx/16)(7ℓ - 8x)
    """
    beam = FEModel3D()

    # Parameters
    L = 10.0  # feet per span
    w = 1.0   # kips/ft

    # Nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)
    beam.add_node('N3', 2*L, 0, 0)

    # Supports
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', False, True, True, True, False, False)
    beam.def_support('N3', False, True, True, True, False, False)

    # Material and section
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    I = 200/12**4  # ft⁴
    beam.add_section('Section', 12/144, I, I, 400/12**4)

    # Member with UDL only on first span
    beam.add_member('M1', 'N1', 'N3', 'Steel', 'Section')
    beam.add_member_dist_load('M1', 'FY', -w, -w, 0, L, case='D')
    beam.add_load_combo('D', {'D': 1.0})

    beam.analyze_linear()

    # Test reactions
    R1_expected = 7 * w * L / 16
    R2_expected = 5 * w * L / 8
    R3_expected = -w * L / 16  # Upward (negative in Pynite convention)

    assert beam.nodes['N1'].RxnFY['D'] == pytest.approx(R1_expected, rel=0.02), \
        f'R1 = {beam.nodes["N1"].RxnFY["D"]}, expected {R1_expected}'
    assert beam.nodes['N2'].RxnFY['D'] == pytest.approx(R2_expected, rel=0.02), \
        f'R2 = {beam.nodes["N2"].RxnFY["D"]}, expected {R2_expected}'
    assert beam.nodes['N3'].RxnFY['D'] == pytest.approx(R3_expected, rel=0.02), \
        f'R3 = {beam.nodes["N3"].RxnFY["D"]}, expected {R3_expected}'

    # Test moment at interior support
    M1_expected = w * L**2 / 16
    M1_actual = beam.members['M1'].moment('Mz', L, 'D')
    assert M1_actual == pytest.approx(M1_expected, rel=0.02), \
        f'M at interior support: {M1_actual}, expected {M1_expected}'

    # Test max moment in loaded span (at x = 7ℓ/16)
    # Note: Pynite uses negative for sagging moments
    x_max = 7 * L / 16
    M_max_expected = 49 * w * L**2 / 512
    M_max_actual = beam.members['M1'].moment('Mz', x_max, 'D')
    assert abs(M_max_actual) == pytest.approx(M_max_expected, rel=0.02), \
        f'M_max at x=7ℓ/16: {abs(M_max_actual)}, expected {M_max_expected}'


def test_figure_27_two_equal_spans_point_load_at_center_one_span():
    """
    AWC Figure 27: Continuous Beam - Two Equal Spans - Concentrated Load at Center
                    of One Span

    Configuration: |----ℓ----•----ℓ----|
                       P
                   (at ℓ/2)

    Formulas from AWC Design Aid No. 6:
    - R₁ = 13P/32
    - R₂ = 11P/16
    - R₃ = -3P/32 (upward)
    - V₂ = 19P/32
    - Mₘₐₓ (at point of load) = 13Pℓ/64
    - M₁ (at support R₂) = -3Pℓ/32
    """
    beam = FEModel3D()

    # Parameters
    L = 12.0  # feet per span
    P = 10.0  # kips

    # Nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L, 0, 0)
    beam.add_node('N3', 2*L, 0, 0)

    # Supports
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', False, True, True, True, False, False)
    beam.def_support('N3', False, True, True, True, False, False)

    # Material and section
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    I = 200/12**4  # ft⁴
    beam.add_section('Section', 12/144, I, I, 400/12**4)

    # Member with point load at center of first span
    beam.add_member('M1', 'N1', 'N3', 'Steel', 'Section')
    beam.add_member_pt_load('M1', 'FY', -P, L/2, case='L')
    beam.add_load_combo('L', {'L': 1.0})

    beam.analyze_linear()

    # Test reactions
    R1_expected = 13 * P / 32
    R2_expected = 11 * P / 16
    R3_expected = -3 * P / 32  # Upward

    assert beam.nodes['N1'].RxnFY['L'] == pytest.approx(R1_expected, rel=0.02), \
        f'R1 = {beam.nodes["N1"].RxnFY["L"]}, expected {R1_expected}'
    assert beam.nodes['N2'].RxnFY['L'] == pytest.approx(R2_expected, rel=0.02), \
        f'R2 = {beam.nodes["N2"].RxnFY["L"]}, expected {R2_expected}'
    assert beam.nodes['N3'].RxnFY['L'] == pytest.approx(R3_expected, rel=0.02), \
        f'R3 = {beam.nodes["N3"].RxnFY["L"]}, expected {R3_expected}'

    # Test moment at point of load (max)
    # Note: Pynite uses negative for sagging moments
    M_max_expected = 13 * P * L / 64
    M_max_actual = beam.members['M1'].moment('Mz', L/2, 'L')
    assert abs(M_max_actual) == pytest.approx(M_max_expected, rel=0.02), \
        f'M_max at load: {abs(M_max_actual)}, expected {M_max_expected}'

    # Test moment at interior support
    M1_expected = 3 * P * L / 32
    M1_actual = beam.members['M1'].moment('Mz', L, 'L')
    assert M1_actual == pytest.approx(M1_expected, rel=0.02), \
        f'M at interior support: {M1_actual}, expected {M1_expected}'


def test_figure_31_two_unequal_spans_udl():
    """
    AWC Figure 31: Continuous Beam - Two Unequal Spans - Uniformly Distributed Load

    Configuration: |----ℓ₁----•----ℓ₂----|
                   w₁         w₂

    For this test, use equal loads w₁ = w₂ = w, but unequal spans.

    Formulas from AWC Design Aid No. 6:
    - R₁ = M₁/ℓ₁ + wℓ₁/2
    - R₂ = wℓ₁ + wℓ₂ - R₁ - R₃
    - R₃ = M₁/ℓ₂ + wℓ₂/2
    - M₁ (at interior support) = -(wℓ₁³ + wℓ₂³)/(8(ℓ₁ + ℓ₂))
    - Mₓ₁ (when x₁ = R₁/w) = R₁x₁ - wx₁²/2
    - Mₓ₂ (when x₂ = R₃/w) = R₃x₂ - wx₂²/2
    """
    beam = FEModel3D()

    # Parameters
    L1 = 12.0  # feet (first span)
    L2 = 8.0   # feet (second span)
    w = 1.0    # kips/ft (same load on both spans)

    # Nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', L1, 0, 0)
    beam.add_node('N3', L1 + L2, 0, 0)

    # Supports
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', False, True, True, True, False, False)
    beam.def_support('N3', False, True, True, True, False, False)

    # Material and section
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    beam.add_material('Steel', E, G, 0.3, 0.490)

    I = 250/12**4  # ft⁴
    beam.add_section('Section', 14/144, I, I, 500/12**4)

    # Member with UDL on both spans
    beam.add_member('M1', 'N1', 'N3', 'Steel', 'Section')
    beam.add_member_dist_load('M1', 'FY', -w, -w, case='D')
    beam.add_load_combo('D', {'D': 1.0})

    beam.analyze_linear()

    # Calculate expected values using AWC formulas (Figure 31, page 19)
    # AWC formula shows: M₁ = -(wℓ₁³ + wℓ₂³)/(8(ℓ₁ + ℓ₂)) (negative = hogging)
    # For reaction formulas, AWC uses the magnitude in: R₁ = M₁/ℓ₁ + wℓ₁/2
    # where M₁ appears to be entered as a positive value in the formula

    # First find magnitude of M₁ at interior support
    M1_magnitude = (w * L1**3 + w * L2**3) / (8 * (L1 + L2))
    M1_expected = M1_magnitude  # Positive for hogging in Pynite convention

    # Calculate reactions - but AWC formula has R₁ = M₁/ℓ₁ + wℓ₁/2
    # This gives R₁ > wℓ₁/2 which makes sense for the longer span
    # However, checking if the signs are correct...
    # The AWC diagram shows these reactions, let me use the formulas as stated
    R1_expected = w * L1 / 2 - M1_magnitude / L1
    R3_expected = w * L2 / 2 - M1_magnitude / L2
    R2_expected = w * (L1 + L2) - R1_expected - R3_expected

    # Test reactions
    assert beam.nodes['N1'].RxnFY['D'] == pytest.approx(R1_expected, rel=0.02), \
        f'R1 = {beam.nodes["N1"].RxnFY["D"]}, expected {R1_expected}'
    assert beam.nodes['N2'].RxnFY['D'] == pytest.approx(R2_expected, rel=0.02), \
        f'R2 = {beam.nodes["N2"].RxnFY["D"]}, expected {R2_expected}'
    assert beam.nodes['N3'].RxnFY['D'] == pytest.approx(R3_expected, rel=0.02), \
        f'R3 = {beam.nodes["N3"].RxnFY["D"]}, expected {R3_expected}'

    # Test moment at interior support
    M1_actual = beam.members['M1'].moment('Mz', L1, 'D')
    assert M1_actual == pytest.approx(M1_expected, rel=0.02), \
        f'M at interior support: {M1_actual}, expected {M1_expected}'

    # Test max moment in first span (at x₁ = R₁/w)
    # Note: Pynite uses negative for sagging moments
    x1_max = R1_expected / w
    M_x1_expected = R1_expected * x1_max - w * x1_max**2 / 2
    M_x1_actual = beam.members['M1'].moment('Mz', x1_max, 'D')
    assert abs(M_x1_actual) == pytest.approx(M_x1_expected, rel=0.02), \
        f'M_max in span 1: {abs(M_x1_actual)}, expected {M_x1_expected}'

    # Test max moment in second span (at x₂ = R₃/w from right end)
    x2_from_right = R3_expected / w
    x2_from_left = L1 + L2 - x2_from_right
    M_x2_expected = R3_expected * x2_from_right - w * x2_from_right**2 / 2
    M_x2_actual = beam.members['M1'].moment('Mz', x2_from_left, 'D')
    assert abs(M_x2_actual) == pytest.approx(M_x2_expected, rel=0.02), \
        f'M_max in span 2: {abs(M_x2_actual)}, expected {M_x2_expected}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
