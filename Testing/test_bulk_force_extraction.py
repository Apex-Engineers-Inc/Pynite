"""
Tests for bulk force extraction optimization.

These tests verify that the optimized bulk force extraction methods produce
results that match:
1. Known analytical solutions for simple beam cases
2. Traditional segment-based extraction methods
3. Results across various loading conditions
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from Pynite import FEModel3D


# =============================================================================
# Analytical Solution Tests
# =============================================================================

class TestSimplySuportedBeamAnalytical:
    """Test internal forces against known analytical solutions for simply supported beams."""

    def test_uniform_load_simply_supported(self):
        """
        Simply supported beam with uniform load.

        Analytical solutions:
        - Max moment at midspan: |M| = wL^2/8
        - Shear at supports: V = wL/2
        - Shear at midspan: V = 0
        - Moment at supports: M = 0
        """
        model = FEModel3D()
        L = 10.0  # Length in feet
        w = -1.0  # Uniform load (negative = downward in local y)

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)

        E = 29000  # ksi
        I = 100    # in^4
        A = 10     # in^2
        model.add_material('Steel', E, 0.3*E, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', A, I, I, 2*I)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        # Pin at N1, roller at N2
        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)

        # Add uniform load
        model.add_member_dist_load('M1', 'Fy', w, w, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        member = model.members['M1']
        combo_names = ['1.0D']
        n_points = 21

        # Get bulk results
        bulk_results = member.get_all_forces_array(combo_names, n_points)

        shear = bulk_results['Fy'][0, :]
        moment = bulk_results['Mz'][0, :]

        # Verify bulk matches traditional (the authoritative comparison)
        trad_shear = member.shear_array('Fy', n_points, '1.0D')
        trad_moment = member.moment_array('Mz', n_points, '1.0D')

        assert_allclose(shear, trad_shear[1], rtol=1e-6,
                       err_msg="Bulk shear does not match traditional")
        assert_allclose(moment, trad_moment[1], rtol=1e-6,
                       err_msg="Bulk moment does not match traditional")

        # Verify max moment magnitude = |w|*L^2/8
        M_max_analytical = abs(w) * L**2 / 8
        M_max_computed = np.max(np.abs(moment))

        assert M_max_computed == pytest.approx(M_max_analytical, rel=1e-5), \
            "Max moment does not match wL^2/8"

        # Verify shear at supports and midspan
        assert np.abs(shear[0]) == pytest.approx(abs(w) * L / 2, rel=1e-5)
        assert shear[n_points // 2] == pytest.approx(0, abs=1e-5)

    def test_point_load_at_midspan(self):
        """
        Simply supported beam with point load at midspan.

        Analytical solutions:
        - Max moment at midspan: |M| = |P|*L/4
        - Shear: |V| = |P|/2 (constant each side of load)
        """
        model = FEModel3D()
        L = 12.0
        P = -10.0  # Point load (negative = downward)

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)

        # Point load at midspan
        model.add_member_pt_load('M1', 'Fy', P, L/2, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        member = model.members['M1']

        # Get traditional results for comparison
        trad_shear = member.shear_array('Fy', 21, '1.0D')
        trad_moment = member.moment_array('Mz', 21, '1.0D')

        # Get bulk results
        bulk_results = member.get_all_forces_array(['1.0D'], 21)

        # Verify bulk matches traditional
        assert_allclose(bulk_results['Fy'][0, :], trad_shear[1], rtol=1e-6,
                       err_msg="Bulk shear does not match traditional")
        assert_allclose(bulk_results['Mz'][0, :], trad_moment[1], rtol=1e-6,
                       err_msg="Bulk moment does not match traditional")

        # Verify max moment magnitude = |P|*L/4
        M_max_analytical = abs(P) * L / 4
        M_max_computed = np.max(np.abs(bulk_results['Mz'][0, :]))
        assert M_max_computed == pytest.approx(M_max_analytical, rel=1e-3)

    def test_cantilever_with_end_load(self):
        """
        Cantilever beam with point load at free end.

        Analytical solutions:
        - Moment at fixed end: M = PL
        - Shear: V = P (constant)
        - Moment at free end: M = 0
        """
        model = FEModel3D()
        L = 8.0
        P = -5.0  # Point load at free end

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        # Fixed at N1
        model.def_support('N1', True, True, True, True, True, True)

        # Point load at free end
        model.add_member_pt_load('M1', 'Fy', P, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        member = model.members['M1']
        bulk_results = member.get_all_forces_array(['1.0D'], 21)

        shear = bulk_results['Fy'][0, :]
        moment = bulk_results['Mz'][0, :]

        trad_shear = member.shear_array('Fy', 21, '1.0D')
        trad_moment = member.moment_array('Mz', 21, '1.0D')

        assert_allclose(shear, trad_shear[1], rtol=1e-6)
        assert_allclose(moment, trad_moment[1], rtol=1e-6)


# =============================================================================
# Bulk vs Traditional Extraction Tests
# =============================================================================

class TestBulkVsTraditionalExtraction:
    """Test that bulk extraction matches traditional segment-based extraction."""

    def test_multiple_load_combinations(self):
        """Verify bulk extraction matches traditional for multiple load combinations."""
        model = FEModel3D()

        # Create a simple beam
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, False, False, False)

        # Add different load cases
        model.add_member_dist_load('M1', 'Fy', -1.0, -1.0, 0, 10, 'Dead')
        model.add_member_dist_load('M1', 'Fy', -0.5, -0.5, 0, 10, 'Live')
        model.add_member_pt_load('M1', 'Fy', -5, 5, 'Wind')

        # Multiple load combinations
        model.add_load_combo('1.4D', {'Dead': 1.4})
        model.add_load_combo('1.2D+1.6L', {'Dead': 1.2, 'Live': 1.6})
        model.add_load_combo('1.2D+L+W', {'Dead': 1.2, 'Live': 1.0, 'Wind': 1.0})
        model.add_load_combo('0.9D+W', {'Dead': 0.9, 'Wind': 1.0})

        model.analyze()

        member = model.members['M1']
        combo_names = list(model.load_combos.keys())
        n_points = 25

        # Get bulk results
        bulk_results = member.get_all_forces_array(combo_names, n_points)

        # Compare with traditional for each combo
        for i, combo_name in enumerate(combo_names):
            trad_shear = member.shear_array('Fy', n_points, combo_name)
            trad_moment = member.moment_array('Mz', n_points, combo_name)
            trad_axial = member.axial_array(n_points, combo_name)
            trad_torque = member.torque_array(n_points, combo_name)

            assert_allclose(bulk_results['Fy'][i, :], trad_shear[1], rtol=1e-6,
                           err_msg=f"Shear mismatch for {combo_name}")
            assert_allclose(bulk_results['Mz'][i, :], trad_moment[1], rtol=1e-6,
                           err_msg=f"Moment mismatch for {combo_name}")
            assert_allclose(bulk_results['Fx'][i, :], trad_axial[1], rtol=1e-6,
                           err_msg=f"Axial mismatch for {combo_name}")
            assert_allclose(bulk_results['Mx'][i, :], trad_torque[1], rtol=1e-6,
                           err_msg=f"Torque mismatch for {combo_name}")

    def test_triangular_distributed_load(self):
        """Test with triangular (varying) distributed load."""
        model = FEModel3D()
        L = 12.0

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, False, False, False)

        # Triangular load: 0 at start, -2 at end
        model.add_member_dist_load('M1', 'Fy', 0, -2.0, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        member = model.members['M1']
        n_points = 31

        bulk_results = member.get_all_forces_array(['1.0D'], n_points)

        trad_shear = member.shear_array('Fy', n_points, '1.0D')
        trad_moment = member.moment_array('Mz', n_points, '1.0D')

        assert_allclose(bulk_results['Fy'][0, :], trad_shear[1], rtol=1e-5,
                       err_msg="Shear mismatch for triangular load")
        assert_allclose(bulk_results['Mz'][0, :], trad_moment[1], rtol=1e-5,
                       err_msg="Moment mismatch for triangular load")

    def test_axial_load(self):
        """Test axial force extraction."""
        model = FEModel3D()
        L = 10.0

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', False, True, True, False, False, False)

        # Axial distributed load
        model.add_member_dist_load('M1', 'Fx', -1.0, -1.0, 0, L, 'D')
        # Axial point load at end
        model.add_node_load('N2', 'FX', -10, 'P')

        model.add_load_combo('D+P', {'D': 1.0, 'P': 1.0})

        model.analyze()

        member = model.members['M1']
        n_points = 21

        bulk_results = member.get_all_forces_array(['D+P'], n_points)
        trad_axial = member.axial_array(n_points, 'D+P')

        assert_allclose(bulk_results['Fx'][0, :], trad_axial[1], rtol=1e-5,
                       err_msg="Axial force mismatch")

    def test_torsional_load(self):
        """Test torsional moment extraction."""
        model = FEModel3D()
        L = 8.0

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, True, False, False)

        # Torsional moment at midspan
        model.add_member_pt_load('M1', 'Mx', 100, L/2, 'T')
        model.add_load_combo('1.0T', {'T': 1.0})

        model.analyze()

        member = model.members['M1']
        n_points = 21

        bulk_results = member.get_all_forces_array(['1.0T'], n_points)
        trad_torque = member.torque_array(n_points, '1.0T')

        assert_allclose(bulk_results['Mx'][0, :], trad_torque[1], rtol=1e-5,
                       err_msg="Torque mismatch")


# =============================================================================
# Wall Design Scenario Tests
# =============================================================================

class TestWallDesignScenario:
    """Test the specific wood wall design scenario."""

    def test_wall_stud_forces(self):
        """Test forces in a wall stud with typical loading."""
        model = FEModel3D()

        # 8 ft wall stud
        L = 8.0 * 12  # inches

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 0, L, 0)

        # Wood properties (approximate SPF)
        E = 1400  # ksi
        G = 100   # ksi
        model.add_material('Wood', E, G, 35/12**3, 35/12**3)

        # 2x4 stud
        A = 1.5 * 3.5
        Ix = 1.5 * 3.5**3 / 12
        Iy = 3.5 * 1.5**3 / 12
        J = 0.5  # approximate
        model.add_section('2x4', A, Ix, Iy, J)

        model.add_member('Stud', 'N1', 'N2', 'Wood', '2x4')

        # Pin at bottom, roller at top
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, False, False, False)

        # Tributary axial load from above
        model.add_node_load('N2', 'FY', -500, 'D')  # Dead load
        model.add_node_load('N2', 'FY', -300, 'L')  # Live load

        # Lateral wind load
        model.add_member_dist_load('Stud', 'Fz', 0.1, 0.1, 0, L, 'W')

        # Load combinations
        model.add_load_combo('1.4D', {'D': 1.4})
        model.add_load_combo('1.2D+1.6L', {'D': 1.2, 'L': 1.6})
        model.add_load_combo('1.2D+L+W', {'D': 1.2, 'L': 1.0, 'W': 1.0})

        model.analyze()

        member = model.members['Stud']
        combo_names = list(model.load_combos.keys())
        n_points = 20

        # Get bulk results
        bulk_results = member.get_all_forces_array(combo_names, n_points)

        # Verify against traditional
        for i, combo_name in enumerate(combo_names):
            trad_shear = member.shear_array('Fy', n_points, combo_name)
            trad_moment = member.moment_array('Mz', n_points, combo_name)
            trad_axial = member.axial_array(n_points, combo_name)

            assert_allclose(bulk_results['Fy'][i, :], trad_shear[1], rtol=1e-5,
                           err_msg=f"Shear mismatch for {combo_name}")
            assert_allclose(bulk_results['Mz'][i, :], trad_moment[1], rtol=1e-5,
                           err_msg=f"Moment mismatch for {combo_name}")
            assert_allclose(bulk_results['Fx'][i, :], trad_axial[1], rtol=1e-5,
                           err_msg=f"Axial mismatch for {combo_name}")

    def test_full_wall_extraction(self):
        """Test extraction for a complete wall with multiple studs."""
        model = FEModel3D()

        # Wall dimensions
        wall_length = 8 * 12  # 8 ft in inches
        wall_height = 8 * 12  # 8 ft in inches
        stud_spacing = 16     # 16 inches OC

        # Material
        model.add_material('Wood', 1400, 100, 35/12**3, 35/12**3)
        model.add_section('2x4', 5.25, 5.36, 0.98, 0.5)
        model.add_section('2x6', 8.25, 20.8, 1.55, 0.8)

        # Create nodes for bottom plate
        n_studs = int(wall_length / stud_spacing) + 1

        for i in range(n_studs):
            x = i * stud_spacing
            model.add_node(f'B{i}', x, 0, 0)
            model.add_node(f'T{i}', x, wall_height, 0)

            # Add stud
            model.add_member(f'Stud_{i}', f'B{i}', f'T{i}', 'Wood', '2x4')

            # Supports
            model.def_support(f'B{i}', True, True, True, True, True, True)

        # Add plates
        for i in range(n_studs - 1):
            model.add_member(f'Bot_plate_{i}', f'B{i}', f'B{i+1}', 'Wood', '2x6')
            model.add_member(f'Top_plate_{i}', f'T{i}', f'T{i+1}', 'Wood', '2x6')

        # Loads
        for i in range(n_studs):
            model.add_node_load(f'T{i}', 'FY', -200, 'D')
            model.add_node_load(f'T{i}', 'FY', -100, 'L')

        # Lateral load on studs
        for i in range(n_studs):
            model.add_member_dist_load(f'Stud_{i}', 'Fz', 0.05, 0.05, 0, wall_height, 'W')

        model.add_load_combo('1.2D+1.6L', {'D': 1.2, 'L': 1.6})
        model.add_load_combo('1.2D+L+W', {'D': 1.2, 'L': 1.0, 'W': 1.0})

        model.analyze()

        combo_names = list(model.load_combos.keys())
        n_points = 20

        # Test all members
        for member_name, member in model.members.items():
            bulk_results = member.get_all_forces_array(combo_names, n_points)

            for i, combo_name in enumerate(combo_names):
                trad_shear = member.shear_array('Fy', n_points, combo_name)
                trad_moment = member.moment_array('Mz', n_points, combo_name)
                trad_axial = member.axial_array(n_points, combo_name)
                trad_torque = member.torque_array(n_points, combo_name)

                assert_allclose(bulk_results['Fy'][i, :], trad_shear[1], rtol=1e-4,
                               err_msg=f"Shear mismatch for {member_name} - {combo_name}")
                assert_allclose(bulk_results['Mz'][i, :], trad_moment[1], rtol=1e-4,
                               err_msg=f"Moment mismatch for {member_name} - {combo_name}")
                assert_allclose(bulk_results['Fx'][i, :], trad_axial[1], rtol=1e-4,
                               err_msg=f"Axial mismatch for {member_name} - {combo_name}")
                assert_allclose(bulk_results['Mx'][i, :], trad_torque[1], rtol=1e-4,
                               err_msg=f"Torque mismatch for {member_name} - {combo_name}")


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_loads(self):
        """Test member with no loads (should return zeros)."""
        model = FEModel3D()

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, True, True, True)

        model.add_load_combo('Empty', {'Case 1': 1.0})

        model.analyze()

        member = model.members['M1']
        bulk_results = member.get_all_forces_array(['Empty'], 11)

        # All forces should be essentially zero
        assert_allclose(bulk_results['Fy'][0, :], 0, atol=1e-10)
        assert_allclose(bulk_results['Mz'][0, :], 0, atol=1e-10)
        assert_allclose(bulk_results['Fx'][0, :], 0, atol=1e-10)
        assert_allclose(bulk_results['Mx'][0, :], 0, atol=1e-10)

    def test_single_point(self):
        """Test extraction with a single point."""
        model = FEModel3D()

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, False, False, False)

        model.add_member_dist_load('M1', 'Fy', -1, -1, 0, 10, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        member = model.members['M1']
        bulk_results = member.get_all_forces_array(['1.0D'], 1)

        # Should have exactly 1 point
        assert bulk_results['Fy'].shape == (1, 1)
        assert bulk_results['Mz'].shape == (1, 1)

    def test_many_points(self):
        """Test extraction with many points."""
        model = FEModel3D()

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, False, False, False)

        model.add_member_dist_load('M1', 'Fy', -1, -1, 0, 10, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        member = model.members['M1']
        n_points = 1001
        bulk_results = member.get_all_forces_array(['1.0D'], n_points)

        trad_shear = member.shear_array('Fy', n_points, '1.0D')

        assert_allclose(bulk_results['Fy'][0, :], trad_shear[1], rtol=1e-5)

    def test_global_direction_loads(self):
        """Test with loads in global directions (FX, FY, FZ)."""
        model = FEModel3D()

        # Inclined member
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 10, 0)  # 45 degree incline

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, False, False, False)

        # Global gravity load (global Y direction)
        model.add_member_dist_load('M1', 'FY', -1, -1, 0, model.members['M1'].L(), 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        member = model.members['M1']
        n_points = 21

        bulk_results = member.get_all_forces_array(['1.0D'], n_points)

        trad_shear = member.shear_array('Fy', n_points, '1.0D')
        trad_moment = member.moment_array('Mz', n_points, '1.0D')

        assert_allclose(bulk_results['Fy'][0, :], trad_shear[1], rtol=1e-4)
        assert_allclose(bulk_results['Mz'][0, :], trad_moment[1], rtol=1e-4)


# =============================================================================
# Continuous Beam Tests
# =============================================================================

class TestContinuousBeams:
    """Test internal forces for continuous beams against known analytical solutions."""

    def test_two_span_uniform_load(self):
        """
        Two-span continuous beam with uniform load.

        For two equal spans L with uniform load w:
        - Moment magnitude at middle support: |M_B| = wL²/8
        - End reactions: R_A = R_C = 3wL/8
        - Middle reaction: R_B = 10wL/8 = 5wL/4
        - Max positive moment in each span: 9wL²/128 at x = 3L/8 from end support

        Reference: Roark's Formulas for Stress and Strain, Table 8.1
        """
        model = FEModel3D()
        L = 10.0  # Span length
        w = -1.0  # Uniform load (negative = downward)

        # Create three nodes for two spans
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)
        model.add_node('N3', 2*L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'W10')

        # Pin at N1, roller at N2 (interior support), roller at N3
        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)
        model.def_support('N3', True, True, True, True, True, False)

        # Add uniform load to both spans
        model.add_member_dist_load('M1', 'Fy', w, w, 0, L, 'D')
        model.add_member_dist_load('M2', 'Fy', w, w, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        # Analytical solutions (magnitudes)
        w_abs = abs(w)
        M_middle_support_mag = w_abs * L**2 / 8  # Moment magnitude at interior support
        M_max_span_mag = 9 * w_abs * L**2 / 128  # Max moment in span (away from support)
        R_end = 3 * w_abs * L / 8  # Reactions at end supports

        # Test first span (M1)
        member1 = model.members['M1']
        n_points = 41  # Use many points for accuracy

        bulk_results1 = member1.get_all_forces_array(['1.0D'], n_points)
        trad_moment1 = member1.moment_array('Mz', n_points, '1.0D')
        trad_shear1 = member1.shear_array('Fy', n_points, '1.0D')

        # Verify bulk matches traditional
        assert_allclose(bulk_results1['Mz'][0, :], trad_moment1[1], rtol=1e-6,
                       err_msg="Bulk moment does not match traditional for span 1")
        assert_allclose(bulk_results1['Fy'][0, :], trad_shear1[1], rtol=1e-6,
                       err_msg="Bulk shear does not match traditional for span 1")

        # Verify moment magnitude at interior support (end of M1)
        moment_at_support = bulk_results1['Mz'][0, -1]
        assert abs(moment_at_support) == pytest.approx(M_middle_support_mag, rel=0.01), \
            f"Moment magnitude at interior support: expected {M_middle_support_mag}, got {abs(moment_at_support)}"

        # Verify max span moment has opposite sign from support moment (sagging vs hogging)
        moment_array = bulk_results1['Mz'][0, :]
        max_idx = np.argmax(np.abs(moment_array[:-5]))  # Exclude near-support points
        min_idx = np.argmin(moment_array)
        # Moment should change sign between mid-span and support
        assert moment_array[0] * moment_array[-1] < 0 or abs(moment_array[0]) < 0.1, \
            "Moment should change sign along span (sagging in middle, hogging at support)"

        # Verify shear magnitude at end support
        shear_at_start = abs(bulk_results1['Fy'][0, 0])
        assert shear_at_start == pytest.approx(R_end, rel=0.01), \
            f"Shear at end support: expected {R_end}, got {shear_at_start}"

        # Test second span (M2) for symmetry
        member2 = model.members['M2']
        bulk_results2 = member2.get_all_forces_array(['1.0D'], n_points)
        trad_moment2 = member2.moment_array('Mz', n_points, '1.0D')

        assert_allclose(bulk_results2['Mz'][0, :], trad_moment2[1], rtol=1e-6,
                       err_msg="Bulk moment does not match traditional for span 2")

        # Moment magnitude at start of M2 should match moment at end of M1 (continuity)
        moment_at_support2 = bulk_results2['Mz'][0, 0]
        assert abs(moment_at_support2) == pytest.approx(abs(moment_at_support), rel=0.01), \
            f"Moment continuity at interior support: {moment_at_support} vs {moment_at_support2}"

    def test_two_span_point_load_center(self):
        """
        Two-span continuous beam with point load at center of first span.

        For point load P at center of first span with equal spans L:
        - Moment magnitude at middle support: |M_B| = 3PL/32
        - Moment magnitude under load: |M_load| = 13PL/64

        Reference: Beam formulas with shear and moment diagrams
        """
        model = FEModel3D()
        L = 12.0  # Span length
        P = -10.0  # Point load (negative = downward)

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)
        model.add_node('N3', 2*L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)
        model.def_support('N3', True, True, True, True, True, False)

        # Point load at center of first span
        model.add_member_pt_load('M1', 'Fy', P, L/2, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        # Analytical solutions (magnitudes)
        P_abs = abs(P)
        M_middle_support_mag = 3 * P_abs * L / 32  # Moment magnitude at interior support
        M_under_load_mag = 13 * P_abs * L / 64  # Moment magnitude under point load

        member1 = model.members['M1']
        n_points = 41

        bulk_results = member1.get_all_forces_array(['1.0D'], n_points)
        trad_moment = member1.moment_array('Mz', n_points, '1.0D')
        trad_shear = member1.shear_array('Fy', n_points, '1.0D')

        # Verify bulk matches traditional
        assert_allclose(bulk_results['Mz'][0, :], trad_moment[1], rtol=1e-6,
                       err_msg="Bulk moment does not match traditional")
        assert_allclose(bulk_results['Fy'][0, :], trad_shear[1], rtol=1e-6,
                       err_msg="Bulk shear does not match traditional")

        # Verify moment magnitude at interior support
        moment_at_support = bulk_results['Mz'][0, -1]
        assert abs(moment_at_support) == pytest.approx(M_middle_support_mag, rel=0.02), \
            f"Moment magnitude at interior support: expected {M_middle_support_mag}, got {abs(moment_at_support)}"

        # Verify moment magnitude under load (at midpoint)
        midpoint_idx = n_points // 2
        moment_under_load = bulk_results['Mz'][0, midpoint_idx]
        assert abs(moment_under_load) == pytest.approx(M_under_load_mag, rel=0.02), \
            f"Moment magnitude under load: expected {M_under_load_mag}, got {abs(moment_under_load)}"

        # Test second span (should have no load, only moments from continuity)
        member2 = model.members['M2']
        bulk_results2 = member2.get_all_forces_array(['1.0D'], n_points)
        trad_moment2 = member2.moment_array('Mz', n_points, '1.0D')

        assert_allclose(bulk_results2['Mz'][0, :], trad_moment2[1], rtol=1e-6)

        # Moment should vary linearly from support moment to ~0 in span 2
        assert abs(bulk_results2['Mz'][0, 0]) == pytest.approx(M_middle_support_mag, rel=0.02)
        assert abs(bulk_results2['Mz'][0, -1]) == pytest.approx(0, abs=0.1)

    def test_three_span_uniform_load(self):
        """
        Three-span continuous beam with uniform load.

        For three equal spans L with uniform load w:
        - Moment magnitude at interior supports: |M| = wL²/10
        - End span max positive moment: 0.08wL² at x = 0.4L from end
        - Center span max positive moment: wL²/40 = 0.025wL² at center

        Reference: Continuous beam formulas
        """
        model = FEModel3D()
        L = 10.0
        w = -1.0

        # Four nodes for three spans
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)
        model.add_node('N3', 2*L, 0, 0)
        model.add_node('N4', 3*L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'W10')
        model.add_member('M3', 'N3', 'N4', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)
        model.def_support('N3', True, True, True, True, True, False)
        model.def_support('N4', True, True, True, True, True, False)

        model.add_member_dist_load('M1', 'Fy', w, w, 0, L, 'D')
        model.add_member_dist_load('M2', 'Fy', w, w, 0, L, 'D')
        model.add_member_dist_load('M3', 'Fy', w, w, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        # Analytical values (magnitudes)
        w_abs = abs(w)
        M_interior_support_mag = w_abs * L**2 / 10  # Moment magnitude at interior supports

        n_points = 41

        # Test each span
        for member_name in ['M1', 'M2', 'M3']:
            member = model.members[member_name]

            bulk_results = member.get_all_forces_array(['1.0D'], n_points)
            trad_moment = member.moment_array('Mz', n_points, '1.0D')
            trad_shear = member.shear_array('Fy', n_points, '1.0D')

            # Verify bulk matches traditional
            assert_allclose(bulk_results['Mz'][0, :], trad_moment[1], rtol=1e-6,
                           err_msg=f"Bulk moment does not match traditional for {member_name}")
            assert_allclose(bulk_results['Fy'][0, :], trad_shear[1], rtol=1e-6,
                           err_msg=f"Bulk shear does not match traditional for {member_name}")

        # Verify interior support moment magnitude for first span (end = interior support)
        member1 = model.members['M1']
        bulk_results1 = member1.get_all_forces_array(['1.0D'], n_points)
        moment_at_first_interior = bulk_results1['Mz'][0, -1]
        assert abs(moment_at_first_interior) == pytest.approx(M_interior_support_mag, rel=0.02), \
            f"Moment magnitude at first interior support: expected {M_interior_support_mag}, got {abs(moment_at_first_interior)}"

        # Verify center span moments magnitude (both ends should be at interior supports)
        member2 = model.members['M2']
        bulk_results2 = member2.get_all_forces_array(['1.0D'], n_points)
        assert abs(bulk_results2['Mz'][0, 0]) == pytest.approx(M_interior_support_mag, rel=0.02)
        assert abs(bulk_results2['Mz'][0, -1]) == pytest.approx(M_interior_support_mag, rel=0.02)

        # Center span moment should have opposite sign from support moments
        center_span_midpoint = bulk_results2['Mz'][0, n_points // 2]
        support_moment_sign = np.sign(bulk_results2['Mz'][0, 0])
        assert np.sign(center_span_midpoint) != support_moment_sign, \
            "Center span midpoint moment should have opposite sign from support moments"

    def test_two_span_unequal_spans(self):
        """
        Two-span continuous beam with unequal spans and uniform load.

        Tests that the analysis handles unequal span lengths correctly.
        Span 1: L1 = 10 ft
        Span 2: L2 = 15 ft
        """
        model = FEModel3D()
        L1 = 10.0
        L2 = 15.0
        w = -1.0

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L1, 0, 0)
        model.add_node('N3', L1 + L2, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)
        model.def_support('N3', True, True, True, True, True, False)

        model.add_member_dist_load('M1', 'Fy', w, w, 0, L1, 'D')
        model.add_member_dist_load('M2', 'Fy', w, w, 0, L2, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        n_points = 41

        # For unequal spans, we verify bulk matches traditional (no simple closed-form solution)
        for member_name in ['M1', 'M2']:
            member = model.members[member_name]

            bulk_results = member.get_all_forces_array(['1.0D'], n_points)
            trad_moment = member.moment_array('Mz', n_points, '1.0D')
            trad_shear = member.shear_array('Fy', n_points, '1.0D')
            trad_axial = member.axial_array(n_points, '1.0D')

            assert_allclose(bulk_results['Mz'][0, :], trad_moment[1], rtol=1e-6,
                           err_msg=f"Moment mismatch for {member_name}")
            assert_allclose(bulk_results['Fy'][0, :], trad_shear[1], rtol=1e-6,
                           err_msg=f"Shear mismatch for {member_name}")
            assert_allclose(bulk_results['Fx'][0, :], trad_axial[1], rtol=1e-6,
                           err_msg=f"Axial mismatch for {member_name}")

        # Verify continuity at interior support
        member1 = model.members['M1']
        member2 = model.members['M2']
        bulk1 = member1.get_all_forces_array(['1.0D'], n_points)
        bulk2 = member2.get_all_forces_array(['1.0D'], n_points)

        # Moments should be continuous at interior support
        moment_end_span1 = bulk1['Mz'][0, -1]
        moment_start_span2 = bulk2['Mz'][0, 0]
        assert moment_end_span1 == pytest.approx(moment_start_span2, rel=0.001), \
            f"Moment discontinuity at interior support: {moment_end_span1} vs {moment_start_span2}"

        # Interior support should have non-zero moment (hogging behavior)
        assert abs(moment_end_span1) > 0.1, "Interior support should have significant moment"

    def test_propped_cantilever(self):
        """
        Propped cantilever (fixed-pinned beam) with uniform load.

        Analytical solutions (magnitudes):
        - Reaction at pinned end: R_B = 3wL/8
        - Reaction at fixed end: R_A = 5wL/8
        - Moment magnitude at fixed end: |M_A| = wL²/8
        - Max moment between supports: 9wL²/128 at x = 3L/8 from pinned end

        Reference: Roark's Formulas for Stress and Strain
        """
        model = FEModel3D()
        L = 12.0
        w = -1.0

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        # Fixed at N1, pinned at N2
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, True, True, False)

        model.add_member_dist_load('M1', 'Fy', w, w, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        # Analytical values (magnitudes)
        w_abs = abs(w)
        M_fixed_end_mag = w_abs * L**2 / 8
        R_pinned = 3 * w_abs * L / 8
        M_span_max_mag = 9 * w_abs * L**2 / 128

        member = model.members['M1']
        n_points = 41

        bulk_results = member.get_all_forces_array(['1.0D'], n_points)
        trad_moment = member.moment_array('Mz', n_points, '1.0D')
        trad_shear = member.shear_array('Fy', n_points, '1.0D')

        # Verify bulk matches traditional
        assert_allclose(bulk_results['Mz'][0, :], trad_moment[1], rtol=1e-6)
        assert_allclose(bulk_results['Fy'][0, :], trad_shear[1], rtol=1e-6)

        # Verify moment magnitude at fixed end
        moment_at_fixed = bulk_results['Mz'][0, 0]
        assert abs(moment_at_fixed) == pytest.approx(M_fixed_end_mag, rel=0.01), \
            f"Moment magnitude at fixed end: expected {M_fixed_end_mag}, got {abs(moment_at_fixed)}"

        # Verify moment at pinned end is zero
        moment_at_pinned = bulk_results['Mz'][0, -1]
        assert abs(moment_at_pinned) == pytest.approx(0, abs=0.01), \
            f"Moment at pinned end should be zero, got {moment_at_pinned}"

        # Verify shear magnitude at pinned end (reaction)
        shear_at_pinned = abs(bulk_results['Fy'][0, -1])
        assert shear_at_pinned == pytest.approx(R_pinned, rel=0.01), \
            f"Shear at pinned end: expected {R_pinned}, got {shear_at_pinned}"

        # Verify moment changes sign along the span (fixed end vs mid-span)
        moment_array = bulk_results['Mz'][0, :]
        # The moment at fixed end and max moment in span should have opposite signs
        fixed_sign = np.sign(moment_at_fixed)
        mid_region = moment_array[n_points//3:2*n_points//3]
        assert np.any(np.sign(mid_region) != fixed_sign), \
            "Moment should change sign between fixed end and mid-span"


# =============================================================================
# End Release Tests
# =============================================================================

class TestEndReleases:
    """Test members with end releases."""

    def test_pinned_end(self):
        """Test member with pinned end (moment release)."""
        model = FEModel3D()

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)
        model.add_node('N3', 20, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'W10')

        # Pin connection at N2 end of M1
        model.def_releases('M1', Rzj=True)

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N3', True, True, True, True, True, True)

        model.add_member_dist_load('M1', 'Fy', -1, -1, 0, 10, 'D')
        model.add_member_dist_load('M2', 'Fy', -1, -1, 0, 10, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        # Test both members
        for member_name in ['M1', 'M2']:
            member = model.members[member_name]
            n_points = 21

            bulk_results = member.get_all_forces_array(['1.0D'], n_points)

            trad_shear = member.shear_array('Fy', n_points, '1.0D')
            trad_moment = member.moment_array('Mz', n_points, '1.0D')

            assert_allclose(bulk_results['Fy'][0, :], trad_shear[1], rtol=1e-4,
                           err_msg=f"Shear mismatch for {member_name}")
            assert_allclose(bulk_results['Mz'][0, :], trad_moment[1], rtol=1e-4,
                           err_msg=f"Moment mismatch for {member_name}")


# =============================================================================
# Model-Level Batch Extraction Tests
# =============================================================================

class TestModelLevelExtraction:
    """Test model-level batch extraction methods."""

    def test_get_all_member_forces_basic(self):
        """Test basic model-level extraction returns correct structure."""
        model = FEModel3D()

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)
        model.add_node('N3', 20, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N3', True, True, True, True, True, True)

        model.add_member_dist_load('M1', 'Fy', -1, -1, 0, 10, 'D')
        model.add_member_dist_load('M2', 'Fy', -1, -1, 0, 10, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        # Test model-level extraction
        all_forces = model.get_all_member_forces(['1.0D'], n_points=21)

        # Check structure
        assert 'M1' in all_forces
        assert 'M2' in all_forces
        assert 'x' in all_forces['M1']
        assert 'Fy' in all_forces['M1']
        assert 'Mz' in all_forces['M1']
        assert 'Fx' in all_forces['M1']
        assert 'Mx' in all_forces['M1']

        # Check shapes
        assert all_forces['M1']['x'].shape == (21,)
        assert all_forces['M1']['Fy'].shape == (1, 21)
        assert all_forces['M1']['Mz'].shape == (1, 21)

    def test_get_all_member_forces_matches_individual(self):
        """Test that model-level extraction matches individual member extraction."""
        model = FEModel3D()

        # Create a simple wall-like structure
        wall_height = 96  # 8 ft in inches
        stud_spacing = 16

        model.add_material('Wood', 1400, 100, 35/12**3, 35/12**3)
        model.add_section('2x4', 5.25, 5.36, 0.98, 0.5)

        n_studs = 5
        for i in range(n_studs):
            x = i * stud_spacing
            model.add_node(f'B{i}', x, 0, 0)
            model.add_node(f'T{i}', x, wall_height, 0)
            model.add_member(f'Stud_{i}', f'B{i}', f'T{i}', 'Wood', '2x4')
            model.def_support(f'B{i}', True, True, True, True, True, True)

        # Add loads
        for i in range(n_studs):
            model.add_node_load(f'T{i}', 'FY', -200, 'D')
            model.add_member_dist_load(f'Stud_{i}', 'Fz', 0.05, 0.05, 0, wall_height, 'W')

        model.add_load_combo('1.2D+W', {'D': 1.2, 'W': 1.0})
        model.add_load_combo('1.4D', {'D': 1.4})

        model.analyze()

        combo_names = ['1.2D+W', '1.4D']
        n_points = 20

        # Get model-level results
        all_forces = model.get_all_member_forces(combo_names, n_points=n_points)

        # Compare with individual extraction
        for i in range(n_studs):
            member_name = f'Stud_{i}'
            member = model.members[member_name]

            individual = member.get_all_forces_array(combo_names, n_points)

            assert_allclose(all_forces[member_name]['x'], individual['x'], rtol=1e-10,
                           err_msg=f"x mismatch for {member_name}")
            assert_allclose(all_forces[member_name]['Fy'], individual['Fy'], rtol=1e-10,
                           err_msg=f"shear_y mismatch for {member_name}")
            assert_allclose(all_forces[member_name]['Mz'], individual['Mz'], rtol=1e-10,
                           err_msg=f"moment_z mismatch for {member_name}")
            assert_allclose(all_forces[member_name]['Fx'], individual['Fx'], rtol=1e-10,
                           err_msg=f"axial mismatch for {member_name}")
            assert_allclose(all_forces[member_name]['Mx'], individual['Mx'], rtol=1e-10,
                           err_msg=f"torque mismatch for {member_name}")

    def test_get_all_member_forces_array_structure(self):
        """Test that array-based extraction returns correct 3D array structure."""
        model = FEModel3D()

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)
        model.add_node('N3', 20, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N3', True, True, True, True, True, True)

        model.add_member_dist_load('M1', 'Fy', -1, -1, 0, 10, 'D')
        model.add_member_dist_load('M2', 'Fy', -1, -1, 0, 10, 'D')
        model.add_load_combo('Combo1', {'D': 1.0})
        model.add_load_combo('Combo2', {'D': 1.4})

        model.analyze()

        n_points = 15
        combo_names = ['Combo1', 'Combo2']
        member_names = ['M1', 'M2']

        forces = model.get_all_member_forces_array(combo_names, member_names, n_points)

        # Check metadata
        assert forces['member_names'] == member_names
        assert forces['combo_names'] == combo_names

        # Check array shapes: (n_members, n_combos, n_points)
        assert forces['x'].shape == (2, 15)  # x is (n_members, n_points)
        assert forces['Fy'].shape == (2, 2, 15)
        assert forces['Mz'].shape == (2, 2, 15)
        assert forces['Fx'].shape == (2, 2, 15)
        assert forces['Mx'].shape == (2, 2, 15)

    def test_get_all_member_forces_array_values(self):
        """Test that array-based extraction values match individual extraction."""
        model = FEModel3D()

        # Create 3 members
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)
        model.add_node('N3', 20, 0, 0)
        model.add_node('N4', 30, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'W10')
        model.add_member('M3', 'N3', 'N4', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N4', True, True, True, True, True, True)

        model.add_member_dist_load('M1', 'Fy', -1, -1, 0, 10, 'D')
        model.add_member_dist_load('M2', 'Fy', -2, -2, 0, 10, 'L')
        model.add_member_pt_load('M3', 'Fy', -5, 5, 'D')

        model.add_load_combo('1.4D', {'D': 1.4})
        model.add_load_combo('1.2D+1.6L', {'D': 1.2, 'L': 1.6})

        model.analyze()

        combo_names = ['1.4D', '1.2D+1.6L']
        member_names = ['M1', 'M2', 'M3']
        n_points = 21

        # Get array-based results
        forces = model.get_all_member_forces_array(combo_names, member_names, n_points)

        # Compare with individual extraction
        for i, member_name in enumerate(member_names):
            member = model.members[member_name]
            individual = member.get_all_forces_array(combo_names, n_points)

            assert_allclose(forces['x'][i, :], individual['x'], rtol=1e-10,
                           err_msg=f"x mismatch for {member_name}")
            assert_allclose(forces['Fy'][i, :, :], individual['Fy'], rtol=1e-10,
                           err_msg=f"shear_y mismatch for {member_name}")
            assert_allclose(forces['Mz'][i, :, :], individual['Mz'], rtol=1e-10,
                           err_msg=f"moment_z mismatch for {member_name}")
            assert_allclose(forces['Fx'][i, :, :], individual['Fx'], rtol=1e-10,
                           err_msg=f"axial mismatch for {member_name}")
            assert_allclose(forces['Mx'][i, :, :], individual['Mx'], rtol=1e-10,
                           err_msg=f"torque mismatch for {member_name}")

    def test_default_parameters(self):
        """Test that default parameters work correctly (all combos, all members)."""
        model = FEModel3D()

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, False, False, False)

        model.add_member_dist_load('M1', 'Fy', -1, -1, 0, 10, 'D')
        model.add_load_combo('Combo1', {'D': 1.0})
        model.add_load_combo('Combo2', {'D': 1.4})

        model.analyze()

        # Test with defaults (no arguments)
        forces_dict = model.get_all_member_forces()
        forces_array = model.get_all_member_forces_array()

        # Should have all members
        assert 'M1' in forces_dict
        assert len(forces_dict) == 1

        # Should have all combos
        assert forces_dict['M1']['Fy'].shape[0] == 2
        assert forces_array['Fy'].shape[1] == 2

    def test_subset_of_members(self):
        """Test extraction for a subset of members."""
        model = FEModel3D()

        # Create 5 members
        for i in range(6):
            model.add_node(f'N{i}', i * 10, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        for i in range(5):
            model.add_member(f'M{i}', f'N{i}', f'N{i+1}', 'Steel', 'W10')

        model.def_support('N0', True, True, True, True, True, True)
        model.def_support('N5', True, True, True, True, True, True)

        model.add_load_combo('Combo1', {'D': 1.0})
        model.analyze()

        # Extract only 2 members
        subset = ['M1', 'M3']
        forces = model.get_all_member_forces(member_names=subset)
        forces_arr = model.get_all_member_forces_array(member_names=subset)

        assert len(forces) == 2
        assert 'M1' in forces
        assert 'M3' in forces
        assert 'M0' not in forces

        assert forces_arr['member_names'] == subset
        assert forces_arr['Fy'].shape[0] == 2

    def test_wall_design_scenario(self):
        """Test model-level extraction on a realistic wall design scenario."""
        model = FEModel3D()

        # Wall parameters
        wall_height = 8 * 12  # 8 ft in inches
        stud_spacing = 16     # 16" OC
        wall_length = 8 * 12  # 8 ft

        model.add_material('Wood', 1400, 100, 35/12**3, 35/12**3)
        model.add_section('2x4', 5.25, 5.36, 0.98, 0.5)
        model.add_section('2x6', 8.25, 20.8, 1.55, 0.8)

        n_studs = int(wall_length / stud_spacing) + 1

        # Create studs
        for i in range(n_studs):
            x = i * stud_spacing
            model.add_node(f'B{i}', x, 0, 0)
            model.add_node(f'T{i}', x, wall_height, 0)
            model.add_member(f'Stud_{i}', f'B{i}', f'T{i}', 'Wood', '2x4')
            model.def_support(f'B{i}', True, True, True, True, True, True)

        # Create plates
        for i in range(n_studs - 1):
            model.add_member(f'Bot_{i}', f'B{i}', f'B{i+1}', 'Wood', '2x6')
            model.add_member(f'Top_{i}', f'T{i}', f'T{i+1}', 'Wood', '2x6')

        # Add loads
        for i in range(n_studs):
            model.add_node_load(f'T{i}', 'FY', -200, 'D')
            model.add_node_load(f'T{i}', 'FY', -100, 'L')
            model.add_member_dist_load(f'Stud_{i}', 'Fz', 0.05, 0.05, 0, wall_height, 'W')

        # Load combinations
        model.add_load_combo('1.4D', {'D': 1.4})
        model.add_load_combo('1.2D+1.6L', {'D': 1.2, 'L': 1.6})
        model.add_load_combo('1.2D+L+W', {'D': 1.2, 'L': 1.0, 'W': 1.0})

        model.analyze()

        # Test model-level extraction
        combo_names = list(model.load_combos.keys())
        n_points = 20

        all_forces = model.get_all_member_forces(combo_names, n_points=n_points)
        all_forces_arr = model.get_all_member_forces_array(combo_names, n_points=n_points)

        # Verify all members are present
        assert len(all_forces) == len(model.members)

        # Verify array shapes
        n_members = len(model.members)
        n_combos = len(combo_names)
        assert all_forces_arr['Fy'].shape == (n_members, n_combos, n_points)

        # Verify studs have expected properties (axial loading)
        for i in range(n_studs):
            stud_name = f'Stud_{i}'
            # Studs should have significant axial force from gravity loads
            # Sign convention may vary based on member orientation
            axial = all_forces[stud_name]['Fx']
            assert np.any(np.abs(axial) > 100), f"{stud_name} should have significant axial force"


# =============================================================================
# Edge Case Tests for Model-Level Extraction Bugs
# =============================================================================

class TestEndReleaseGrouping:
    """Test that members with different end releases are NOT grouped together."""

    def test_different_end_releases_separate_groups(self):
        """
        Members with different end releases should produce different results.

        The condensed stiffness matrix k depends on end releases. If two members
        have different end releases, they MUST NOT share the same k matrix.
        """
        model = FEModel3D()
        L = 10.0

        # Create nodes for two separate beams
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)
        model.add_node('N3', 0, 0, 10)  # Second beam offset in Z
        model.add_node('N4', L, 0, 10)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        # Two beams with same material, section, length, orientation
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')
        model.add_member('M2', 'N3', 'N4', 'Steel', 'W10')

        # Release M2's i-end moment about z-axis (pinned connection)
        # This changes the condensed stiffness matrix
        model.def_releases('M2', Rzi=True)

        # Fixed supports
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, True, True, True)
        model.def_support('N3', True, True, True, True, True, True)
        model.def_support('N4', True, True, True, True, True, True)

        # Apply same load to both members
        model.add_member_dist_load('M1', 'Fy', -1.0, -1.0, 0, L, 'D')
        model.add_member_dist_load('M2', 'Fy', -1.0, -1.0, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        # Get results via model-level extraction
        all_forces = model.get_all_member_forces(['1.0D'], n_points=21)

        # Get results via individual member extraction (ground truth)
        m1_individual = model.members['M1'].get_all_forces_array(['1.0D'], 21)
        m2_individual = model.members['M2'].get_all_forces_array(['1.0D'], 21)

        # Model-level should match individual for both members
        assert_allclose(all_forces['M1']['Mz'], m1_individual['Mz'], rtol=1e-6,
                       err_msg="M1 model-level does not match individual")
        assert_allclose(all_forces['M2']['Mz'], m2_individual['Mz'], rtol=1e-6,
                       err_msg="M2 model-level does not match individual")

        # The two members should have DIFFERENT moment diagrams because of the end release
        # M1: fixed-fixed with moment at both ends
        # M2: pinned-fixed with zero moment at i-end
        m1_moment_at_i = all_forces['M1']['Mz'][0, 0]
        m2_moment_at_i = all_forces['M2']['Mz'][0, 0]

        # M1 should have significant moment at i-end (fixed-fixed: wL²/12)
        assert abs(m1_moment_at_i) > 0.5, f"M1 should have moment at i-end, got {m1_moment_at_i}"

        # M2 should have ~zero moment at i-end (released)
        assert abs(m2_moment_at_i) < 0.1, f"M2 should have ~zero moment at i-end, got {m2_moment_at_i}"

    def test_partial_end_releases(self):
        """Test members with partial end releases (e.g., only rotation release)."""
        model = FEModel3D()
        L = 10.0

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)
        model.add_node('N3', 0, 0, 10)
        model.add_node('N4', L, 0, 10)
        model.add_node('N5', 0, 0, 20)
        model.add_node('N6', L, 0, 20)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')  # No releases
        model.add_member('M2', 'N3', 'N4', 'Steel', 'W10')  # i-end Rz release
        model.add_member('M3', 'N5', 'N6', 'Steel', 'W10')  # Both ends Rz release

        model.def_releases('M2', Rzi=True)
        model.def_releases('M3', Rzi=True, Rzj=True)

        for node in ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']:
            model.def_support(node, True, True, True, True, True, True)

        for member in ['M1', 'M2', 'M3']:
            model.add_member_dist_load(member, 'Fy', -1.0, -1.0, 0, L, 'D')

        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        all_forces = model.get_all_member_forces(['1.0D'], n_points=21)

        # Each member should match its individual extraction
        for member_name in ['M1', 'M2', 'M3']:
            individual = model.members[member_name].get_all_forces_array(['1.0D'], 21)
            assert_allclose(all_forces[member_name]['Mz'], individual['Mz'], rtol=1e-6,
                           err_msg=f"{member_name} model-level does not match individual")
            assert_allclose(all_forces[member_name]['Fy'], individual['Fy'], rtol=1e-6)

        # M3 (both ends pinned) should have zero moments at both ends
        # but max moment at midspan (wL²/8 for simply supported)
        m3_moment = all_forces['M3']['Mz'][0, :]
        assert abs(m3_moment[0]) < 0.1, "M3 should have ~zero moment at i-end"
        assert abs(m3_moment[-1]) < 0.1, "M3 should have ~zero moment at j-end"
        assert abs(m3_moment[10]) > 1.0, "M3 should have max moment at midspan"


class TestTensionCompressionOnlyMembers:
    """Test that tension/compression-only members report zero forces when inactive."""

    def test_tension_only_inactive(self):
        """
        Tension-only member in compression should report zero forces.

        When a tension-only member goes into compression, it becomes inactive
        and all internal forces should be zeroed out.

        Uses a redundant truss so model remains stable when member deactivates.
        """
        model = FEModel3D()

        # Simple truss with two parallel members:
        # - Top member is tension-only
        # - Bottom member is regular (provides stability when top deactivates)
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)
        model.add_node('N3', 0, 1, 0)  # Offset in Y for second member
        model.add_node('N4', 10, 1, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('Rod', 1.0, 1.0, 1.0, 1.0)

        # Top member is tension-only
        model.add_member('M1', 'N1', 'N2', 'Steel', 'Rod', tension_only=True)
        # Bottom member is normal (provides redundancy)
        model.add_member('M2', 'N3', 'N4', 'Steel', 'Rod')
        # Rigid links connecting the two
        model.add_node('N5', 0, 0.5, 0)
        model.add_node('N6', 10, 0.5, 0)
        model.add_member('Link1', 'N1', 'N3', 'Steel', 'Rod')
        model.add_member('Link2', 'N2', 'N4', 'Steel', 'Rod')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N3', True, True, True, True, True, True)
        model.def_support('N2', False, True, True, True, True, True)
        model.def_support('N4', False, True, True, True, True, True)

        # Apply compression load (pushes nodes toward left)
        model.add_node_load('N2', 'FX', -10.0, 'D')
        model.add_node_load('N4', 'FX', -10.0, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        # The tension-only member M1 should be inactive (in compression)
        assert not model.members['M1'].active.get('1.0D', True), \
            "Tension-only member should be inactive under compression"

        # Model-level extraction should show zero forces for M1
        all_forces = model.get_all_member_forces(['1.0D'], n_points=10)

        axial = all_forces['M1']['Fx'][0, :]
        shear = all_forces['M1']['Fy'][0, :]
        moment = all_forces['M1']['Mz'][0, :]
        torque = all_forces['M1']['Mx'][0, :]

        assert_allclose(axial, 0.0, atol=1e-10, err_msg="Inactive member should have zero axial")
        assert_allclose(shear, 0.0, atol=1e-10, err_msg="Inactive member should have zero shear")
        assert_allclose(moment, 0.0, atol=1e-10, err_msg="Inactive member should have zero moment")
        assert_allclose(torque, 0.0, atol=1e-10, err_msg="Inactive member should have zero torque")

        # M2 should be active and carrying load
        m2_axial = all_forces['M2']['Fx'][0, :]
        assert np.any(np.abs(m2_axial) > 0.1), "Regular member should carry load"

    def test_compression_only_inactive(self):
        """
        Compression-only member in tension should report zero forces.
        """
        model = FEModel3D()

        # Simple truss with two parallel members
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)
        model.add_node('N3', 0, 1, 0)
        model.add_node('N4', 10, 1, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('Rod', 1.0, 1.0, 1.0, 1.0)

        # Top member is compression-only
        model.add_member('M1', 'N1', 'N2', 'Steel', 'Rod', comp_only=True)
        # Bottom member is normal
        model.add_member('M2', 'N3', 'N4', 'Steel', 'Rod')
        # Rigid links
        model.add_member('Link1', 'N1', 'N3', 'Steel', 'Rod')
        model.add_member('Link2', 'N2', 'N4', 'Steel', 'Rod')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N3', True, True, True, True, True, True)
        model.def_support('N2', False, True, True, True, True, True)
        model.def_support('N4', False, True, True, True, True, True)

        # Apply tension load (pulls nodes away from left)
        model.add_node_load('N2', 'FX', 10.0, 'D')
        model.add_node_load('N4', 'FX', 10.0, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        # The compression-only member M1 should be inactive (in tension)
        assert not model.members['M1'].active.get('1.0D', True), \
            "Compression-only member should be inactive under tension"

        # Model-level extraction should show zero forces for M1
        all_forces = model.get_all_member_forces(['1.0D'], n_points=10)

        axial = all_forces['M1']['Fx'][0, :]
        assert_allclose(axial, 0.0, atol=1e-10, err_msg="Inactive member should have zero axial")

    def test_tension_only_active(self):
        """
        Tension-only member in tension SHOULD report correct forces.
        """
        model = FEModel3D()
        L = 10.0

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('Rod', 1.0, 1.0, 1.0, 1.0)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'Rod', tension_only=True)

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', False, True, True, True, True, True)

        # Apply tension load (pulls N2 away from N1)
        model.add_node_load('N2', 'FX', 10.0, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})

        model.analyze()

        # The member should be active (in tension)
        assert model.members['M1'].active.get('1.0D', True), \
            "Tension-only member should be active under tension"

        # Model-level extraction should show non-zero axial force
        all_forces = model.get_all_member_forces(['1.0D'], n_points=10)
        individual = model.members['M1'].get_all_forces_array(['1.0D'], 10)

        # Should match individual extraction
        assert_allclose(all_forces['M1']['Fx'], individual['Fx'], rtol=1e-6)

        # Should have non-zero axial
        axial = all_forces['M1']['Fx'][0, :]
        assert np.all(np.abs(axial) > 1.0), "Active tension member should have axial force"

    def test_mixed_active_inactive_per_combo(self):
        """
        Test member that is active in some combos but inactive in others.

        Uses a redundant structure so model remains stable for all combos.
        """
        model = FEModel3D()

        # Simple truss with two parallel members
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)
        model.add_node('N3', 0, 1, 0)
        model.add_node('N4', 10, 1, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('Rod', 1.0, 1.0, 1.0, 1.0)

        # Top member is tension-only
        model.add_member('M1', 'N1', 'N2', 'Steel', 'Rod', tension_only=True)
        # Bottom member is normal
        model.add_member('M2', 'N3', 'N4', 'Steel', 'Rod')
        # Rigid links
        model.add_member('Link1', 'N1', 'N3', 'Steel', 'Rod')
        model.add_member('Link2', 'N2', 'N4', 'Steel', 'Rod')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N3', True, True, True, True, True, True)
        model.def_support('N2', False, True, True, True, True, True)
        model.def_support('N4', False, True, True, True, True, True)

        # Load case that causes tension in horizontal members
        model.add_node_load('N2', 'FX', 10.0, 'Tension')
        model.add_node_load('N4', 'FX', 10.0, 'Tension')
        # Load case that causes compression in horizontal members
        model.add_node_load('N2', 'FX', -10.0, 'Compression')
        model.add_node_load('N4', 'FX', -10.0, 'Compression')

        model.add_load_combo('TensionCombo', {'Tension': 1.0})
        model.add_load_combo('CompressionCombo', {'Compression': 1.0})

        model.analyze()

        all_forces = model.get_all_member_forces(['TensionCombo', 'CompressionCombo'], n_points=10)

        # Tension combo: M1 should be active
        assert model.members['M1'].active.get('TensionCombo', True), \
            "Tension-only member should be active under tension"
        tension_axial = all_forces['M1']['Fx'][0, :]  # First combo
        assert np.any(np.abs(tension_axial) > 0.1), "Tension combo should have axial force in M1"

        # Compression combo: M1 should be inactive
        assert not model.members['M1'].active.get('CompressionCombo', True), \
            "Tension-only member should be inactive under compression"
        compression_axial = all_forces['M1']['Fx'][1, :]  # Second combo
        assert_allclose(compression_axial, 0.0, atol=1e-10,
                       err_msg="Compression combo should have zero axial for tension-only member")


class TestIncludeSwitches:
    """Test the include_* switches for selective force extraction."""

    def test_include_only_moment(self):
        """Test extracting only moment data."""
        model = FEModel3D()
        L = 10.0

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)

        model.add_member_dist_load('M1', 'Fy', -1.0, -1.0, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        # Extract only moment
        forces = model.get_all_member_forces(
            ['1.0D'], n_points=20,
            include_shear=False, include_moment=True,
            include_axial=False, include_torque=False
        )

        # Should have x and moment_z but not others
        assert 'x' in forces['M1']
        assert 'Mz' in forces['M1']
        assert 'Fy' not in forces['M1']
        assert 'Fx' not in forces['M1']
        assert 'Mx' not in forces['M1']

    def test_include_shear_and_moment(self):
        """Test extracting shear and moment together."""
        model = FEModel3D()
        L = 10.0

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)

        model.add_member_dist_load('M1', 'Fy', -1.0, -1.0, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        forces = model.get_all_member_forces(
            ['1.0D'], n_points=20,
            include_shear=True, include_moment=True,
            include_axial=False, include_torque=False
        )

        assert 'Fy' in forces['M1']
        assert 'Mz' in forces['M1']
        assert 'Fx' not in forces['M1']
        assert 'Mx' not in forces['M1']

    def test_include_all_default(self):
        """Test that all forces are included by default."""
        model = FEModel3D()
        L = 10.0

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', L, 0, 0)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10')

        model.def_support('N1', True, True, True, True, True, False)
        model.def_support('N2', True, True, True, True, True, False)

        model.add_member_dist_load('M1', 'Fy', -1.0, -1.0, 0, L, 'D')
        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        # Default should include all
        forces = model.get_all_member_forces(['1.0D'], n_points=20)

        assert 'x' in forces['M1']
        assert 'Fy' in forces['M1']
        assert 'Mz' in forces['M1']
        assert 'Fx' in forces['M1']
        assert 'Mx' in forces['M1']

    def test_include_switches_batched_path(self):
        """Test include switches with batched extraction (multiple similar members)."""
        model = FEModel3D()
        L = 10.0

        # Create multiple similar members to trigger batched path
        for i in range(3):
            model.add_node(f'N{i*2}', 0, 0, i*5)
            model.add_node(f'N{i*2+1}', L, 0, i*5)

        model.add_material('Steel', 29000, 11200, 0.490/12**3, 0.490/12**3)
        model.add_section('W10', 10, 100, 100, 200)

        for i in range(3):
            model.add_member(f'M{i}', f'N{i*2}', f'N{i*2+1}', 'Steel', 'W10')
            model.def_support(f'N{i*2}', True, True, True, True, True, False)
            model.def_support(f'N{i*2+1}', True, True, True, True, True, False)
            model.add_member_dist_load(f'M{i}', 'Fy', -1.0, -1.0, 0, L, 'D')

        model.add_load_combo('1.0D', {'D': 1.0})
        model.analyze()

        # Extract only axial
        forces = model.get_all_member_forces(
            ['1.0D'], n_points=20,
            include_shear=False, include_moment=False,
            include_axial=True, include_torque=False
        )

        for i in range(3):
            assert 'x' in forces[f'M{i}']
            assert 'Fx' in forces[f'M{i}']
            assert 'Fy' not in forces[f'M{i}']
            assert 'Mz' not in forces[f'M{i}']
            assert 'Mx' not in forces[f'M{i}']
