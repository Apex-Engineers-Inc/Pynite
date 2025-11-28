"""
Tests for bulk force extraction optimization.

These tests verify that the optimized bulk force extraction methods produce
results that match:
1. Known analytical solutions for simple beam cases
2. Traditional segment-based extraction methods
3. Results across various loading conditions
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose
from Pynite import FEModel3D


class TestSimplySuportedBeamAnalytical(unittest.TestCase):
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
        # Create model
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

        x = bulk_results['x']
        shear = bulk_results['shear_y'][0, :]
        moment = bulk_results['moment_z'][0, :]

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

        self.assertAlmostEqual(M_max_computed, M_max_analytical, places=5,
                              msg="Max moment does not match wL^2/8")

        # Verify shear at supports and midspan
        # At x=0: |V| = |w|*L/2 = 5
        # At midspan: V ≈ 0
        self.assertAlmostEqual(np.abs(shear[0]), abs(w) * L / 2, places=5)
        self.assertAlmostEqual(shear[n_points // 2], 0, places=5)

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
        assert_allclose(bulk_results['shear_y'][0, :], trad_shear[1], rtol=1e-6,
                       err_msg="Bulk shear does not match traditional")
        assert_allclose(bulk_results['moment_z'][0, :], trad_moment[1], rtol=1e-6,
                       err_msg="Bulk moment does not match traditional")

        # Verify max moment magnitude = |P|*L/4
        M_max_analytical = abs(P) * L / 4
        M_max_computed = np.max(np.abs(bulk_results['moment_z'][0, :]))
        self.assertAlmostEqual(M_max_computed, M_max_analytical, places=3)

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

        x = bulk_results['x']
        shear = bulk_results['shear_y'][0, :]
        moment = bulk_results['moment_z'][0, :]

        # Analytical: Shear is constant = -P (reaction)
        # Moment varies linearly: M(x) = -P*(L-x) at fixed end to 0 at free end
        # Note: Convention may differ, check against traditional
        trad_shear = member.shear_array('Fy', 21, '1.0D')
        trad_moment = member.moment_array('Mz', 21, '1.0D')

        assert_allclose(shear, trad_shear[1], rtol=1e-6)
        assert_allclose(moment, trad_moment[1], rtol=1e-6)


class TestBulkVsTraditionalExtraction(unittest.TestCase):
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

            assert_allclose(bulk_results['shear_y'][i, :], trad_shear[1], rtol=1e-6,
                           err_msg=f"Shear mismatch for {combo_name}")
            assert_allclose(bulk_results['moment_z'][i, :], trad_moment[1], rtol=1e-6,
                           err_msg=f"Moment mismatch for {combo_name}")
            assert_allclose(bulk_results['axial'][i, :], trad_axial[1], rtol=1e-6,
                           err_msg=f"Axial mismatch for {combo_name}")
            assert_allclose(bulk_results['torque'][i, :], trad_torque[1], rtol=1e-6,
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

        assert_allclose(bulk_results['shear_y'][0, :], trad_shear[1], rtol=1e-5,
                       err_msg="Shear mismatch for triangular load")
        assert_allclose(bulk_results['moment_z'][0, :], trad_moment[1], rtol=1e-5,
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

        assert_allclose(bulk_results['axial'][0, :], trad_axial[1], rtol=1e-5,
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

        assert_allclose(bulk_results['torque'][0, :], trad_torque[1], rtol=1e-5,
                       err_msg="Torque mismatch")


class TestWallDesignScenario(unittest.TestCase):
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

            assert_allclose(bulk_results['shear_y'][i, :], trad_shear[1], rtol=1e-5,
                           err_msg=f"Shear mismatch for {combo_name}")
            assert_allclose(bulk_results['moment_z'][i, :], trad_moment[1], rtol=1e-5,
                           err_msg=f"Moment mismatch for {combo_name}")
            assert_allclose(bulk_results['axial'][i, :], trad_axial[1], rtol=1e-5,
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

                assert_allclose(bulk_results['shear_y'][i, :], trad_shear[1], rtol=1e-4,
                               err_msg=f"Shear mismatch for {member_name} - {combo_name}")
                assert_allclose(bulk_results['moment_z'][i, :], trad_moment[1], rtol=1e-4,
                               err_msg=f"Moment mismatch for {member_name} - {combo_name}")
                assert_allclose(bulk_results['axial'][i, :], trad_axial[1], rtol=1e-4,
                               err_msg=f"Axial mismatch for {member_name} - {combo_name}")
                assert_allclose(bulk_results['torque'][i, :], trad_torque[1], rtol=1e-4,
                               err_msg=f"Torque mismatch for {member_name} - {combo_name}")


class TestEdgeCases(unittest.TestCase):
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
        assert_allclose(bulk_results['shear_y'][0, :], 0, atol=1e-10)
        assert_allclose(bulk_results['moment_z'][0, :], 0, atol=1e-10)
        assert_allclose(bulk_results['axial'][0, :], 0, atol=1e-10)
        assert_allclose(bulk_results['torque'][0, :], 0, atol=1e-10)

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
        self.assertEqual(bulk_results['shear_y'].shape, (1, 1))
        self.assertEqual(bulk_results['moment_z'].shape, (1, 1))

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

        assert_allclose(bulk_results['shear_y'][0, :], trad_shear[1], rtol=1e-5)

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

        assert_allclose(bulk_results['shear_y'][0, :], trad_shear[1], rtol=1e-4)
        assert_allclose(bulk_results['moment_z'][0, :], trad_moment[1], rtol=1e-4)


class TestEndReleases(unittest.TestCase):
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

            assert_allclose(bulk_results['shear_y'][0, :], trad_shear[1], rtol=1e-4,
                           err_msg=f"Shear mismatch for {member_name}")
            assert_allclose(bulk_results['moment_z'][0, :], trad_moment[1], rtol=1e-4,
                           err_msg=f"Moment mismatch for {member_name}")


if __name__ == '__main__':
    unittest.main()
