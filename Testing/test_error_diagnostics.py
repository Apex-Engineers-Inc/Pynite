# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 D. Craig Brinck, SE; tamalone1

Tests for the _diagnose_singularity function in Analysis.py.
These tests verify that meaningful error messages are provided when
analysis fails due to various modeling issues.
"""

import unittest
import warnings
from Pynite import FEModel3D
import sys
from io import StringIO


class TestDiagnoseSingularity(unittest.TestCase):
    """Tests for the _diagnose_singularity diagnostic function."""

    def setUp(self):
        # Suppress printed output temporarily
        sys.stdout = StringIO()
        # Treat warnings as errors so scipy's MatrixRankWarning raises an exception
        warnings.filterwarnings('error')

    def tearDown(self):
        # Reset the print function to normal
        sys.stdout = sys.__stdout__
        # Reset warning filters
        warnings.resetwarnings()

    def _create_basic_model(self):
        """Helper to create a basic model with material and section defined."""
        model = FEModel3D()
        model.add_material('Steel', 29000, 11200, 0.3, 490/1000/12**3)
        model.add_section('W8x31', 9.13, 37.1, 110, 0.536)
        return model

    def test_no_supports_defined(self):
        """Test that missing supports are correctly identified."""
        model = self._create_basic_model()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W8x31')
        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        model.add_member_dist_load('M1', 'FY', -1, -1, case='Case 1')

        with self.assertRaises(Exception) as context:
            model.analyze_linear(check_stability=False)

        self.assertIn('NO SUPPORTS DEFINED', str(context.exception))

    def test_disconnected_node(self):
        """Test that disconnected nodes are correctly identified."""
        model = self._create_basic_model()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 240, 0, 0)  # Disconnected node
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W8x31')
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', False, True, True, True, True, True)
        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        model.add_member_dist_load('M1', 'FY', -1, -1, case='Case 1')

        with self.assertRaises(Exception) as context:
            model.analyze_linear(check_stability=False)

        self.assertIn('DISCONNECTED NODES', str(context.exception))
        self.assertIn('N3', str(context.exception))

    def test_multiple_disconnected_nodes(self):
        """Test that multiple disconnected nodes are identified."""
        model = self._create_basic_model()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 240, 0, 0)  # Disconnected
        model.add_node('N4', 360, 0, 0)  # Disconnected
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W8x31')
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', False, True, True, True, True, True)
        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        model.add_member_dist_load('M1', 'FY', -1, -1, case='Case 1')

        with self.assertRaises(Exception) as context:
            model.analyze_linear(check_stability=False)

        error_msg = str(context.exception)
        self.assertIn('DISCONNECTED NODES', error_msg)
        self.assertIn('N3', error_msg)
        self.assertIn('N4', error_msg)

    def test_unstable_dof_rotation(self):
        """Test that unstable rotational DOFs are identified for a truss-like model."""
        model = self._create_basic_model()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 60, 60, 0)

        # Create truss members with moment releases at both ends
        model.add_member('M1', 'N1', 'N3', 'Steel', 'W8x31')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'W8x31')

        # Release moments at all ends (truss behavior)
        model.def_releases('M1', Ryi=True, Rzi=True, Ryj=True, Rzj=True)
        model.def_releases('M2', Ryi=True, Rzi=True, Ryj=True, Rzj=True)

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, True, True, True)
        # N3 has no rotational support and no rotational stiffness from members

        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        model.add_node_load('N3', 'FY', -10, case='Case 1')

        with self.assertRaises(Exception) as context:
            model.analyze_linear(check_stability=False)

        error_msg = str(context.exception)
        # Should identify unstable DOFs or mechanism
        self.assertTrue(
            'UNSTABLE DEGREES OF FREEDOM' in error_msg or
            'POTENTIAL MECHANISM' in error_msg or
            'SINGULAR STIFFNESS MATRIX' in error_msg
        )

    def test_unsupported_translation(self):
        """Test that unsupported translational DOFs are identified."""
        model = self._create_basic_model()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 240, 0, 0)
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W8x31')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'W8x31')

        # Only partially support - missing X translation restraint globally
        model.def_support('N1', False, True, True, True, True, True)  # DX not supported
        model.def_support('N3', False, True, True, True, True, True)  # DX not supported
        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        model.add_member_dist_load('M1', 'FY', -1, -1, case='Case 1')

        with self.assertRaises(Exception) as context:
            model.analyze_linear(check_stability=False, sparse=True)

        error_msg = str(context.exception)
        # Should identify the global instability
        self.assertIn('SINGULAR STIFFNESS MATRIX', error_msg)

    def test_cantilever_without_support(self):
        """Test that a cantilever without proper support is identified."""
        model = self._create_basic_model()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W8x31')

        # Only support one DOF - not enough to prevent all rigid body motion
        model.def_support('N1', True, False, False, False, False, False)  # Only DX
        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        model.add_member_dist_load('M1', 'FY', -1, -1, case='Case 1')

        with self.assertRaises(Exception) as context:
            model.analyze_linear(check_stability=False, sparse=True)

        error_msg = str(context.exception)
        # Should identify unstable DOFs
        self.assertIn('SINGULAR STIFFNESS MATRIX', error_msg)

    def test_valid_model_no_error(self):
        """Test that a valid, stable model does not trigger diagnostics."""
        model = self._create_basic_model()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W8x31')
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', False, True, True, True, True, True)
        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        model.add_member_dist_load('M1', 'FY', -1, -1, case='Case 1')

        # This should not raise an exception
        model.analyze_linear(check_stability=False)

        # Check that the model was solved
        self.assertEqual(model.solution, 'Linear')

    def test_analyze_method_diagnostics(self):
        """Test that the analyze() method also uses diagnostics."""
        model = self._create_basic_model()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W8x31')
        # No supports - should fail
        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        model.add_member_dist_load('M1', 'FY', -1, -1, case='Case 1')

        with self.assertRaises(Exception) as context:
            model.analyze(check_stability=False)

        self.assertIn('NO SUPPORTS DEFINED', str(context.exception))

    def test_partially_supported_disconnected_node(self):
        """Test that a partially supported disconnected node is identified."""
        model = self._create_basic_model()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 240, 0, 0)  # Disconnected
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W8x31')
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', False, True, True, True, True, True)
        # Partially support N3 - still disconnected
        model.def_support('N3', True, False, False, False, False, False)
        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        model.add_member_dist_load('M1', 'FY', -1, -1, case='Case 1')

        with self.assertRaises(Exception) as context:
            model.analyze_linear(check_stability=False)

        # N3 is disconnected but partially supported - should still be flagged
        self.assertIn('DISCONNECTED NODES', str(context.exception))

    def test_fully_supported_disconnected_node_ignored(self):
        """Test that a fully supported disconnected node is NOT flagged."""
        model = self._create_basic_model()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 240, 0, 0)  # Disconnected but fully supported
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W8x31')
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', False, True, True, True, True, True)
        # Fully support N3 - should not cause issues
        model.def_support('N3', True, True, True, True, True, True)
        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        model.add_member_dist_load('M1', 'FY', -1, -1, case='Case 1')

        # This should not raise an exception because N3 is fully supported
        model.analyze_linear(check_stability=False)
        self.assertEqual(model.solution, 'Linear')

    def test_error_message_format(self):
        """Test that error messages follow expected format."""
        model = self._create_basic_model()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W8x31')
        model.add_load_combo('Combo 1', {'Case 1': 1.0})

        with self.assertRaises(Exception) as context:
            model.analyze_linear(check_stability=False)

        error_msg = str(context.exception)
        # Check for expected format elements
        self.assertIn('SINGULAR STIFFNESS MATRIX', error_msg)
        self.assertIn('Root cause(s) identified', error_msg)


class TestDiagnoseSingularityWithPlates(unittest.TestCase):
    """Tests for diagnostics with plate elements."""

    def setUp(self):
        sys.stdout = StringIO()
        warnings.filterwarnings('error')

    def tearDown(self):
        sys.stdout = sys.__stdout__
        warnings.resetwarnings()

    def test_disconnected_node_with_plates(self):
        """Test that disconnected nodes are identified even with plate models."""
        model = FEModel3D()
        model.add_material('Concrete', 3600, 1500, 0.17, 150/1000/12**3)

        # Create a simple plate
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)
        model.add_node('N3', 10, 10, 0)
        model.add_node('N4', 0, 10, 0)
        model.add_node('N5', 20, 0, 0)  # Disconnected

        model.add_plate('P1', 'N1', 'N2', 'N3', 'N4', 0.5, 'Concrete')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, True, True, True)
        model.def_support('N3', False, False, True, True, True, True)
        model.def_support('N4', False, False, True, True, True, True)

        model.add_load_combo('Combo 1', {'Case 1': 1.0})

        with self.assertRaises(Exception) as context:
            model.analyze_linear(check_stability=False)

        self.assertIn('DISCONNECTED NODES', str(context.exception))
        self.assertIn('N5', str(context.exception))


class TestDiagnoseSingularityWithSprings(unittest.TestCase):
    """Tests for diagnostics with spring elements."""

    def setUp(self):
        sys.stdout = StringIO()
        warnings.filterwarnings('error')

    def tearDown(self):
        sys.stdout = sys.__stdout__
        warnings.resetwarnings()

    def test_spring_connects_node(self):
        """Test that springs properly connect nodes."""
        model = FEModel3D()
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 10, 0, 0)

        # Connect with a spring instead of a member
        model.add_spring('S1', 'N1', 'N2', 1000, tension_only=False, comp_only=False)

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', False, True, True, True, True, True)

        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        model.add_node_load('N2', 'FX', 10, case='Case 1')

        # This should work - spring connects the nodes
        model.analyze_linear(check_stability=False)
        self.assertEqual(model.solution, 'Linear')


if __name__ == '__main__':
    unittest.main()
