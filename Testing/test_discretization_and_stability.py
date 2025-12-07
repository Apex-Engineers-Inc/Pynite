# -*- coding: utf-8 -*-
"""
Tests for PhysMember discretization and stability detection features.

This module tests:
1. PhysMember.descritize() - automatic subdivision at intermediate nodes
2. _identify_floating_nodes() - detection of disconnected nodes
3. _identify_unstable_nodes() - detection of unstable DOFs
4. Improved error messages for unstable structures

MIT License
Copyright (c) 2020 D. Craig Brinck, SE; tamalone1
"""

import pytest
import sys
from io import StringIO
from math import isclose

from Pynite import FEModel3D
from Pynite.PhysMember import PhysMember
from Pynite import Analysis


# =============================================================================
# Fixtures for common model setups
# =============================================================================

@pytest.fixture
def basic_model():
    """Create a basic model with material and section defined."""
    model = FEModel3D()
    model.add_material('Steel', 29000, 11200, 0.3, 0.000284)
    model.add_section('W10x30', 8.84, 16.7, 170, 0.622)
    return model


@pytest.fixture
def wood_model():
    """Create a model with wood material properties for wall framing tests."""
    model = FEModel3D()
    model.add_material('Wood', 1600000, 100000, 0.3, 0.000017)  # psi units
    model.add_section('2x4', 5.25, 5.359, 0.984, 1.0)
    model.add_section('2x6', 8.25, 20.8, 1.505, 1.0)
    return model


# =============================================================================
# Tests for PhysMember.descritize() - Basic Cases
# =============================================================================

class TestDiscretizationBasic:
    """Tests for basic discretization functionality."""

    def test_no_intermediate_nodes(self, basic_model):
        """Member with no intermediate nodes should have 1 sub-member."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)

        phys = PhysMember(model, 'M1', model.nodes['N1'], model.nodes['N2'],
                         'Steel', 'W10x30')
        model.members['M1'] = phys

        phys.descritize()

        assert len(phys.sub_members) == 1
        assert 'M1a' in phys.sub_members

    def test_one_intermediate_node(self, basic_model):
        """Member with one intermediate node should have 2 sub-members."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 60, 0, 0)  # Midpoint

        phys = PhysMember(model, 'M1', model.nodes['N1'], model.nodes['N2'],
                         'Steel', 'W10x30')
        model.members['M1'] = phys

        phys.descritize()

        assert len(phys.sub_members) == 2
        assert 'M1a' in phys.sub_members
        assert 'M1b' in phys.sub_members

        # Verify sub-member connectivity
        assert phys.sub_members['M1a'].i_node is model.nodes['N1']
        assert phys.sub_members['M1a'].j_node is model.nodes['N3']
        assert phys.sub_members['M1b'].i_node is model.nodes['N3']
        assert phys.sub_members['M1b'].j_node is model.nodes['N2']

    def test_multiple_intermediate_nodes(self, basic_model):
        """Member with multiple intermediate nodes should discretize correctly."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 30, 0, 0)
        model.add_node('N4', 60, 0, 0)
        model.add_node('N5', 90, 0, 0)

        phys = PhysMember(model, 'M1', model.nodes['N1'], model.nodes['N2'],
                         'Steel', 'W10x30')
        model.members['M1'] = phys

        phys.descritize()

        assert len(phys.sub_members) == 4
        # Verify correct ordering by checking x-coordinates increase
        x_coords = []
        for sub in phys.sub_members.values():
            x_coords.append(sub.i_node.X)
        x_coords.append(list(phys.sub_members.values())[-1].j_node.X)
        assert x_coords == sorted(x_coords)

    def test_nodes_added_after_member_creation(self, basic_model):
        """Nodes added after member creation should still cause discretization."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)

        phys = PhysMember(model, 'M1', model.nodes['N1'], model.nodes['N2'],
                         'Steel', 'W10x30')
        model.members['M1'] = phys

        # Add intermediate node AFTER member creation
        model.add_node('N3', 60, 0, 0)

        phys.descritize()

        assert len(phys.sub_members) == 2


# =============================================================================
# Tests for PhysMember.descritize() - 3D Cases
# =============================================================================

class TestDiscretization3D:
    """Tests for discretization with 3D member orientations."""

    def test_vertical_member(self, basic_model):
        """Vertical member should discretize at intermediate nodes."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 0, 120, 0)
        model.add_node('N3', 0, 60, 0)

        phys = PhysMember(model, 'M1', model.nodes['N1'], model.nodes['N2'],
                         'Steel', 'W10x30')
        model.members['M1'] = phys

        phys.descritize()

        assert len(phys.sub_members) == 2

    def test_diagonal_member_xy(self, basic_model):
        """Diagonal member in XY plane should discretize correctly."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 100, 100, 0)
        model.add_node('N3', 50, 50, 0)  # Midpoint on diagonal

        phys = PhysMember(model, 'M1', model.nodes['N1'], model.nodes['N2'],
                         'Steel', 'W10x30')
        model.members['M1'] = phys

        phys.descritize()

        assert len(phys.sub_members) == 2

    def test_3d_diagonal_member(self, basic_model):
        """3D diagonal member should discretize at collinear nodes."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 100, 100, 100)
        model.add_node('N3', 50, 50, 50)
        model.add_node('N4', 25, 25, 25)

        phys = PhysMember(model, 'M1', model.nodes['N1'], model.nodes['N2'],
                         'Steel', 'W10x30')
        model.members['M1'] = phys

        phys.descritize()

        assert len(phys.sub_members) == 3


# =============================================================================
# Tests for Discretization Edge Cases
# =============================================================================

class TestDiscretizationEdgeCases:
    """Tests for edge cases in discretization."""

    def test_node_at_member_start(self, basic_model):
        """Node at same location as i_node should not cause extra sub-member."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 0, 0, 0)  # Same location as N1

        phys = PhysMember(model, 'M1', model.nodes['N1'], model.nodes['N2'],
                         'Steel', 'W10x30')
        model.members['M1'] = phys

        # Should not crash, just produce warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phys.descritize()

        # Node at same location should be ignored (would create zero-length member)
        assert len(phys.sub_members) == 1

    def test_node_at_member_end(self, basic_model):
        """Node at same location as j_node should not cause extra sub-member."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 120, 0, 0)  # Same location as N2

        phys = PhysMember(model, 'M1', model.nodes['N1'], model.nodes['N2'],
                         'Steel', 'W10x30')
        model.members['M1'] = phys

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phys.descritize()

        # Node at j_node location should be ignored
        assert len(phys.sub_members) == 1

    def test_node_slightly_off_axis(self, basic_model):
        """Node slightly off the member axis should NOT cause discretization."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 60, 0.001, 0)  # Slightly off in Y

        phys = PhysMember(model, 'M1', model.nodes['N1'], model.nodes['N2'],
                         'Steel', 'W10x30')
        model.members['M1'] = phys

        phys.descritize()

        # Node off axis should not cause discretization
        assert len(phys.sub_members) == 1

    def test_floating_point_precision(self, basic_model):
        """Nodes with floating point coordinates should discretize correctly."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        # 1/3 of 120 = 40, but use floating point representation
        model.add_node('N3', 120/3, 0, 0)

        phys = PhysMember(model, 'M1', model.nodes['N1'], model.nodes['N2'],
                         'Steel', 'W10x30')
        model.members['M1'] = phys

        phys.descritize()

        assert len(phys.sub_members) == 2

    def test_tiny_offset_still_discretizes(self, basic_model):
        """Very tiny offset (floating point error) should still discretize."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 60.0000000001, 0, 0)  # Tiny X offset

        phys = PhysMember(model, 'M1', model.nodes['N1'], model.nodes['N2'],
                         'Steel', 'W10x30')
        model.members['M1'] = phys

        phys.descritize()

        # Very tiny offset should still be considered on the member
        assert len(phys.sub_members) == 2


# =============================================================================
# Tests for Wood Wall Framing Scenarios
# =============================================================================

class TestWoodWallFraming:
    """Tests for typical wood wall framing scenarios."""

    def test_studs_connected_to_plates(self, wood_model):
        """Studs with nodes on plate lines should create connected structure."""
        model = wood_model

        # Bottom plate from (0,0,0) to (48,0,0)
        model.add_node('BP1', 0, 0, 0)
        model.add_node('BP2', 48, 0, 0)

        # Top plate from (0,96,0) to (48,96,0)
        model.add_node('TP1', 0, 96, 0)
        model.add_node('TP2', 48, 96, 0)

        # Intermediate stud nodes - ON the plate lines
        model.add_node('S_B', 24, 0, 0)   # On bottom plate
        model.add_node('S_T', 24, 96, 0)  # On top plate

        # Add plates
        model.add_member('BPlate', 'BP1', 'BP2', 'Wood', '2x6')
        model.add_member('TPlate', 'TP1', 'TP2', 'Wood', '2x6')

        # Add studs
        model.add_member('Stud1', 'BP1', 'TP1', 'Wood', '2x4')
        model.add_member('Stud2', 'S_B', 'S_T', 'Wood', '2x4')
        model.add_member('Stud3', 'BP2', 'TP2', 'Wood', '2x4')

        # Support bottom plate
        model.def_support('BP1', True, True, True, True, True, True)
        model.def_support('BP2', True, True, True, False, True, True)
        model.def_support('S_B', True, True, True, False, True, True)

        # Add load and combo
        model.add_node_load('TP1', 'FY', -1000, 'Case 1')
        model.add_load_combo('Combo 1', {'Case 1': 1.0})

        # Prepare model to trigger discretization
        Analysis._prepare_model(model)

        # Check that plates are discretized at stud locations
        assert len(model.members['BPlate'].sub_members) == 2
        assert len(model.members['TPlate'].sub_members) == 2

        # Verify connectivity
        s_b = model.nodes['S_B']
        bp_connected = any(sub.i_node is s_b or sub.j_node is s_b
                          for sub in model.members['BPlate'].sub_members.values())
        assert bp_connected, "Stud bottom should be connected to bottom plate"

    def test_studs_at_16_inch_oc(self, wood_model):
        """Multiple studs at 16\" on center should all connect to plates."""
        model = wood_model

        # 48" wall with studs at 0, 16, 32, 48
        model.add_node('BP1', 0, 0, 0)
        model.add_node('BP2', 48, 0, 0)
        model.add_node('TP1', 0, 96, 0)
        model.add_node('TP2', 48, 96, 0)

        # Add intermediate stud nodes
        for x in [16, 32]:
            model.add_node(f'S_B_{x}', x, 0, 0)
            model.add_node(f'S_T_{x}', x, 96, 0)

        # Add plates
        model.add_member('BPlate', 'BP1', 'BP2', 'Wood', '2x6')
        model.add_member('TPlate', 'TP1', 'TP2', 'Wood', '2x6')

        # Add studs
        model.add_member('Stud1', 'BP1', 'TP1', 'Wood', '2x4')
        model.add_member('Stud2', 'S_B_16', 'S_T_16', 'Wood', '2x4')
        model.add_member('Stud3', 'S_B_32', 'S_T_32', 'Wood', '2x4')
        model.add_member('Stud4', 'BP2', 'TP2', 'Wood', '2x4')

        Analysis._prepare_model(model)

        # Plates should be discretized into 3 segments each
        assert len(model.members['BPlate'].sub_members) == 3
        assert len(model.members['TPlate'].sub_members) == 3


# =============================================================================
# Tests for _identify_floating_nodes()
# =============================================================================

class TestIdentifyFloatingNodes:
    """Tests for the _identify_floating_nodes function."""

    def test_no_floating_nodes(self, basic_model):
        """Fully connected structure should have no floating nodes."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 240, 0, 0)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10x30')
        model.add_member('M2', 'N2', 'N3', 'Steel', 'W10x30')

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N3', True, True, True, True, True, True)

        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        Analysis._prepare_model(model)

        floating = Analysis._identify_floating_nodes(model)
        assert len(floating) == 0

    def test_single_floating_node(self, basic_model):
        """Completely isolated node should be identified as floating."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 60, 60, 0)  # Not connected to anything

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10x30')
        model.def_support('N1', True, True, True, True, True, True)

        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        Analysis._prepare_model(model)

        floating = Analysis._identify_floating_nodes(model)
        assert 'N3' in floating
        assert len(floating) == 1

    def test_floating_substructure(self, basic_model):
        """Isolated substructure (member not connected to supports) should be floating."""
        model = basic_model

        # Main structure with supports
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10x30')
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, True, True, True)

        # Floating member not connected to supports
        model.add_node('N3', 0, 100, 0)
        model.add_node('N4', 120, 100, 0)
        model.add_member('M2', 'N3', 'N4', 'Steel', 'W10x30')

        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        Analysis._prepare_model(model)

        floating = Analysis._identify_floating_nodes(model)
        assert 'N3' in floating
        assert 'N4' in floating
        assert 'N1' not in floating
        assert 'N2' not in floating

    def test_floating_stud_in_wall(self, wood_model):
        """Stud with ends off plate lines should have floating nodes."""
        model = wood_model

        # Plates
        model.add_node('BP1', 0, 0, 0)
        model.add_node('BP2', 48, 0, 0)
        model.add_node('TP1', 0, 96, 0)
        model.add_node('TP2', 48, 96, 0)

        model.add_member('BPlate', 'BP1', 'BP2', 'Wood', '2x6')
        model.add_member('TPlate', 'TP1', 'TP2', 'Wood', '2x6')

        # Corner studs (connected)
        model.add_member('Stud1', 'BP1', 'TP1', 'Wood', '2x4')
        model.add_member('Stud2', 'BP2', 'TP2', 'Wood', '2x4')

        # Floating stud - both ends off plate lines!
        model.add_node('S_B', 24, 0.01, 0)   # Off bottom plate
        model.add_node('S_T', 24, 95.99, 0)  # Off top plate
        model.add_member('Stud3', 'S_B', 'S_T', 'Wood', '2x4')

        model.def_support('BP1', True, True, True, True, True, True)
        model.def_support('BP2', True, True, True, True, True, True)

        model.add_load_combo('Combo 1', {'Case 1': 1.0})
        Analysis._prepare_model(model)

        floating = Analysis._identify_floating_nodes(model)
        assert 'S_B' in floating, "S_B should be floating (not on plate)"
        assert 'S_T' in floating, "S_T should be floating (not on plate)"


# =============================================================================
# Tests for _identify_unstable_nodes()
# =============================================================================

class TestIdentifyUnstableNodes:
    """Tests for the _identify_unstable_nodes function."""

    def test_no_unstable_nodes_in_stable_structure(self, basic_model):
        """Stable structure should have no unstable nodes identified."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10x30')
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', False, True, False, False, False, False)

        model.add_node_load('N2', 'FY', -10, 'Case 1')
        model.add_load_combo('Combo 1', {'Case 1': 1.0})

        # This should analyze successfully
        model.analyze_linear()

    def test_unsupported_dof_detected(self, basic_model):
        """Unsupported DOF with zero stiffness should be detected."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10x30')

        # No support at all on N1 - this will definitely be unstable
        # Only support N2 partially
        model.def_support('N2', False, True, False, False, False, False)

        model.add_node_load('N2', 'FY', -10, 'Case 1')
        model.add_load_combo('Combo 1', {'Case 1': 1.0})

        # This should raise an instability error
        with pytest.raises(Exception) as exc_info:
            model.analyze_linear(check_stability=True)

        # The error should mention instability
        error_msg = str(exc_info.value).lower()
        assert 'unstable' in error_msg or 'singular' in error_msg


# =============================================================================
# Tests for Improved Error Messages
# =============================================================================

class TestImprovedErrorMessages:
    """Tests for improved error messages in instability detection."""

    def test_floating_nodes_in_error_message(self, wood_model):
        """Error message should include names of floating nodes."""
        model = wood_model

        # Plates
        model.add_node('BP1', 0, 0, 0)
        model.add_node('BP2', 48, 0, 0)
        model.add_node('TP1', 0, 96, 0)
        model.add_node('TP2', 48, 96, 0)

        model.add_member('BPlate', 'BP1', 'BP2', 'Wood', '2x6')
        model.add_member('TPlate', 'TP1', 'TP2', 'Wood', '2x6')

        # Corner studs
        model.add_member('Stud1', 'BP1', 'TP1', 'Wood', '2x4')
        model.add_member('Stud2', 'BP2', 'TP2', 'Wood', '2x4')

        # Floating stud
        model.add_node('S_B', 24, 0.01, 0)
        model.add_node('S_T', 24, 95.99, 0)
        model.add_member('FloatingStud', 'S_B', 'S_T', 'Wood', '2x4')

        model.def_support('BP1', True, True, True, True, True, True)
        model.def_support('BP2', True, True, True, True, True, True)

        model.add_node_load('TP1', 'FY', -1000, 'Case 1')
        model.add_load_combo('Combo 1', {'Case 1': 1.0})

        with pytest.raises(Exception) as exc_info:
            model.analyze_linear()

        error_msg = str(exc_info.value)
        # Check that floating nodes are mentioned
        assert 'S_B' in error_msg or 'S_T' in error_msg or 'disconnected' in error_msg.lower()

    def test_error_message_has_guidance(self, basic_model):
        """Error message should include guidance for fixing the issue."""
        model = basic_model

        # Create a completely floating node with a member attached
        # This creates a floating substructure that won't be caught by
        # the diagonal stability check
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 60, 60, 0)  # Floating
        model.add_node('N4', 180, 60, 0)  # Also floating

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10x30')
        model.add_member('M2', 'N3', 'N4', 'Steel', 'W10x30')  # Floating member
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True, True, True, True)

        model.add_node_load('N3', 'FY', -10, 'Case 1')  # Load on floating node
        model.add_load_combo('Combo 1', {'Case 1': 1.0})

        # Run with check_stability=False to trigger our improved error path
        with pytest.raises(Exception) as exc_info:
            model.analyze_linear(check_stability=False)

        error_msg = str(exc_info.value).lower()
        # Should have some guidance about checking connections/supports
        assert 'connect' in error_msg or 'support' in error_msg or 'check' in error_msg


# =============================================================================
# Tests for Integration with Mesh-Generated Nodes
# =============================================================================

class TestMeshDiscretization:
    """Tests for discretization with mesh-generated nodes."""

    def test_member_through_mesh(self, basic_model):
        """Member passing through mesh should discretize at mesh nodes."""
        model = basic_model

        # Create a vertical member from (5,0,0) to (5,10,0)
        model.add_node('N1', 5, 0, 0)
        model.add_node('N2', 5, 10, 0)
        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10x30')

        # Create a horizontal mesh in XY plane
        model.add_rectangle_mesh('MESH1',
            mesh_size=2.5,
            width=10,
            height=10,
            thickness=0.1,
            material_name='Steel',
            origin=[0, 0, 0],
            plane='XY')

        model.def_support('N1', True, True, True, True, True, True)

        model.add_node_load('N2', 'FY', -10, 'Case 1')
        model.add_load_combo('Combo 1', {'Case 1': 1.0})

        # Prepare model to generate mesh and discretize
        Analysis._prepare_model(model)

        # Member should be discretized at mesh node locations
        # Mesh at y=0, 2.5, 5.0, 7.5, 10.0 along x=5
        assert len(model.members['M1'].sub_members) >= 2


# =============================================================================
# Tests for Load Distribution After Discretization
# =============================================================================

class TestLoadDistributionAfterDiscretization:
    """Tests that loads are properly distributed to sub-members after discretization."""

    def test_distributed_load_split_across_submembers(self, basic_model):
        """Distributed load should be split across sub-members."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 60, 0, 0)  # Midpoint

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10x30')
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', False, True, False, False, False, False)

        # Full-length distributed load
        model.add_member_dist_load('M1', 'Fy', -1.0, -1.0, 0, 120, 'Case 1')
        model.add_load_combo('Combo 1', {'Case 1': 1.0})

        Analysis._prepare_model(model)

        # Check that both sub-members have distributed loads
        for sub in model.members['M1'].sub_members.values():
            assert len(sub.DistLoads) > 0

    def test_point_load_on_correct_submember(self, basic_model):
        """Point load should end up on the correct sub-member."""
        model = basic_model
        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 120, 0, 0)
        model.add_node('N3', 60, 0, 0)  # Midpoint

        model.add_member('M1', 'N1', 'N2', 'Steel', 'W10x30')
        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', False, True, False, False, False, False)

        # Point load at x=30 (on first sub-member)
        model.add_member_pt_load('M1', 'Fy', -10, 30, 'Case 1')
        model.add_load_combo('Combo 1', {'Case 1': 1.0})

        Analysis._prepare_model(model)

        # First sub-member should have the point load
        first_sub = model.members['M1'].sub_members['M1a']
        assert len(first_sub.PtLoads) == 1


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
