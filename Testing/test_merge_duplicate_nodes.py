import pytest
from typing import Generator
from Pynite.FEModel3D import FEModel3D
from Pynite.Node3D import Node3D
from Pynite.Material import Material
from Pynite.Section import Section
from Pynite.LoadCombo import LoadCombo


@pytest.fixture
def basic_model() -> Generator[FEModel3D, None, None]:
    """Create a basic FE model with materials and sections for testing."""
    model = FEModel3D()
    
    # Add material and section
    mat = Material(model, 'Steel', E=29000, G=11500, nu=0.3, rho=0.283, fy=50)
    sec = Section(model, 'W10x30', A=8.85, Iy=146, Iz=31.8, J=0.458)
    model.materials = {'Steel': mat}
    model.sections = {'W10x30': sec}
    
    # Add load combination
    combo = LoadCombo('Combo 1')
    model.load_combos = {'Combo 1': combo}
    
    yield model


class TestBasicNodeMerging:
    """Test basic node merging functionality and tolerance handling."""

    def test_no_duplicate_nodes(self, basic_model: FEModel3D) -> None:
        """Test that method returns empty list when no duplicates exist."""
        # Add nodes that are not duplicates
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 10, 0, 0)
        basic_model.add_node('N3', 0, 10, 0)
        
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        assert removed_nodes == []
        assert len(basic_model.nodes) == 3
        assert 'N1' in basic_model.nodes
        assert 'N2' in basic_model.nodes
        assert 'N3' in basic_model.nodes

    def test_exact_duplicate_nodes(self, basic_model: FEModel3D) -> None:
        """Test merging nodes at exactly the same location."""
        # Add identical nodes
        basic_model.add_node('N1', 5, 5, 5)
        basic_model.add_node('N2', 5, 5, 5)  # Exact duplicate
        basic_model.add_node('N3', 10, 10, 10)
        
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        assert len(removed_nodes) == 1
        assert 'N2' in removed_nodes
        assert len(basic_model.nodes) == 2
        assert 'N1' in basic_model.nodes
        assert 'N3' in basic_model.nodes
        assert 'N2' not in basic_model.nodes

    def test_tolerance_based_merging(self, basic_model: FEModel3D) -> None:
        """Test that nodes within tolerance are merged."""
        # Add nodes within tolerance
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0.0005, 0)  # Within 0.001 tolerance
        basic_model.add_node('N3', 0.002, 0, 0)        # Outside tolerance
        
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        assert len(removed_nodes) == 1
        assert 'N2' in removed_nodes
        assert len(basic_model.nodes) == 2
        assert 'N1' in basic_model.nodes
        assert 'N3' in basic_model.nodes

    def test_multiple_duplicates(self, basic_model: FEModel3D) -> None:
        """Test merging multiple sets of duplicate nodes."""
        # Add multiple sets of duplicates
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)      # Duplicate of N1
        basic_model.add_node('N3', 10, 10, 10)
        basic_model.add_node('N4', 10.0005, 10, 10)   # Duplicate of N3
        basic_model.add_node('N5', 20, 20, 20)        # No duplicate
        
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        assert len(removed_nodes) == 2
        assert 'N2' in removed_nodes
        assert 'N4' in removed_nodes
        assert len(basic_model.nodes) == 3
        assert 'N1' in basic_model.nodes
        assert 'N3' in basic_model.nodes
        assert 'N5' in basic_model.nodes

    def test_custom_tolerance(self, basic_model: FEModel3D) -> None:
        """Test that custom tolerance values work correctly."""
        # Add nodes with specific distances
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.005, 0, 0)  # 0.005 units away
        basic_model.add_node('N3', 0.015, 0, 0)  # 0.015 units away
        
        # Test with tolerance of 0.01 - should merge N1 and N2 but not N3
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.01)
        
        assert len(removed_nodes) == 1
        assert 'N2' in removed_nodes
        assert len(basic_model.nodes) == 2
        assert 'N1' in basic_model.nodes
        assert 'N3' in basic_model.nodes

    def test_3d_distance_calculation(self, basic_model: FEModel3D) -> None:
        """Test that 3D distance calculation works correctly in all dimensions."""
        # Add nodes that are close in 3D space
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0003, 0.0004, 0)       # Distance = 0.0005, within tolerance
        basic_model.add_node('N3', 0.0006, 0.0008, 0)       # Distance = 0.001, exactly at tolerance
        basic_model.add_node('N4', 0.0006, 0.0008, 0.0001)  # Distance > tolerance due to Z component
        
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # N2 should merge with N1, N3 should merge with N1, but N4 should not
        assert len(removed_nodes) == 2
        assert 'N2' in removed_nodes
        assert 'N3' in removed_nodes
        assert 'N4' not in removed_nodes
        assert len(basic_model.nodes) == 2

    def test_negative_coordinates(self, basic_model: FEModel3D) -> None:
        """Test merging nodes with negative coordinates."""
        # Add nodes with negative coordinates
        basic_model.add_node('N1', -10, -5, -2)
        basic_model.add_node('N2', -10.0005, -5, -2)  # Duplicate with slight offset
        basic_model.add_node('N3', 10, 5, 2)          # Positive coordinates, far away
        
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        assert len(removed_nodes) == 1
        assert 'N2' in removed_nodes
        assert len(basic_model.nodes) == 2
        
        # Verify coordinates are preserved correctly
        n1 = basic_model.nodes['N1']
        assert n1.X == -10
        assert n1.Y == -5
        assert n1.Z == -2


class TestElementReferenceUpdates:
    """Test that element node references are properly updated after merging."""

    def test_member_references_updated(self, basic_model: FEModel3D) -> None:
        """Test that member node references are properly updated after merging."""
        # Add nodes
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)  # Duplicate of N1
        basic_model.add_node('N3', 10, 0, 0)
        
        # Add member using the duplicate node
        basic_model.add_member('M1', 'N2', 'N3', 'Steel', 'W10x30')
        
        # Verify member initially references N2
        assert basic_model.members['M1'].i_node.name == 'N2'
        assert basic_model.members['M1'].j_node.name == 'N3'
        
        # Merge duplicates
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Verify member now references N1 instead of N2
        assert 'N2' in removed_nodes
        assert basic_model.members['M1'].i_node.name == 'N1'
        assert basic_model.members['M1'].j_node.name == 'N3'
        assert basic_model.members['M1'].i_node is basic_model.nodes['N1']

    def test_spring_references_updated(self, basic_model: FEModel3D) -> None:
        """Test that spring node references are properly updated after merging."""
        # Add nodes
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)  # Duplicate of N1
        basic_model.add_node('N3', 10, 0, 0)
        
        # Add spring using the duplicate node
        basic_model.add_spring('S1', 'N2', 'N3', 1000)
        
        # Verify spring initially references N2
        assert basic_model.springs['S1'].i_node.name == 'N2'
        
        # Merge duplicates
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Verify spring now references N1
        assert 'N2' in removed_nodes
        assert basic_model.springs['S1'].i_node.name == 'N1'
        assert basic_model.springs['S1'].i_node is basic_model.nodes['N1']

    def test_quad_references_updated(self, basic_model: FEModel3D) -> None:
        """Test that quad element node references are properly updated after merging."""
        # Add nodes for a quad
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)  # Duplicate of N1
        basic_model.add_node('N3', 1, 0, 0)
        basic_model.add_node('N4', 1, 1, 0)
        basic_model.add_node('N5', 0, 1, 0)
        
        # Add quad using the duplicate node
        basic_model.add_quad('Q1', 'N2', 'N3', 'N4', 'N5', 0.1, 'Steel')
        
        # Verify quad initially references N2
        assert basic_model.quads['Q1'].i_node.name == 'N2'
        
        # Merge duplicates
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Verify quad now references N1
        assert 'N2' in removed_nodes
        assert basic_model.quads['Q1'].i_node.name == 'N1'
        assert basic_model.quads['Q1'].i_node is basic_model.nodes['N1']

    def test_plate_element_references_updated(self, basic_model: FEModel3D) -> None:
        """Test that plate element node references are properly updated after merging."""
        # Add nodes for a plate
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)  # Duplicate of N1
        basic_model.add_node('N3', 1, 0, 0)
        basic_model.add_node('N4', 1, 1, 0)
        basic_model.add_node('N5', 0, 1, 0)
        
        # Add plate using the duplicate node
        basic_model.add_plate('P1', 'N2', 'N3', 'N4', 'N5', 0.1, 'Steel')
        
        # Verify plate initially references N2
        assert basic_model.plates['P1'].i_node.name == 'N2'
        
        # Merge duplicates
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Verify plate now references N1
        assert 'N2' in removed_nodes
        assert basic_model.plates['P1'].i_node.name == 'N1'
        assert basic_model.plates['P1'].i_node is basic_model.nodes['N1']

    def test_element_creation_order_independence(self, basic_model: FEModel3D) -> None:
        """Test that element creation order doesn't affect merging results."""
        # Add nodes
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)  # Duplicate
        basic_model.add_node('N3', 10, 0, 0)
        
        # Add elements in different order
        basic_model.add_member('M1', 'N1', 'N3', 'Steel', 'W10x30')
        basic_model.add_spring('S1', 'N2', 'N3', 1000)
        basic_model.add_member('M2', 'N2', 'N3', 'Steel', 'W10x30')
        
        # Merge duplicates
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Verify all elements now reference N1
        assert 'N2' in removed_nodes
        assert basic_model.members['M1'].i_node.name == 'N1'
        assert basic_model.members['M2'].i_node.name == 'N1'
        assert basic_model.springs['S1'].i_node.name == 'N1'


class TestBoundaryConditionMerging:
    """Test merging of boundary conditions from duplicate nodes."""

    def test_support_conditions_merged(self, basic_model: FEModel3D) -> None:
        """Test that support conditions are properly merged from duplicate nodes."""
        # Add duplicate nodes
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)  # Duplicate of N1
        
        # Apply different support conditions to each node
        basic_model.def_support('N1', support_DX=True, support_DY=False)
        basic_model.def_support('N2', support_DX=False, support_DZ=True, support_RX=True)
        
        # Merge duplicates
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Verify support conditions are merged (should be OR of both)
        assert 'N2' in removed_nodes
        n1 = basic_model.nodes['N1']
        assert n1.support_DX == True   # From N1
        assert n1.support_DY == False  # From N1 (N2 didn't override)
        assert n1.support_DZ == True   # From N2
        assert n1.support_RX == True   # From N2

    def test_spring_supports_merged(self, basic_model: FEModel3D) -> None:
        """Test that spring support conditions are properly merged from duplicate nodes."""
        # Add duplicate nodes
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)  # Duplicate of N1
        
        # Apply spring supports to N2
        basic_model.def_support_spring('N2', 'DX', 1000.0)
        
        # Merge duplicates
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Verify spring support is transferred to N1
        assert 'N2' in removed_nodes
        n1 = basic_model.nodes['N1']
        assert n1.spring_DX != [None, None, None]  # Should have spring support

    def test_boundary_conditions_complex_merge(self, basic_model: FEModel3D) -> None:
        """Test complex boundary condition merging scenarios."""
        # Add duplicate nodes
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)
        
        # Apply overlapping and non-overlapping support conditions
        basic_model.def_support('N1', support_DX=True, support_RY=True)
        basic_model.def_support('N2', support_DX=False, support_DY=True, support_RZ=True)
        
        # Apply spring support to N2
        basic_model.def_support_spring('N2', 'DZ', 5000.0)
        
        # Merge duplicates
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Verify all conditions are properly merged
        assert 'N2' in removed_nodes
        n1 = basic_model.nodes['N1']
        assert n1.support_DX == True   # Should keep True from N1
        assert n1.support_DY == True   # Should get True from N2
        assert n1.support_RY == True   # Should keep True from N1
        assert n1.support_RZ == True   # Should get True from N2
        assert n1.spring_DZ != [None, None, None]  # Should get spring from N2


class TestMeshIntegration:
    """Test mesh integration and node reference updates."""

    def test_mesh_node_references_updated(self, basic_model: FEModel3D) -> None:
        """Test that mesh node references are properly updated after merging."""
        # Add nodes
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)  # Duplicate of N1
        
        # Create a simple mesh and manually add the duplicate node
        basic_model.add_rectangle_mesh('Mesh1', 1.0, 2.0, 2.0, 0.1, 'Steel')
        mesh = basic_model.meshes['Mesh1']
        
        # Manually add N2 to the mesh to test the mesh fixing logic
        mesh.nodes['N2'] = basic_model.nodes['N2']
        
        # Merge duplicates
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Verify mesh nodes are updated
        assert 'N2' in removed_nodes
        if 'N2' in mesh.nodes:
            # If N2 was in mesh, it should be replaced with N1
            assert mesh.nodes['N2'] is basic_model.nodes['N1']
            assert 'N1' in mesh.nodes


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness scenarios."""

    def test_chain_of_duplicates(self, basic_model: FEModel3D) -> None:
        """Test handling of chains of duplicate nodes (A≈B≈C)."""
        # Add a chain of nodes within tolerance of each other
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)      # Within tolerance of N1
        basic_model.add_node('N3', 0.0008, 0, 0)      # Within tolerance of N2, but processed after N1-N2 merge
        basic_model.add_node('N4', 10, 0, 0)          # Far away
        
        # Add member connecting to middle node in chain
        basic_model.add_member('M1', 'N2', 'N4', 'Steel', 'W10x30')
        
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Should merge N2 into N1, then N3 into N1
        assert len(removed_nodes) == 2
        assert 'N2' in removed_nodes
        assert 'N3' in removed_nodes
        assert len(basic_model.nodes) == 2
        assert 'N1' in basic_model.nodes
        assert 'N4' in basic_model.nodes
        
        # Member should now reference N1
        assert basic_model.members['M1'].i_node.name == 'N1'

    def test_empty_model(self, basic_model: FEModel3D) -> None:
        """Test merge_duplicate_nodes on empty model."""
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        assert removed_nodes == []
        assert len(basic_model.nodes) == 0

    def test_single_node(self, basic_model: FEModel3D) -> None:
        """Test merge_duplicate_nodes with only one node."""
        basic_model.add_node('N1', 0, 0, 0)
        
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        assert removed_nodes == []
        assert len(basic_model.nodes) == 1
        assert 'N1' in basic_model.nodes

    def test_large_tolerance(self, basic_model: FEModel3D) -> None:
        """Test with very large tolerance that would merge all nodes."""
        # Add nodes far apart
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 100, 100, 100)
        basic_model.add_node('N3', 200, 200, 200)
        
        # Use tolerance larger than distances
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=1000.0)
        
        # Should merge all into first node
        assert len(removed_nodes) == 2
        assert 'N2' in removed_nodes
        assert 'N3' in removed_nodes
        assert len(basic_model.nodes) == 1
        assert 'N1' in basic_model.nodes

    def test_zero_tolerance(self, basic_model: FEModel3D) -> None:
        """Test with zero tolerance - only exact matches should merge."""
        # Add nodes, some exact duplicates, some very close
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0, 0, 0)          # Exact duplicate
        basic_model.add_node('N3', 1e-10, 0, 0)      # Very close but not exact
        
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.0)
        
        # Should only merge exact duplicates
        assert len(removed_nodes) == 1
        assert 'N2' in removed_nodes
        assert len(basic_model.nodes) == 2
        assert 'N1' in basic_model.nodes
        assert 'N3' in basic_model.nodes

    def test_solution_flagged_as_unsolved(self, basic_model: FEModel3D) -> None:
        """Test that model solution is flagged as None after merging nodes."""
        # Add duplicate nodes
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)
        
        # Set a fake solution to test it gets cleared
        basic_model.solution = "fake_solution"
        
        # Merge duplicates
        basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Verify solution is cleared
        assert basic_model.solution is None

    def test_return_value_format(self, basic_model: FEModel3D) -> None:
        """Test that return value is properly formatted as a list of strings."""
        # Add duplicate nodes
        basic_model.add_node('N1', 0, 0, 0)
        basic_model.add_node('N2', 0.0005, 0, 0)
        basic_model.add_node('N3', 5, 0, 0)
        basic_model.add_node('N4', 5.0005, 0, 0)
        
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Verify return value format
        assert isinstance(removed_nodes, list)
        assert len(removed_nodes) == 2
        for node_name in removed_nodes:
            assert isinstance(node_name, str)
        assert 'N2' in removed_nodes
        assert 'N4' in removed_nodes


class TestPerformanceAndScale:
    """Test performance and scalability scenarios."""

    def test_performance_with_many_nodes(self, basic_model: FEModel3D) -> None:
        """Test performance with a moderate number of nodes."""
        # Add many nodes in a grid pattern with some duplicates
        for i in range(20):
            for j in range(20):
                basic_model.add_node(f'N_{i}_{j}', i * 1.0, j * 1.0, 0)
        
        # Add some duplicate nodes
        basic_model.add_node('DUP1', 5.0001, 5.0, 0)  # Near N_5_5
        basic_model.add_node('DUP2', 10.0001, 10.0, 0)  # Near N_10_10
        basic_model.add_node('DUP3', 15.0001, 15.0, 0)  # Near N_15_15
        
        initial_count = len(basic_model.nodes)
        removed_nodes = basic_model.merge_duplicate_nodes(tolerance=0.001)
        
        # Should merge the 3 duplicate nodes
        assert len(removed_nodes) == 3
        assert len(basic_model.nodes) == initial_count - 3
        assert 'DUP1' in removed_nodes
        assert 'DUP2' in removed_nodes
        assert 'DUP3' in removed_nodes
