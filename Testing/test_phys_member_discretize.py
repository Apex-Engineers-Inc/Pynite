import pytest
from Pynite.PhysMember import PhysMember
from Pynite.Node3D import Node3D
from Pynite.FEModel3D import FEModel3D

@pytest.fixture
def model_with_nodes():
    from Pynite.LoadCombo import LoadCombo
    from Pynite.Material import Material
    from Pynite.Section import Section
    model = FEModel3D()
    n1 = Node3D('N1', 0, 0, 0)
    n2 = Node3D('N2', 10, 0, 0)
    n3 = Node3D('N3', 5, 0, 0)
    model.nodes = {'N1': n1, 'N2': n2, 'N3': n3}
    # Add a real LoadCombo
    combo = LoadCombo('Combo 1')
    model.load_combos = {'Combo 1': combo}
    # Add a real Material and Section
    mat = Material(model, 'Steel', E=29000, G=11500, nu=0.3, rho=0.283, fy=50)
    sec = Section(model, 'W10x30', A=8.85, Iy=146, Iz=31.8, J=0.458)
    model.materials = {'Steel': mat}
    model.sections = {'W10x30': sec}
    return model, n1, n2, n3

def test_discretize_creates_submembers(model_with_nodes):
    model, n1, n2, n3 = model_with_nodes
    pm = PhysMember(model, 'PM1', n1, n2, 'Steel', 'W10x30')
    pm.DistLoads = []
    pm.PtLoads = []
    pm.discretize()
    assert len(pm.sub_members) == 2
    sub_names = sorted(pm.sub_members.keys())
    assert sub_names == ['PM1a', 'PM1b']
    sm1 = pm.sub_members['PM1a']
    sm2 = pm.sub_members['PM1b']
    assert sm1.i_node == n1
    assert sm1.j_node == n3
    assert sm2.i_node == n3
    assert sm2.j_node == n2

def test_discretize_distributes_point_loads(model_with_nodes):
    model, n1, n2, n3 = model_with_nodes
    pm = PhysMember(model, 'PM2', n1, n2, 'Steel', 'W10x30')
    pm.DistLoads = []
    # Place a point load at x=2 (should go to first sub-member)
    pm.PtLoads = [('Fy', 5.0, 2.0, 'Combo 1')]
    pm.discretize()
    sm1 = pm.sub_members['PM2a']
    sm2 = pm.sub_members['PM2b']
    assert len(sm1.PtLoads) == 1
    assert len(sm2.PtLoads) == 0
    # Place a point load at x=7 (should go to second sub-member)
    pm.PtLoads = [('Fy', 5.0, 7.0, 'Combo 1')]
    pm.discretize()
    sm1 = pm.sub_members['PM2a']
    sm2 = pm.sub_members['PM2b']
    assert len(sm1.PtLoads) == 0
    assert len(sm2.PtLoads) == 1

def test_discretize_distributes_dist_loads(model_with_nodes):
    model, n1, n2, n3 = model_with_nodes
    pm = PhysMember(model, 'PM3', n1, n2, 'Steel', 'W10x30')
    pm.DistLoads = [('Fy', 1.0, 2.0, 0.0, 10.0, 'Combo 1')]
    pm.PtLoads = []
    pm.discretize()
    sm1 = pm.sub_members['PM3a']
    sm2 = pm.sub_members['PM3b']
    assert len(sm1.DistLoads) == 1
    assert len(sm2.DistLoads) == 1
    l1 = sm1.DistLoads[0][4] - sm1.DistLoads[0][3]
    l2 = sm2.DistLoads[0][4] - sm2.DistLoads[0][3]
    assert pytest.approx(l1 + l2, 0.01) == 10.0

def test_discretize_no_internal_nodes():
    from Pynite.LoadCombo import LoadCombo
    from Pynite.Material import Material
    from Pynite.Section import Section
    model = FEModel3D()
    n1 = Node3D('N1', 0, 0, 0)
    n2 = Node3D('N2', 10, 0, 0)
    model.nodes = {'N1': n1, 'N2': n2}
    combo = LoadCombo('Combo 1')
    model.load_combos = {'Combo 1': combo}
    mat = Material(model, 'Steel', E=29000, G=11500, nu=0.3, rho=0.283, fy=50)
    sec = Section(model, 'W10x30', A=8.85, Iy=146, Iz=31.8, J=0.458)
    model.materials = {'Steel': mat}
    model.sections = {'W10x30': sec}
    pm = PhysMember(model, 'PM4', n1, n2, 'Steel', 'W10x30')
    pm.DistLoads = []
    pm.PtLoads = []
    pm.discretize()
    # Should only have one sub-member (no splits)
    assert len(pm.sub_members) == 1
    sub = next(iter(pm.sub_members.values()))
    assert sub.i_node == n1
    assert sub.j_node == n2

def test_discretize_multiple_internal_nodes():
    from Pynite.LoadCombo import LoadCombo
    from Pynite.Material import Material
    from Pynite.Section import Section
    model = FEModel3D()
    n1 = Node3D('N1', 0, 0, 0)
    n2 = Node3D('N2', 10, 0, 0)
    n3 = Node3D('N3', 3, 0, 0)
    n4 = Node3D('N4', 7, 0, 0)
    model.nodes = {'N1': n1, 'N2': n2, 'N3': n3, 'N4': n4}
    combo = LoadCombo('Combo 1')
    model.load_combos = {'Combo 1': combo}
    mat = Material(model, 'Steel', E=29000, G=11500, nu=0.3, rho=0.283, fy=50)
    sec = Section(model, 'W10x30', A=8.85, Iy=146, Iz=31.8, J=0.458)
    model.materials = {'Steel': mat}
    model.sections = {'W10x30': sec}
    pm = PhysMember(model, 'PM5', n1, n2, 'Steel', 'W10x30')
    pm.DistLoads = []
    pm.PtLoads = []
    pm.discretize()
    # Should split into 3 sub-members: N1-N3, N3-N4, N4-N2
    assert len(pm.sub_members) == 3
    names = sorted(pm.sub_members.keys())
    assert names == ['PM5a', 'PM5b', 'PM5c']
    sm_a = pm.sub_members['PM5a']
    sm_b = pm.sub_members['PM5b']
    sm_c = pm.sub_members['PM5c']
    assert sm_a.i_node == n1 and sm_a.j_node == n3
    assert sm_b.i_node == n3 and sm_b.j_node == n4
    assert sm_c.i_node == n4 and sm_c.j_node == n2

def test_discretize_ignores_off_member_nodes():
    from Pynite.LoadCombo import LoadCombo
    from Pynite.Material import Material
    from Pynite.Section import Section
    model = FEModel3D()
    n1 = Node3D('N1', 0, 0, 0)
    n2 = Node3D('N2', 10, 0, 0)
    n3 = Node3D('N3', 5, 1, 0)  # Not colinear
    model.nodes = {'N1': n1, 'N2': n2, 'N3': n3}
    combo = LoadCombo('Combo 1')
    model.load_combos = {'Combo 1': combo}
    mat = Material(model, 'Steel', E=29000, G=11500, nu=0.3, rho=0.283, fy=50)
    sec = Section(model, 'W10x30', A=8.85, Iy=146, Iz=31.8, J=0.458)
    model.materials = {'Steel': mat}
    model.sections = {'W10x30': sec}
    pm = PhysMember(model, 'PM6', n1, n2, 'Steel', 'W10x30')
    pm.DistLoads = []
    pm.PtLoads = []
    pm.discretize()
    # Should not split, only one sub-member
    assert len(pm.sub_members) == 1
    sub = next(iter(pm.sub_members.values()))
    assert sub.i_node == n1
    assert sub.j_node == n2

def test_discretize_node_beyond_endpoints():
    """Test that nodes beyond the member endpoints are ignored"""
    from Pynite.LoadCombo import LoadCombo
    from Pynite.Material import Material
    from Pynite.Section import Section
    model = FEModel3D()
    n1 = Node3D('N1', 0, 0, 0)
    n2 = Node3D('N2', 10, 0, 0)
    n3 = Node3D('N3', -5, 0, 0)  # Before start
    n4 = Node3D('N4', 15, 0, 0)  # After end
    model.nodes = {'N1': n1, 'N2': n2, 'N3': n3, 'N4': n4}
    combo = LoadCombo('Combo 1')
    model.load_combos = {'Combo 1': combo}
    mat = Material(model, 'Steel', E=29000, G=11500, nu=0.3, rho=0.283, fy=50)
    sec = Section(model, 'W10x30', A=8.85, Iy=146, Iz=31.8, J=0.458)
    model.materials = {'Steel': mat}
    model.sections = {'W10x30': sec}
    pm = PhysMember(model, 'PM7', n1, n2, 'Steel', 'W10x30')
    pm.DistLoads = []
    pm.PtLoads = []
    pm.discretize()
    # Should not split despite colinear nodes beyond endpoints
    assert len(pm.sub_members) == 1
    sub = next(iter(pm.sub_members.values()))
    assert sub.i_node == n1
    assert sub.j_node == n2

def test_discretize_point_load_at_boundaries(model_with_nodes):
    """Test point loads exactly at node boundaries"""
    model, n1, n2, n3 = model_with_nodes
    pm = PhysMember(model, 'PM8', n1, n2, 'Steel', 'W10x30')
    pm.DistLoads = []
    # Point load exactly at internal node position (x=5)
    pm.PtLoads = [('Fy', 10.0, 5.0, 'Combo 1')]
    pm.discretize()
    sm1 = pm.sub_members['PM8a']
    sm2 = pm.sub_members['PM8b']
    # Load at boundary should go to second sub-member
    assert len(sm1.PtLoads) == 0
    assert len(sm2.PtLoads) == 1
    assert sm2.PtLoads[0][2] == 0.0  # Adjusted to start of second segment

def test_discretize_point_load_at_end(model_with_nodes):
    """Test point load at the very end of the member"""
    model, n1, n2, n3 = model_with_nodes
    pm = PhysMember(model, 'PM9', n1, n2, 'Steel', 'W10x30')
    pm.DistLoads = []
    # Point load at the end (x=10)
    pm.PtLoads = [('Fy', 10.0, 10.0, 'Combo 1')]
    pm.discretize()
    sm1 = pm.sub_members['PM9a']
    sm2 = pm.sub_members['PM9b']
    # Load at end should go to last sub-member
    assert len(sm1.PtLoads) == 0
    assert len(sm2.PtLoads) == 1
    assert sm2.PtLoads[0][2] == 5.0  # Adjusted to end of second segment

def test_discretize_distributed_load_partial_overlap():
    """Test distributed load that only partially overlaps sub-members"""
    from Pynite.LoadCombo import LoadCombo
    from Pynite.Material import Material
    from Pynite.Section import Section
    model = FEModel3D()
    n1 = Node3D('N1', 0, 0, 0)
    n2 = Node3D('N2', 12, 0, 0)
    n3 = Node3D('N3', 4, 0, 0)
    n4 = Node3D('N4', 8, 0, 0)
    model.nodes = {'N1': n1, 'N2': n2, 'N3': n3, 'N4': n4}
    combo = LoadCombo('Combo 1')
    model.load_combos = {'Combo 1': combo}
    mat = Material(model, 'Steel', E=29000, G=11500, nu=0.3, rho=0.283, fy=50)
    sec = Section(model, 'W10x30', A=8.85, Iy=146, Iz=31.8, J=0.458)
    model.materials = {'Steel': mat}
    model.sections = {'W10x30': sec}
    pm = PhysMember(model, 'PM10', n1, n2, 'Steel', 'W10x30')
    # Distributed load from x=2 to x=10 (spans across multiple segments)
    pm.DistLoads = [('Fy', 1.0, 3.0, 2.0, 10.0, 'Combo 1')]
    pm.PtLoads = []
    pm.discretize()
    # Should have 3 sub-members
    assert len(pm.sub_members) == 3
    sm_a = pm.sub_members['PM10a']  # 0-4
    sm_b = pm.sub_members['PM10b']  # 4-8
    sm_c = pm.sub_members['PM10c']  # 8-12
    # First segment should have partial load (x=2 to x=4)
    assert len(sm_a.DistLoads) == 1
    # Second segment should have full load (x=0 to x=4 in local coords)
    assert len(sm_b.DistLoads) == 1
    # Third segment should have partial load (x=0 to x=2 in local coords)
    assert len(sm_c.DistLoads) == 1

def test_discretize_3d_member():
    """Test discretize with a 3D member (not aligned with axes)"""
    from Pynite.LoadCombo import LoadCombo
    from Pynite.Material import Material
    from Pynite.Section import Section
    model = FEModel3D()
    n1 = Node3D('N1', 0, 0, 0)
    n2 = Node3D('N2', 3, 4, 5)  # 3D member
    n3 = Node3D('N3', 1.5, 2, 2.5)  # Midpoint
    model.nodes = {'N1': n1, 'N2': n2, 'N3': n3}
    combo = LoadCombo('Combo 1')
    model.load_combos = {'Combo 1': combo}
    mat = Material(model, 'Steel', E=29000, G=11500, nu=0.3, rho=0.283, fy=50)
    sec = Section(model, 'W10x30', A=8.85, Iy=146, Iz=31.8, J=0.458)
    model.materials = {'Steel': mat}
    model.sections = {'W10x30': sec}
    pm = PhysMember(model, 'PM11', n1, n2, 'Steel', 'W10x30')
    pm.DistLoads = []
    pm.PtLoads = []
    pm.discretize()
    # Should split at the midpoint
    assert len(pm.sub_members) == 2
    sm1 = pm.sub_members['PM11a']
    sm2 = pm.sub_members['PM11b']
    assert sm1.i_node == n1 and sm1.j_node == n3
    assert sm2.i_node == n3 and sm2.j_node == n2

def test_discretize_with_end_releases():
    """Test that end releases are properly applied to sub-members"""
    from Pynite.LoadCombo import LoadCombo
    from Pynite.Material import Material
    from Pynite.Section import Section
    model = FEModel3D()
    n1 = Node3D('N1', 0, 0, 0)
    n2 = Node3D('N2', 10, 0, 0)
    n3 = Node3D('N3', 5, 0, 0)
    model.nodes = {'N1': n1, 'N2': n2, 'N3': n3}
    combo = LoadCombo('Combo 1')
    model.load_combos = {'Combo 1': combo}
    mat = Material(model, 'Steel', E=29000, G=11500, nu=0.3, rho=0.283, fy=50)
    sec = Section(model, 'W10x30', A=8.85, Iy=146, Iz=31.8, J=0.458)
    model.materials = {'Steel': mat}
    model.sections = {'W10x30': sec}
    pm = PhysMember(model, 'PM12', n1, n2, 'Steel', 'W10x30')
    # Set some end releases
    pm.Releases = [True, True, False, False, False, False, False, False, True, True, False, False]  # Release i-end Fx,Fy and j-end My,Mz
    pm.DistLoads = []
    pm.PtLoads = []
    pm.discretize()
    sm1 = pm.sub_members['PM12a']
    sm2 = pm.sub_members['PM12b']
    # First sub-member should have i-end releases
    assert sm1.Releases[0] == True  # i-end Fx
    assert sm1.Releases[1] == True  # i-end Fy
    assert sm1.Releases[8] == False  # j-end My (no release at internal node)
    # Second sub-member should have j-end releases
    assert sm2.Releases[0] == False  # i-end Fx (no release at internal node)
    assert sm2.Releases[8] == True  # j-end My
    assert sm2.Releases[9] == True  # j-end Mz

def test_discretize_multiple_calls():
    """Test that calling discretize multiple times clears previous sub-members"""
    model = FEModel3D()
    from Pynite.LoadCombo import LoadCombo
    from Pynite.Material import Material
    from Pynite.Section import Section
    n1 = Node3D('N1', 0, 0, 0)
    n2 = Node3D('N2', 10, 0, 0)
    n3 = Node3D('N3', 5, 0, 0)
    model.nodes = {'N1': n1, 'N2': n2, 'N3': n3}
    combo = LoadCombo('Combo 1')
    model.load_combos = {'Combo 1': combo}
    mat = Material(model, 'Steel', E=29000, G=11500, nu=0.3, rho=0.283, fy=50)
    sec = Section(model, 'W10x30', A=8.85, Iy=146, Iz=31.8, J=0.458)
    model.materials = {'Steel': mat}
    model.sections = {'W10x30': sec}
    pm = PhysMember(model, 'PM13', n1, n2, 'Steel', 'W10x30')
    pm.DistLoads = []
    pm.PtLoads = []
    # First call
    pm.discretize()
    assert len(pm.sub_members) == 2
    first_keys = set(pm.sub_members.keys())
    # Second call should clear and recreate
    pm.discretize()
    assert len(pm.sub_members) == 2
    second_keys = set(pm.sub_members.keys())
    assert first_keys == second_keys  # Same keys, but new objects
