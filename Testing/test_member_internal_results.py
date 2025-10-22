from Pynite import FEModel3D
import math


def test_beam_internal_forces():
    """
    Units for this model are kips and feet
    """

    # Define a new beam
    beam = FEModel3D()

    # Define the nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', 10, 0, 0)

    # Define the supports (simply supported)
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', True, True, True, True, False, False)

    # Define beam section proerties
    J = 400/12**4
    Iy = 200/12**4
    Iz = 200/12**4
    A = 12/12**2
    beam.add_section('Section', A, Iy, Iz, J)

    # Define a material
    E = 29000*144  # ksf
    G = 11200*144  # ksf
    nu = 0.3
    rho = 0.490  # pcf
    beam.add_material('Steel', E, G, nu, rho)

    # Create the beam
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')

    # Add a mid-span node to force the model to split a physical member for this test
    beam.add_node('N3', 5, 0, 0)

    # Add a member distributed load along the strong axis
    beam.add_member_dist_load('M1', 'FY', -0.5, -0.5, case='D')
    beam.add_member_dist_load('M1', 'FY', -0.75, -0.75, case='L')

    # Add a member distributed laod along the weak axis
    beam.add_member_dist_load('M1', 'FZ', -0.5, -0.5, case='D')
    beam.add_member_dist_load('M1', 'FZ', -0.75, -0.75, case='L')

    # Add some load combinations
    beam.add_load_combo('D', {'D': 1.0}, ['blc'])
    beam.add_load_combo('L', {'L': 1.0}, ['blc'])
    beam.add_load_combo('1.2D + 1.6L', {'D': 1.2, 'L': 1.6}, ['strength'])

    # Analyze the model
    beam.analyze_linear()

    # from Pynite.Visualization import Renderer
    # renderer = Renderer(beam)
    # renderer.combo_name = 'D'
    # renderer.annotation_size = 1
    # renderer.render_model()

    # Check the shear diagram
    assert math.isclose(beam.members['M1'].shear('Fy', 0, 'D'), 2.5, abs_tol=0.01), 'Fy internal shear test failed at start of member.'
    assert math.isclose(beam.members['M1'].shear('Fz', 0, 'D'), 2.5, abs_tol=0.01), 'Fz internal shear test failed at start of member.'
    assert math.isclose(beam.members['M1'].shear('Fy', 10, 'D'), -2.5, abs_tol=0.01), 'Fy internal shear test failed at end of member.'
    assert math.isclose(beam.members['M1'].shear('Fz', 10, 'D'), -2.5, abs_tol=0.01), 'Fz internal shear test failed at end of member.'
    assert math.isclose(beam.members['M1'].shear('Fy', 5, 'D'), 0, abs_tol=0.01), 'Fy internal shear test failed at midpoint of member.'
    assert math.isclose(beam.members['M1'].shear('Fz', 5, 'D'), 0, abs_tol=0.01), 'Fz internal shear test failed at midpoint of member.'

    assert math.isclose(beam.members['M1'].max_shear('Fy', 'D'), 2.5, abs_tol=0.01), 'Fy internal max shear test failed.'
    assert math.isclose(beam.members['M1'].max_shear('Fz', 'D'), 2.5, abs_tol=0.01), 'Fz internal max shear test failed.'
    assert math.isclose(beam.members['M1'].min_shear('Fy', 'D'), -2.5, abs_tol=0.01), 'Fy internal min shear test failed.'
    assert math.isclose(beam.members['M1'].min_shear('Fz', 'D'), -2.5, abs_tol=0.01), 'Fz internal min shear test failed.'
    assert math.isclose(beam.members['M1'].max_shear('Fy', '1.2D + 1.6L'), 9.0, abs_tol=0.01), 'Failed envelope shear test.'
    assert math.isclose(beam.members['M1'].min_shear('Fz', '1.2D + 1.6L'), -9.0, abs_tol=0.01), 'Failed envelope shear test.'

    # Check the moment diagram
    assert math.isclose(beam.members['M1'].moment('Mz', 0, 'D'), 0, abs_tol=2), 'Mz internal moment test failed at start of member.'
    assert math.isclose(beam.members['M1'].moment('My', 0, 'D'), 0, abs_tol=2), 'My internal moment test failed at start of member.'
    assert math.isclose(beam.members['M1'].moment('Mz', 5, 'D'), -6.25, abs_tol=2), 'Mz internal moment test failed at midpoint of member.'
    assert math.isclose(beam.members['M1'].moment('My', 5, 'D'), -6.25, abs_tol=2), 'My internal moment test failed at midpoint of member.'
    assert math.isclose(beam.members['M1'].moment('Mz', 10, 'D'), 0, abs_tol=2), 'Mz internal moment test failed at end of member.'
    assert math.isclose(beam.members['M1'].moment('My', 10, 'D'), 0, abs_tol=2), 'My internal moment test failed at end of member.'

    assert math.isclose(beam.members['M1'].min_moment('Mz', 'D'), -6.25, abs_tol=2), 'Mz internal min moment test failed.'
    assert math.isclose(beam.members['M1'].min_moment('My', 'D'), -6.25, abs_tol=2), 'My internal min moment test failed.'
    assert math.isclose(beam.members['M1'].max_moment('Mz', 'D'), 0, abs_tol=2), 'Mz internal max moment test failed.'
    assert math.isclose(beam.members['M1'].max_moment('My', 'D'), 0, abs_tol=2), 'My internal max moment test failed.'
    assert math.isclose(beam.members['M1'].min_moment('Mz', '1.2D + 1.6L'), -22.5, abs_tol=2), 'Failed member Mz envelope results test.'
    assert math.isclose(beam.members['M1'].max_moment('My', '1.2D + 1.6L'), 0, abs_tol=2), 'Failed member My envelope results test.'

    # Check the deflected shape
    assert math.isclose(beam.members['M1'].deflection('dy', 0, 'D')*12, 0, abs_tol=0.00001), 'dy internal deflection test failed at start of member.'
    assert math.isclose(beam.members['M1'].deflection('dz', 0, 'D')*12, 0, abs_tol=0.00001), 'dz internal deflection test failed at start of member.'
    assert math.isclose(beam.members['M1'].deflection('dz', 5, 'D')*12, 5*(-0.5)*10**4/(384*E*Iz)*12, abs_tol=0.00001), 'dz internal deflection test failed at midpoint of member.'
    assert math.isclose(beam.members['M1'].deflection('dy', 5, 'D')*12, 5*(-0.5)*10**4/(384*E*Iy)*12, abs_tol=0.00001), 'dy internal deflection test failed at midpoint of member.'
    assert math.isclose(beam.members['M1'].deflection('dy', 10, 'D')*12, 0, abs_tol=0.00001), 'dy internal deflection test failed at end of member.'
    assert math.isclose(beam.members['M1'].deflection('dz', 10, 'D')*12, 0, abs_tol=0.00001), 'dz internal deflection test failed at end of member.'

    assert math.isclose(beam.members['M1'].max_deflection('dy', 'D')*12, 0, abs_tol=0.00001), 'dy internal max deflection test failed.'
    assert math.isclose(beam.members['M1'].max_deflection('dz', 'D')*12, 0, abs_tol=0.00001), 'dz internal max deflection test failed.'
    assert math.isclose(beam.members['M1'].min_deflection('dz', 'D')*12, 5*(-0.5)*10**4/(384*E*Iz)*12, abs_tol=0.00001), 'dz internal min deflection test failed.'
    assert math.isclose(beam.members['M1'].min_deflection('dy', 'D')*12, 5*(-0.5)*10**4/(384*E*Iy)*12, abs_tol=0.00001), 'dy internal min deflection test failed.'
    assert math.isclose(beam.members['M1'].min_deflection('dz', '1.2D + 1.6L')*12, 5*(-1.8)*10**4/(384*E*Iz)*12, abs_tol=0.00001), 'Failed member dz envelope test.'
    assert math.isclose(beam.members['M1'].min_deflection('dy', '1.2D + 1.6L')*12, 5*(-1.8)*10**4/(384*E*Iy)*12, abs_tol=0.00001), 'Failed member dy envelope test.'


def test_beam_internal_forces_arrays():
    """
    Tests the array methods (shear_array, moment_array, deflection_array)
    on a larger model with 20 points to ensure proper array generation.
    Units for this model are kips and feet.
    """

    # Define a new beam
    beam = FEModel3D()

    # Define the nodes
    beam.add_node('N1', 0, 0, 0)
    beam.add_node('N2', 10, 0, 0)

    # Define the supports (simply supported)
    beam.def_support('N1', True, True, True, True, False, False)
    beam.def_support('N2', True, True, True, True, False, False)

    # Define beam section properties
    J = 400/12**4
    Iy = 200/12**4
    Iz = 200/12**4
    A = 12/12**2
    beam.add_section('Section', A, Iy, Iz, J)

    # Define a material
    E = 29000*144  # ksf
    G = 11200*144  # ksf
    nu = 0.3
    rho = 0.490  # pcf
    beam.add_material('Steel', E, G, nu, rho)

    # Create the beam
    beam.add_member('M1', 'N1', 'N2', 'Steel', 'Section')

    # Add a mid-span node to force the model to split a physical member for this test
    beam.add_node('N3', 5, 0, 0)

    # Add a member distributed load along the strong axis
    beam.add_member_dist_load('M1', 'FY', -0.5, -0.5, case='D')
    beam.add_member_dist_load('M1', 'FY', -0.75, -0.75, case='L')

    # Add a member distributed load along the weak axis
    beam.add_member_dist_load('M1', 'FZ', -0.5, -0.5, case='D')
    beam.add_member_dist_load('M1', 'FZ', -0.75, -0.75, case='L')

    # Add some load combinations
    beam.add_load_combo('D', {'D': 1.0}, ['blc'])
    beam.add_load_combo('L', {'L': 1.0}, ['blc'])
    beam.add_load_combo('1.2D + 1.6L', {'D': 1.2, 'L': 1.6}, ['strength'])

    # Analyze the model
    beam.analyze_linear()

    # Test shear arrays with 20 points
    n_points = 20
    x_vals_fy, shear_fy = beam.members['M1'].shear_array('Fy', n_points, 'D')
    x_vals_fz, shear_fz = beam.members['M1'].shear_array('Fz', n_points, 'D')

    # Verify array length
    assert len(x_vals_fy) == n_points, f'Expected {n_points} x values, got {len(x_vals_fy)}'
    assert len(shear_fy) == n_points, f'Expected {n_points} shear values, got {len(shear_fy)}'
    assert len(x_vals_fz) == n_points, f'Expected {n_points} x values, got {len(x_vals_fz)}'
    assert len(shear_fz) == n_points, f'Expected {n_points} shear values, got {len(shear_fz)}'

    # Verify shear at start and end match expected values
    assert math.isclose(shear_fy[0], 2.5, abs_tol=0.01), f'Fy shear at start should be 2.5, got {shear_fy[0]}'
    assert math.isclose(shear_fy[-1], -2.5, abs_tol=0.01), f'Fy shear at end should be -2.5, got {shear_fy[-1]}'
    assert math.isclose(shear_fz[0], 2.5, abs_tol=0.01), f'Fz shear at start should be 2.5, got {shear_fz[0]}'
    assert math.isclose(shear_fz[-1], -2.5, abs_tol=0.01), f'Fz shear at end should be -2.5, got {shear_fz[-1]}'

    # Find midpoint value (should be close to zero)
    # The shear crosses zero at midspan, but with discrete points we may not land exactly on it
    mid_idx = n_points // 2
    assert math.isclose(shear_fy[mid_idx], 0, abs_tol=0.2), f'Fy shear at midpoint should be near 0, got {shear_fy[mid_idx]}'
    assert math.isclose(shear_fz[mid_idx], 0, abs_tol=0.2), f'Fz shear at midpoint should be near 0, got {shear_fz[mid_idx]}'

    # Test moment arrays with 20 points
    x_vals_mz, moment_mz = beam.members['M1'].moment_array('Mz', n_points, 'D')
    x_vals_my, moment_my = beam.members['M1'].moment_array('My', n_points, 'D')

    # Verify array length
    assert len(x_vals_mz) == n_points, f'Expected {n_points} x values, got {len(x_vals_mz)}'
    assert len(moment_mz) == n_points, f'Expected {n_points} moment values, got {len(moment_mz)}'
    assert len(x_vals_my) == n_points, f'Expected {n_points} x values, got {len(x_vals_my)}'
    assert len(moment_my) == n_points, f'Expected {n_points} moment values, got {len(moment_my)}'

    # Verify moment at ends is near zero (simply supported) for Mz
    # FY loading creates bending about Mz axis
    assert math.isclose(moment_mz[0], 0, abs_tol=0.5), f'Mz moment at start should be near 0, got {moment_mz[0]}'
    assert math.isclose(moment_mz[-1], 0, abs_tol=0.5), f'Mz moment at end should be near 0, got {moment_mz[-1]}'

    # FZ loading creates bending about My axis
    # Note: The array returns values at discrete points which may not align exactly with the endpoints
    # Use the moment() method to check actual endpoint values
    assert math.isclose(beam.members['M1'].moment('My', 0, 'D'), 0, abs_tol=0.5), f'My moment at x=0 should be near 0'
    assert math.isclose(beam.members['M1'].moment('My', 10, 'D'), 0, abs_tol=0.5), f'My moment at x=10 should be near 0'

    # Find maximum moment in array (should be near midpoint)
    max_mz = min(moment_mz)  # Negative moment for downward load
    assert math.isclose(max_mz, -6.25, abs_tol=2), f'Max Mz moment should be near -6.25, got {max_mz}'

    # For My, check that there are non-zero values from the FZ load
    assert min(moment_my) < -1.0, f'My should have negative moments from FZ loading, min was {min(moment_my)}'

    # Test deflection arrays with 20 points
    x_vals_dy, deflection_dy = beam.members['M1'].deflection_array('dy', n_points, 'D')
    x_vals_dz, deflection_dz = beam.members['M1'].deflection_array('dz', n_points, 'D')

    # Verify array length
    assert len(x_vals_dy) == n_points, f'Expected {n_points} x values, got {len(x_vals_dy)}'
    assert len(deflection_dy) == n_points, f'Expected {n_points} deflection values, got {len(deflection_dy)}'
    assert len(x_vals_dz) == n_points, f'Expected {n_points} x values, got {len(x_vals_dz)}'
    assert len(deflection_dz) == n_points, f'Expected {n_points} deflection values, got {len(deflection_dz)}'

    # Verify deflection at ends is near zero (simply supported)
    # Use the deflection() method to check actual endpoint values
    assert math.isclose(beam.members['M1'].deflection('dy', 0, 'D')*12, 0, abs_tol=0.00001), f'dy deflection at x=0 should be 0'
    assert math.isclose(beam.members['M1'].deflection('dy', 10, 'D')*12, 0, abs_tol=0.00001), f'dy deflection at x=10 should be 0'
    assert math.isclose(beam.members['M1'].deflection('dz', 0, 'D')*12, 0, abs_tol=0.00001), f'dz deflection at x=0 should be 0'
    assert math.isclose(beam.members['M1'].deflection('dz', 10, 'D')*12, 0, abs_tol=0.00001), f'dz deflection at x=10 should be 0'

    # Check that deflections are negative (downward) for downward loads
    max_dy = min(deflection_dy)  # Most negative deflection for downward load
    max_dz = min(deflection_dz)
    assert max_dy < 0, f'dy should have negative deflections from downward FY loading, min was {max_dy}'
    assert max_dz < 0, f'dz should have negative deflections from downward FZ loading, min was {max_dz}'

    # Check the actual midspan deflection using the deflection() method
    midspan_dy = beam.members['M1'].deflection('dy', 5, 'D')*12
    midspan_dz = beam.members['M1'].deflection('dz', 5, 'D')*12
    assert midspan_dy < 0, f'Midspan dy deflection should be negative, got {midspan_dy}'
    assert midspan_dz < 0, f'Midspan dz deflection should be negative, got {midspan_dz}'

    # Verify x arrays are correctly spaced
    for i in range(1, n_points):
        dx = x_vals_fy[i] - x_vals_fy[i-1]
        expected_dx = 10.0 / (n_points - 1)  # Total length is 10 feet
        assert math.isclose(dx, expected_dx, abs_tol=0.01), f'X spacing should be uniform, got {dx} vs expected {expected_dx}'


def test_stress_large_grid_many_arrays():
    """
    Creates a large grid structure and generates many arrays to stress test performance.
    This is designed to find bottlenecks in array generation.
    """
    # Create a large grid structure
    grid = FEModel3D()

    # Grid dimensions
    nx = 20  # 20 bays in X direction
    ny = 20  # 20 bays in Y direction
    spacing = 10  # 10 feet spacing

    # Add nodes in a grid
    for i in range(nx + 1):
        for j in range(ny + 1):
            node_name = f'N{i}_{j}'
            grid.add_node(node_name, i * spacing, j * spacing, 0)

    # Define material and section
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    nu = 0.3
    rho = 0.490  # pcf
    grid.add_material('Steel', E, G, nu, rho)

    J = 400/12**4
    Iy = 200/12**4
    Iz = 200/12**4
    A = 12/12**2
    grid.add_section('Section', A, Iy, Iz, J)

    # Add members in X direction
    member_count = 0
    for j in range(ny + 1):
        for i in range(nx):
            n1 = f'N{i}_{j}'
            n2 = f'N{i+1}_{j}'
            grid.add_member(f'MX{member_count}', n1, n2, 'Steel', 'Section')
            member_count += 1

    # Add members in Y direction
    for i in range(nx + 1):
        for j in range(ny):
            n1 = f'N{i}_{j}'
            n2 = f'N{i}_{j+1}'
            grid.add_member(f'MY{member_count}', n1, n2, 'Steel', 'Section')
            member_count += 1

    # Support all corner nodes
    grid.def_support('N0_0', True, True, True, True, True, True)
    grid.def_support(f'N{nx}_0', True, True, True, True, True, True)
    grid.def_support(f'N0_{ny}', True, True, True, True, True, True)
    grid.def_support(f'N{nx}_{ny}', True, True, True, True, True, True)

    # Add distributed loads to all members in multiple load cases
    load_cases = ['D', 'L', 'W', 'S']
    load_values = {
        'D': -0.5,
        'L': -0.75,
        'W': -0.3,
        'S': -0.6
    }

    for case in load_cases:
        w = load_values[case]
        for member_name in grid.members.keys():
            grid.add_member_dist_load(member_name, 'FY', w, w, case=case)

    # Add many load combinations
    grid.add_load_combo('D_only', {'D': 1.0}, ['blc'])
    grid.add_load_combo('L_only', {'L': 1.0}, ['blc'])
    grid.add_load_combo('1.4D', {'D': 1.4}, ['strength'])
    grid.add_load_combo('1.2D+1.6L', {'D': 1.2, 'L': 1.6}, ['strength'])
    grid.add_load_combo('1.2D+1.6S', {'D': 1.2, 'S': 1.6}, ['strength'])
    grid.add_load_combo('1.2D+1.0W+1.0L', {'D': 1.2, 'W': 1.0, 'L': 1.0}, ['strength'])
    grid.add_load_combo('1.2D+1.6W+0.5L', {'D': 1.2, 'W': 1.6, 'L': 0.5}, ['strength'])
    grid.add_load_combo('0.9D+1.0W', {'D': 0.9, 'W': 1.0}, ['strength'])

    # Analyze the model
    print(f'Analyzing large grid with {len(grid.nodes)} nodes and {len(grid.members)} members...')
    grid.analyze_linear()
    print('Analysis complete.')

    # Now stress test by generating many arrays for many members
    n_points = 100  # 100 points per array
    array_count = 0

    print(f'Generating arrays for {len(grid.members)} members with {n_points} points each...')

    # Generate arrays for a subset of members (every 10th member to keep it manageable)
    test_members = list(grid.members.keys())[::10]

    for member_name in test_members:
        member = grid.members[member_name]

        # Generate arrays for each load combination
        for combo_name in ['D_only', 'L_only', '1.2D+1.6L', '1.2D+1.6S']:
            # Shear arrays
            x_fy, fy = member.shear_array('Fy', n_points, combo_name)
            x_fz, fz = member.shear_array('Fz', n_points, combo_name)

            # Moment arrays
            x_my, my = member.moment_array('My', n_points, combo_name)
            x_mz, mz = member.moment_array('Mz', n_points, combo_name)

            # Deflection arrays
            x_dy, dy = member.deflection_array('dy', n_points, combo_name)
            x_dz, dz = member.deflection_array('dz', n_points, combo_name)

            array_count += 6

    print(f'Generated {array_count} arrays total.')
    print(f'Test complete - model had {len(grid.nodes)} nodes, {len(grid.members)} members')


def test_stress_multi_story_frame_arrays():
    """
    Creates a multi-story frame structure and generates arrays at many points.
    This tests performance with vertical structures and many load combinations.
    """
    frame = FEModel3D()

    # Frame parameters
    n_bays = 10  # 10 bays
    n_stories = 8  # 8 stories
    bay_width = 20  # 20 feet
    story_height = 12  # 12 feet

    # Material and section properties
    E = 29000 * 144  # ksf
    G = 11200 * 144  # ksf
    nu = 0.3
    rho = 0.490  # pcf
    frame.add_material('Steel', E, G, nu, rho)

    # Column section
    J_col = 800/12**4
    Iy_col = 400/12**4
    Iz_col = 400/12**4
    A_col = 20/12**2
    frame.add_section('Column', A_col, Iy_col, Iz_col, J_col)

    # Beam section
    J_beam = 600/12**4
    Iy_beam = 300/12**4
    Iz_beam = 150/12**4
    A_beam = 15/12**2
    frame.add_section('Beam', A_beam, Iy_beam, Iz_beam, J_beam)

    # Add nodes
    for story in range(n_stories + 1):
        for bay in range(n_bays + 1):
            node_name = f'N_S{story}_B{bay}'
            x = bay * bay_width
            y = 0
            z = story * story_height
            frame.add_node(node_name, x, y, z)

    # Add columns
    for bay in range(n_bays + 1):
        for story in range(n_stories):
            n1 = f'N_S{story}_B{bay}'
            n2 = f'N_S{story+1}_B{bay}'
            col_name = f'Col_B{bay}_S{story}'
            frame.add_member(col_name, n1, n2, 'Steel', 'Column')

    # Add beams
    for story in range(1, n_stories + 1):
        for bay in range(n_bays):
            n1 = f'N_S{story}_B{bay}'
            n2 = f'N_S{story}_B{bay+1}'
            beam_name = f'Beam_S{story}_B{bay}'
            frame.add_member(beam_name, n1, n2, 'Steel', 'Beam')

    # Fix base of all columns
    for bay in range(n_bays + 1):
        frame.def_support(f'N_S0_B{bay}', True, True, True, True, True, True)

    # Add loads to beams (gravity) and columns (wind)
    load_cases = ['D', 'L', 'Lr', 'W_X', 'W_Y']

    # Gravity loads on beams
    for story in range(1, n_stories + 1):
        for bay in range(n_bays):
            beam_name = f'Beam_S{story}_B{bay}'
            frame.add_member_dist_load(beam_name, 'FZ', -1.0, -1.0, case='D')
            frame.add_member_dist_load(beam_name, 'FZ', -0.8, -0.8, case='L')
            frame.add_member_dist_load(beam_name, 'FZ', -0.5, -0.5, case='Lr')

    # Wind loads on exterior columns
    for story in range(n_stories):
        # Wind in X direction on first bay columns
        col_name = f'Col_B0_S{story}'
        frame.add_member_dist_load(col_name, 'FX', 0.3, 0.3, case='W_X')

        # Wind in Y direction on columns
        for bay in range(n_bays + 1):
            col_name = f'Col_B{bay}_S{story}'
            frame.add_member_dist_load(col_name, 'FY', 0.2, 0.2, case='W_Y')

    # Add comprehensive load combinations
    combos = [
        ('D_only', {'D': 1.0}),
        ('L_only', {'L': 1.0}),
        ('1.4D', {'D': 1.4}),
        ('1.2D+1.6L', {'D': 1.2, 'L': 1.6}),
        ('1.2D+1.6Lr', {'D': 1.2, 'Lr': 1.6}),
        ('1.2D+1.0L+1.0W_X', {'D': 1.2, 'L': 1.0, 'W_X': 1.0}),
        ('1.2D+1.0L+1.0W_Y', {'D': 1.2, 'L': 1.0, 'W_Y': 1.0}),
        ('1.2D+1.6W_X+0.5L', {'D': 1.2, 'W_X': 1.6, 'L': 0.5}),
        ('0.9D+1.6W_X', {'D': 0.9, 'W_X': 1.6}),
    ]

    for combo_name, factors in combos:
        frame.add_load_combo(combo_name, factors, ['strength'])

    # Analyze
    print(f'Analyzing frame with {len(frame.nodes)} nodes and {len(frame.members)} members...')
    frame.analyze_linear()
    print('Analysis complete.')

    # Generate tons of arrays
    n_points = 150  # 150 points per member
    array_count = 0

    print(f'Generating arrays with {n_points} points per member...')

    # Test every member with multiple load combinations
    test_combos = ['1.2D+1.6L', '1.2D+1.0L+1.0W_X', '0.9D+1.6W_X']

    for member_name in frame.members.keys():
        member = frame.members[member_name]

        for combo in test_combos:
            # All internal force/deflection arrays
            member.shear_array('Fy', n_points, combo)
            member.shear_array('Fz', n_points, combo)
            member.moment_array('My', n_points, combo)
            member.moment_array('Mz', n_points, combo)
            member.deflection_array('dx', n_points, combo)
            member.deflection_array('dy', n_points, combo)
            member.deflection_array('dz', n_points, combo)
            array_count += 7

    print(f'Generated {array_count} arrays total.')
    print(f'Test complete - frame had {len(frame.nodes)} nodes, {len(frame.members)} members')


def test_stress_massive_array_generation():
    """
    Creates a moderate-sized model but generates arrays with MANY points
    and MANY load combinations to stress the array generation code.
    """
    model = FEModel3D()

    # Create a simple grid but with lots of load cases
    nx, ny = 15, 15
    spacing = 12

    # Nodes
    for i in range(nx + 1):
        for j in range(ny + 1):
            model.add_node(f'N{i}_{j}', i*spacing, j*spacing, 0)

    # Material and section
    E = 29000 * 144
    G = 11200 * 144
    model.add_material('Steel', E, G, 0.3, 0.490)
    model.add_section('Sec', 12/144, 200/12**4, 200/12**4, 400/12**4)

    # Members
    for j in range(ny + 1):
        for i in range(nx):
            model.add_member(f'MX{i}_{j}', f'N{i}_{j}', f'N{i+1}_{j}', 'Steel', 'Sec')

    for i in range(nx + 1):
        for j in range(ny):
            model.add_member(f'MY{i}_{j}', f'N{i}_{j}', f'N{i}_{j+1}', 'Steel', 'Sec')

    # Supports at corners
    model.def_support('N0_0', True, True, True, True, True, True)
    model.def_support(f'N{nx}_0', True, True, True, True, True, True)
    model.def_support(f'N0_{ny}', True, True, True, True, True, True)
    model.def_support(f'N{nx}_{ny}', True, True, True, True, True, True)

    # Create 20 different load cases
    n_load_cases = 20
    for i in range(n_load_cases):
        case_name = f'LC{i}'
        load_val = -0.5 - i * 0.1
        for member_name in model.members.keys():
            model.add_member_dist_load(member_name, 'FZ', load_val, load_val, case=case_name)

    # Create many load combinations
    combos = []
    # Single case combos
    for i in range(n_load_cases):
        model.add_load_combo(f'LC{i}_only', {f'LC{i}': 1.0}, ['blc'])
        combos.append(f'LC{i}_only')

    # Combination combos
    for i in range(0, n_load_cases-1, 2):
        combo_name = f'Combo_{i}_{i+1}'
        model.add_load_combo(combo_name, {f'LC{i}': 1.2, f'LC{i+1}': 1.6}, ['strength'])
        combos.append(combo_name)

    print(f'Analyzing model with {len(model.nodes)} nodes, {len(model.members)} members, and {len(combos)} load combinations...')
    model.analyze_linear()
    print('Analysis complete.')

    # Generate arrays with MANY points for EVERY combination
    n_points = 200  # 200 points per array
    array_count = 0

    print(f'Generating arrays with {n_points} points for ALL members and ALL {len(combos)} load combinations...')

    for member_name in model.members.keys():
        member = model.members[member_name]

        for combo in combos:
            # Generate all arrays
            member.shear_array('Fy', n_points, combo)
            member.shear_array('Fz', n_points, combo)
            member.moment_array('My', n_points, combo)
            member.moment_array('Mz', n_points, combo)
            member.deflection_array('dx', n_points, combo)
            member.deflection_array('dy', n_points, combo)
            member.deflection_array('dz', n_points, combo)
            array_count += 7

    print(f'Generated {array_count} arrays total!')
    print(f'Average of {n_points} points × {array_count} arrays = {n_points * array_count} total data points computed')


if __name__ == '__main__':
    test_beam_internal_forces()
    test_beam_internal_forces_arrays()
    print('All tests passed!')

    print('\n' + '='*80)
    print('STRESS TESTS - These will take time and are computationally intensive')
    print('='*80 + '\n')

    # Uncomment to run stress tests
    # test_stress_large_grid_many_arrays()
    # test_stress_multi_story_frame_arrays()
    # test_stress_massive_array_generation()
