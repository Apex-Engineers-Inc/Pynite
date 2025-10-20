"""
Performance test for large beam systems.

This test creates models with approximately 1000 beam-type members with various
lengths and support conditions to benchmark the performance of the FE solver.
"""

import time
import random
from Pynite import FEModel3D


def test_large_beam_grid_performance():
    """
    Creates a large grid of interconnected beams with approximately 1000 members.
    This represents a typical floor framing system with beams in both directions.
    """

    model = FEModel3D()

    # Define material and section properties (typical steel wide flange)
    E = 29000  # ksi
    G = 11200  # ksi
    nu = 0.3
    rho = 0.490  # kips/ft^3
    model.add_material('Steel', E, G, nu, rho)

    # W12x26 section properties
    A = 7.65  # in^2
    Iy = 204  # in^4
    Iz = 17.3  # in^4
    J = 0.300  # in^4
    model.add_section('W12x26', A, Iy, Iz, J)

    # Create a grid of beams
    # Grid dimensions: 32 x 32 nodes = 1024 nodes
    # Will create beams in X and Y directions
    grid_size = 32
    bay_width = 20 * 12  # 20 feet in inches

    print(f"Creating grid of {grid_size}x{grid_size} nodes...")
    start_time = time.time()

    # Create nodes
    for i in range(grid_size):
        for j in range(grid_size):
            node_name = f'N{i}_{j}'
            X = i * bay_width
            Y = 0
            Z = j * bay_width
            model.add_node(node_name, X, Y, Z)

    node_creation_time = time.time() - start_time
    print(f"Node creation time: {node_creation_time:.3f} seconds")
    print(f"Total nodes: {len(model.nodes)}")

    # Create beams in X direction
    beam_count = 0
    start_time = time.time()

    for j in range(grid_size):
        for i in range(grid_size - 1):
            beam_name = f'BX{i}_{j}'
            node_i = f'N{i}_{j}'
            node_j = f'N{i+1}_{j}'
            model.add_member(beam_name, node_i, node_j, 'Steel', 'W12x26')
            beam_count += 1

    # Create beams in Z direction
    for i in range(grid_size):
        for j in range(grid_size - 1):
            beam_name = f'BZ{i}_{j}'
            node_i = f'N{i}_{j}'
            node_j = f'N{i}_{j+1}'
            model.add_member(beam_name, node_i, node_j, 'Steel', 'W12x26')
            beam_count += 1

    beam_creation_time = time.time() - start_time
    print(f"Beam creation time: {beam_creation_time:.3f} seconds")
    print(f"Total beams: {beam_count}")

    # Add supports - pin supports on all perimeter nodes
    start_time = time.time()

    support_count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            # Support nodes on the perimeter
            if i == 0 or i == grid_size-1 or j == 0 or j == grid_size-1:
                node_name = f'N{i}_{j}'
                # Pin support (fixed translation, free rotation)
                model.def_support(node_name, True, True, True, False, False, False)
                support_count += 1

    support_creation_time = time.time() - start_time
    print(f"Support creation time: {support_creation_time:.3f} seconds")
    print(f"Total supports: {support_count}")

    # Add distributed loads to all beams
    start_time = time.time()

    w = -0.1  # kips/in (approximately 100 psf on 10 ft tributary width)

    for member in model.members.values():
        model.add_member_dist_load(member.name, 'Fy', w, w, case='D')

    load_creation_time = time.time() - start_time
    print(f"Load creation time: {load_creation_time:.3f} seconds")

    # Add load combination
    model.add_load_combo('1.0D', {'D': 1.0})

    # Analyze the model
    print("\nAnalyzing model...")
    start_time = time.time()
    model.analyze(log=False, check_statics=False, calc_reactions=False)
    analysis_time = time.time() - start_time

    print(f"Analysis time: {analysis_time:.3f} seconds")

    # Total time
    total_time = (node_creation_time + beam_creation_time +
                  support_creation_time + load_creation_time + analysis_time)
    print(f"\nTotal time: {total_time:.3f} seconds")

    # Performance metrics
    print(f"\nPerformance metrics:")
    safe_node_time = node_creation_time if node_creation_time > 0 else 1e-9
    safe_beam_time = beam_creation_time if beam_creation_time > 0 else 1e-9
    safe_analysis_time = analysis_time if analysis_time > 0 else 1e-9
    print(f"  Nodes/second (creation): {len(model.nodes)/safe_node_time:.1f}")
    print(f"  Beams/second (creation): {beam_count/safe_beam_time:.1f}")
    print(f"  Nodes/second (analysis): {len(model.nodes)/safe_analysis_time:.1f}")
    print(f"  Beams/second (analysis): {beam_count/safe_analysis_time:.1f}")

    # Verify results
    max_deflection = 0
    for node in model.nodes.values():
        if abs(node.DY['1.0D']) > abs(max_deflection):
            max_deflection = node.DY['1.0D']

    print(f"\nMax deflection: {max_deflection:.4f} inches")
    assert max_deflection < 0, "Expected negative deflection"

    return model


def test_cantilever_forest_performance():
    """
    Creates approximately 1000 cantilever beams with varying lengths.
    This tests the performance with many independent structural systems.
    """

    model = FEModel3D()

    # Define material and section properties
    E = 29000  # ksi
    G = 11200  # ksi
    nu = 0.3
    rho = 0.490  # kips/ft^3
    model.add_material('Steel', E, G, nu, rho)

    # Various section sizes
    sections = [
        ('W8x18', 5.26, 61.9, 7.97, 0.130),
        ('W10x22', 6.49, 118, 11.4, 0.239),
        ('W12x26', 7.65, 204, 17.3, 0.300),
        ('W14x30', 8.85, 291, 19.6, 0.462),
    ]

    for name, A, Iy, Iz, J in sections:
        model.add_section(name, A, Iy, Iz, J)

    print("Creating cantilever forest with ~1000 beams...")
    start_time = time.time()

    # Create a grid of cantilevers
    num_cantilevers = 1024
    grid_size = 32  # 32 x 32 = 1024
    spacing = 10 * 12  # 10 feet in inches

    random.seed(42)  # For reproducibility

    beam_count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            # Create two nodes for each cantilever
            base_node = f'N{i}_{j}_base'
            tip_node = f'N{i}_{j}_tip'
            beam_name = f'B{i}_{j}'

            # Random length between 5 and 20 feet
            length = random.uniform(5, 20) * 12

            # Random orientation (0, 90, 180, 270 degrees)
            angle_idx = random.randint(0, 3)
            angles = [0, 90, 180, 270]
            angle = angles[angle_idx]

            # Position based on grid
            base_X = i * spacing
            base_Z = j * spacing

            # Tip position based on angle and length
            if angle == 0:
                tip_X = base_X + length
                tip_Z = base_Z
            elif angle == 90:
                tip_X = base_X
                tip_Z = base_Z + length
            elif angle == 180:
                tip_X = base_X - length
                tip_Z = base_Z
            else:  # 270
                tip_X = base_X
                tip_Z = base_Z - length

            # Random height
            height = random.uniform(0, 5) * 12

            # Add nodes
            model.add_node(base_node, base_X, height, base_Z)
            model.add_node(tip_node, tip_X, height, tip_Z)

            # Random section
            section_name = random.choice([s[0] for s in sections])

            # Add beam
            model.add_member(beam_name, base_node, tip_node, 'Steel', section_name)

            # Fixed support at base
            model.def_support(base_node, True, True, True, True, True, True)

            # Point load at tip (random between 1 and 10 kips)
            tip_load = -random.uniform(1, 10)
            model.add_node_load(tip_node, 'FY', tip_load, case='L')

            beam_count += 1

    model_creation_time = time.time() - start_time
    print(f"Model creation time: {model_creation_time:.3f} seconds")
    print(f"Total nodes: {len(model.nodes)}")
    print(f"Total beams: {beam_count}")

    # Add load combination
    model.add_load_combo('1.0L', {'L': 1.0})

    # Analyze the model
    print("\nAnalyzing model...")
    start_time = time.time()
    model.analyze(log=False, check_statics=False, calc_reactions=False)
    analysis_time = time.time() - start_time

    print(f"Analysis time: {analysis_time:.3f} seconds")

    # Total time
    total_time = model_creation_time + analysis_time
    print(f"Total time: {total_time:.3f} seconds")

    # Performance metrics
    print(f"\nPerformance metrics:")
    print(f"  Beams/second (creation): {beam_count/model_creation_time:.1f}")
    print(f"  Beams/second (analysis): {beam_count/analysis_time:.1f}")

    # Verify some results exist
    sample_node = f'N0_0_tip'
    deflection = model.nodes[sample_node].DY['1.0L']
    print(f"\nSample deflection at {sample_node}: {deflection:.4f} inches")
    assert deflection < 0, "Expected negative deflection"

    return model


def test_continuous_multi_span_beams():
    """
    Creates approximately 1000 members as continuous multi-span beams.
    This tests performance with various support conditions.
    """

    model = FEModel3D()

    # Define material and section properties
    E = 29000  # ksi
    G = 11200  # ksi
    nu = 0.3
    rho = 0.490  # kips/ft^3
    model.add_material('Steel', E, G, nu, rho)

    A = 7.65  # in^2
    Iy = 204  # in^4
    Iz = 17.3  # in^4
    J = 0.300  # in^4
    model.add_section('W12x26', A, Iy, Iz, J)

    print("Creating continuous multi-span beam system with ~1000 members...")
    start_time = time.time()

    # Create 64 continuous beams, each with 16 spans (64 * 16 = 1024 members)
    num_beams = 64
    spans_per_beam = 16
    span_length = 20 * 12  # 20 feet in inches
    beam_spacing = 10 * 12  # 10 feet spacing between parallel beams

    random.seed(42)  # For reproducibility

    total_members = 0

    for beam_idx in range(num_beams):
        # Create nodes for this continuous beam
        for node_idx in range(spans_per_beam + 1):
            node_name = f'N{beam_idx}_{node_idx}'
            X = node_idx * span_length
            Y = 0
            Z = beam_idx * beam_spacing
            model.add_node(node_name, X, Y, Z)

        # Create members for this continuous beam
        for member_idx in range(spans_per_beam):
            member_name = f'M{beam_idx}_{member_idx}'
            node_i = f'N{beam_idx}_{member_idx}'
            node_j = f'N{beam_idx}_{member_idx + 1}'
            model.add_member(member_name, node_i, node_j, 'Steel', 'W12x26')
            total_members += 1

        # Add supports with varying conditions
        for node_idx in range(spans_per_beam + 1):
            node_name = f'N{beam_idx}_{node_idx}'

            if node_idx == 0:
                # First node - fixed support
                model.def_support(node_name, True, True, True, True, True, False)
            elif node_idx == spans_per_beam:
                # Last node - roller support
                model.def_support(node_name, False, True, False, False, False, False)
            else:
                # Interior supports - vary between pin and roller
                if random.random() < 0.5:
                    # Pin support
                    model.def_support(node_name, True, True, True, False, False, False)
                else:
                    # Roller support
                    model.def_support(node_name, False, True, False, False, False, False)

        # Add distributed loads to all members in this beam
        w = random.uniform(-0.05, -0.15)  # Random load between 50-150 plf converted to kips/in
        for member_idx in range(spans_per_beam):
            member_name = f'M{beam_idx}_{member_idx}'
            model.add_member_dist_load(member_name, 'Fy', w, w, case='D')

    model_creation_time = time.time() - start_time
    print(f"Model creation time: {model_creation_time:.3f} seconds")
    print(f"Total nodes: {len(model.nodes)}")
    print(f"Total members: {total_members}")
    print(f"Total supports: {len([n for n in model.nodes.values() if any([n.support_DX, n.support_DY, n.support_DZ])])}")

    # Add load combination
    model.add_load_combo('1.0D', {'D': 1.0})

    # Analyze the model
    print("\nAnalyzing model...")
    start_time = time.time()
    model.analyze(log=False, check_statics=False, calc_reactions=False)
    analysis_time = time.time() - start_time

    print(f"Analysis time: {analysis_time:.3f} seconds")

    # Total time
    total_time = model_creation_time + analysis_time
    print(f"Total time: {total_time:.3f} seconds")

    # Performance metrics
    print(f"\nPerformance metrics:")
    print(f"  Members/second (creation): {total_members/model_creation_time:.1f}")
    print(f"  Members/second (analysis): {total_members/analysis_time:.1f}")

    # Check for maximum deflection by sampling midspan behavior of each member
    max_deflection = 0.0
    critical_location = None
    for member in model.members.values():
        # Sample several points along the member's length to capture peak response
        x_vals, deflections = member.deflection_array('dy', n_points=11, combo_name='1.0D')
        for x_local, deflection in zip(x_vals, deflections):
            if abs(deflection) > abs(max_deflection):
                max_deflection = deflection
                critical_location = f"{member.name}@{x_local:.2f}in"

    print(f"\nMax deflection: {max_deflection:.4f} inches at {critical_location}")
    assert max_deflection < 0, "Expected negative deflection"

    return model


def test_space_frame_performance():
    """
    Creates a 3D space frame with approximately 1000 members.
    Members are oriented in various directions in 3D space.
    """

    model = FEModel3D()

    # Define material and section properties
    E = 29000  # ksi
    G = 11200  # ksi
    nu = 0.3
    rho = 0.490  # kips/ft^3
    model.add_material('Steel', E, G, nu, rho)

    # HSS section for space frame
    A = 3.37  # in^2
    I = 7.80  # in^4 (assuming circular section, Iy = Iz)
    J = 15.6  # in^4
    model.add_section('HSS4x4', A, I, I, J)

    print("Creating 3D space frame with ~1000 members...")
    start_time = time.time()

    # Create a 3D grid: 11x11x9 nodes = 1089 nodes
    # This will create members between nodes in all directions
    nx, ny, nz = 11, 9, 11
    spacing = 10 * 12  # 10 feet in inches

    # Create all nodes
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                node_name = f'N{i}_{j}_{k}'
                X = i * spacing
                Y = j * spacing
                Z = k * spacing
                model.add_node(node_name, X, Y, Z)

    # Create members
    member_count = 0

    # Horizontal members in X direction
    for j in range(ny):
        for k in range(nz):
            for i in range(nx - 1):
                member_name = f'MX{i}_{j}_{k}'
                node_i = f'N{i}_{j}_{k}'
                node_j = f'N{i+1}_{j}_{k}'
                model.add_member(member_name, node_i, node_j, 'Steel', 'HSS4x4')
                member_count += 1

    # Horizontal members in Z direction
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1):
                member_name = f'MZ{i}_{j}_{k}'
                node_i = f'N{i}_{j}_{k}'
                node_j = f'N{i}_{j}_{k+1}'
                model.add_member(member_name, node_i, node_j, 'Steel', 'HSS4x4')
                member_count += 1

    # Vertical members in Y direction
    for i in range(nx):
        for k in range(nz):
            for j in range(ny - 1):
                member_name = f'MY{i}_{j}_{k}'
                node_i = f'N{i}_{j}_{k}'
                node_j = f'N{i}_{j+1}_{k}'
                model.add_member(member_name, node_i, node_j, 'Steel', 'HSS4x4')
                member_count += 1

    model_creation_time = time.time() - start_time
    print(f"Model creation time: {model_creation_time:.3f} seconds")
    print(f"Total nodes: {len(model.nodes)}")
    print(f"Total members: {member_count}")

    # Add supports at the base (j=0)
    start_time = time.time()
    support_count = 0
    for i in range(nx):
        for k in range(nz):
            node_name = f'N{i}_0_{k}'
            # Pin supports
            model.def_support(node_name, True, True, True, False, False, False)
            support_count += 1

    support_time = time.time() - start_time
    print(f"Support creation time: {support_time:.3f} seconds")
    print(f"Total supports: {support_count}")

    # Add gravity loads to all nodes
    start_time = time.time()
    for node in model.nodes.values():
        # Apply a small downward force at each node
        model.add_node_load(node.name, 'FY', -1.0, case='D')

    load_time = time.time() - start_time
    print(f"Load creation time: {load_time:.3f} seconds")

    # Add load combination
    model.add_load_combo('1.0D', {'D': 1.0})

    # Analyze the model
    print("\nAnalyzing model...")
    start_time = time.time()
    model.analyze(log=False, check_statics=False, calc_reactions=False)
    analysis_time = time.time() - start_time

    print(f"Analysis time: {analysis_time:.3f} seconds")

    # Total time
    total_time = model_creation_time + support_time + load_time + analysis_time
    print(f"Total time: {total_time:.3f} seconds")

    # Performance metrics
    print(f"\nPerformance metrics:")
    print(f"  Members/second (creation): {member_count/model_creation_time:.1f}")
    print(f"  Members/second (analysis): {member_count/analysis_time:.1f}")

    # Find maximum deflection
    max_deflection = 0
    for node in model.nodes.values():
        if abs(node.DY['1.0D']) > abs(max_deflection):
            max_deflection = node.DY['1.0D']

    print(f"\nMax deflection: {max_deflection:.4f} inches")
    assert max_deflection < 0, "Expected negative deflection"

    return model


if __name__ == '__main__':
    print("="*80)
    print("BEAM PERFORMANCE TEST SUITE")
    print("="*80)

    print("\n" + "="*80)
    print("Test 1: Large Beam Grid (Floor Framing System)")
    print("="*80)
    test_large_beam_grid_performance()

    print("\n" + "="*80)
    print("Test 2: Cantilever Forest (Independent Systems)")
    print("="*80)
    test_cantilever_forest_performance()

    print("\n" + "="*80)
    print("Test 3: Continuous Multi-Span Beams (Various Support Conditions)")
    print("="*80)
    test_continuous_multi_span_beams()

    print("\n" + "="*80)
    print("Test 4: 3D Space Frame")
    print("="*80)
    test_space_frame_performance()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
