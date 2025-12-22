"""
Performance benchmark for wood framed wall design use case.

Simulates:
- Walls 4-40 ft long with studs at approximately 16 inch on center
- 10 load combinations per wall
- Extracting shear, moment, axial, and torsion arrays with 20 points each
- Iterating over all members for force extraction

This benchmark compares:
1. Traditional approach: Call individual array methods per combo
2. Optimized approach: Use bulk force extraction methods

Run with: python -m pytest Testing/benchmark_wall_design.py -v -s
"""

import time
import sys
sys.path.insert(0, '/home/user/Pynite')

from Pynite import FEModel3D
import numpy as np


def create_wall_model(wall_length_ft: float, stud_spacing_in: float = 16.0):
    """
    Create a wood framed wall model.

    Parameters
    ----------
    wall_length_ft : float
        Wall length in feet
    stud_spacing_in : float
        Stud spacing in inches (default 16")

    Returns
    -------
    FEModel3D
        The finite element model
    """
    model = FEModel3D()

    # Wall height (typical 8 ft)
    wall_height_in = 96.0  # 8 ft in inches

    # Convert wall length to inches
    wall_length_in = wall_length_ft * 12.0

    # Calculate number of studs
    num_studs = int(wall_length_in / stud_spacing_in) + 1

    # Wood material properties (SPF No. 2)
    E = 1400000  # psi (modulus of elasticity)
    G = E / 16   # Approximate shear modulus
    nu = 0.3     # Poisson's ratio
    rho = 0.000017  # lb/in^3 (density)

    model.add_material('Wood', E, G, nu, rho)

    # 2x4 stud section properties
    b = 1.5   # inches (actual width)
    d = 3.5   # inches (actual depth)
    A = b * d
    Iy = b * d**3 / 12
    Iz = d * b**3 / 12
    J = 0.141 * b * d**3  # Approximate for rectangular section

    model.add_section('2x4', A, Iy, Iz, J)

    # Top and bottom plate section (doubled 2x4)
    model.add_section('2x4_plate', 2*A, 2*Iy, 2*Iz, 2*J)

    # Create nodes for bottom and top of each stud
    for i in range(num_studs):
        x = i * stud_spacing_in
        # Bottom node
        model.add_node(f'N_bot_{i}', x, 0, 0)
        # Top node
        model.add_node(f'N_top_{i}', x, wall_height_in, 0)

    # Create stud members
    for i in range(num_studs):
        model.add_member(f'Stud_{i}', f'N_bot_{i}', f'N_top_{i}', 'Wood', '2x4')

    # Create bottom plate members
    for i in range(num_studs - 1):
        model.add_member(f'Bot_plate_{i}', f'N_bot_{i}', f'N_bot_{i+1}', 'Wood', '2x4_plate')

    # Create top plate members
    for i in range(num_studs - 1):
        model.add_member(f'Top_plate_{i}', f'N_top_{i}', f'N_top_{i+1}', 'Wood', '2x4_plate')

    # Add supports (pinned at bottom, roller at top)
    for i in range(num_studs):
        # Bottom: fixed against translation
        model.def_support(f'N_bot_{i}', True, True, True, False, False, False)

    # Add lateral load (wind)
    wind_load_plf = 20  # lb/ft = lb/12in
    wind_load_pli = wind_load_plf / 12.0  # lb/in

    # Add gravity load to top plates
    gravity_load_plf = 500  # lb/ft
    gravity_load_pli = gravity_load_plf / 12.0  # lb/in

    # Define load cases
    # Case 1: Dead load on top plates
    for i in range(num_studs - 1):
        model.add_member_dist_load(f'Top_plate_{i}', 'Fy', -gravity_load_pli/2, -gravity_load_pli/2, 0, stud_spacing_in, 'Dead')

    # Case 2: Live load on top plates
    for i in range(num_studs - 1):
        model.add_member_dist_load(f'Top_plate_{i}', 'Fy', -gravity_load_pli, -gravity_load_pli, 0, stud_spacing_in, 'Live')

    # Case 3: Wind load on studs
    for i in range(num_studs):
        model.add_member_dist_load(f'Stud_{i}', 'Fz', wind_load_pli, wind_load_pli, 0, wall_height_in, 'Wind')

    # Case 4: Snow load on top plates
    for i in range(num_studs - 1):
        model.add_member_dist_load(f'Top_plate_{i}', 'Fy', -gravity_load_pli*0.5, -gravity_load_pli*0.5, 0, stud_spacing_in, 'Snow')

    # Define 10 load combinations (typical ASCE 7 combinations)
    model.add_load_combo('1.4D', {'Dead': 1.4})
    model.add_load_combo('1.2D+1.6L', {'Dead': 1.2, 'Live': 1.6})
    model.add_load_combo('1.2D+1.6L+0.5S', {'Dead': 1.2, 'Live': 1.6, 'Snow': 0.5})
    model.add_load_combo('1.2D+1.6S+L', {'Dead': 1.2, 'Live': 1.0, 'Snow': 1.6})
    model.add_load_combo('1.2D+1.0W+L+0.5S', {'Dead': 1.2, 'Live': 1.0, 'Wind': 1.0, 'Snow': 0.5})
    model.add_load_combo('0.9D+1.0W', {'Dead': 0.9, 'Wind': 1.0})
    model.add_load_combo('1.2D+0.5L+1.0W', {'Dead': 1.2, 'Live': 0.5, 'Wind': 1.0})
    model.add_load_combo('1.0D+1.0L', {'Dead': 1.0, 'Live': 1.0})
    model.add_load_combo('1.0D+0.75L+0.75W', {'Dead': 1.0, 'Live': 0.75, 'Wind': 0.75})
    model.add_load_combo('1.0D+0.75L+0.75S', {'Dead': 1.0, 'Live': 0.75, 'Snow': 0.75})

    return model


def benchmark_traditional_extraction(model, n_points=20):
    """
    Traditional approach: Call individual array methods for each member and combo.
    """
    combo_names = list(model.load_combos.keys())
    results = {}

    for member_name, member in model.members.items():
        member_results = {}
        for combo_name in combo_names:
            combo_results = {}

            # Extract all force arrays
            shear_y = member.shear_array('Fy', n_points, combo_name)
            moment_z = member.moment_array('Mz', n_points, combo_name)
            axial = member.axial_array(n_points, combo_name)
            torque = member.torque_array(n_points, combo_name)

            combo_results['shear_y'] = shear_y
            combo_results['moment_z'] = moment_z
            combo_results['axial'] = axial
            combo_results['torque'] = torque

            member_results[combo_name] = combo_results
        results[member_name] = member_results

    return results


def benchmark_bulk_extraction(model, n_points=20):
    """
    Optimized approach: Use bulk force extraction methods.
    """
    combo_names = list(model.load_combos.keys())
    results = {}

    for member_name, member in model.members.items():
        # Use the new bulk extraction method
        member_results = member.get_all_forces_array(combo_names, n_points)
        results[member_name] = member_results

    return results


def run_benchmark():
    """Run the performance benchmark."""
    print("=" * 70)
    print("WOOD FRAMED WALL DESIGN - PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Test configurations
    wall_configs = [
        (8, "8 ft wall (~6 studs)"),
        (16, "16 ft wall (~12 studs)"),
        (24, "24 ft wall (~18 studs)"),
        (32, "32 ft wall (~24 studs)"),
        (40, "40 ft wall (~30 studs)"),
    ]

    n_points = 20
    n_iterations = 3  # Average over multiple runs

    print(f"\nConfiguration:")
    print(f"  - Points per array: {n_points}")
    print(f"  - Load combinations: 10")
    print(f"  - Stud spacing: 16 inches")
    print(f"  - Iterations per test: {n_iterations}")

    print("\n" + "-" * 70)
    print(f"{'Wall Size':<25} {'Members':<10} {'Traditional':<15} {'Bulk':<15} {'Speedup':<10}")
    print("-" * 70)

    for wall_length, description in wall_configs:
        # Create and analyze model
        model = create_wall_model(wall_length)
        model.analyze()

        n_members = len(model.members)

        # Benchmark traditional extraction
        trad_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            benchmark_traditional_extraction(model, n_points)
            trad_times.append(time.perf_counter() - start)
        trad_avg = np.mean(trad_times)

        # Benchmark bulk extraction
        bulk_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            benchmark_bulk_extraction(model, n_points)
            bulk_times.append(time.perf_counter() - start)
        bulk_avg = np.mean(bulk_times)

        speedup = trad_avg / bulk_avg if bulk_avg > 0 else float('inf')

        print(f"{description:<25} {n_members:<10} {trad_avg*1000:>10.2f} ms   {bulk_avg*1000:>10.2f} ms   {speedup:>6.1f}x")

    print("-" * 70)
    print("\nBenchmark complete!")

    return True


def test_correctness():
    """Verify that traditional and bulk extraction produce the same results."""
    print("\nVerifying correctness of bulk extraction...")

    model = create_wall_model(16)  # 16 ft wall
    model.analyze()

    combo_names = list(model.load_combos.keys())
    n_points = 20

    for member_name, member in model.members.items():
        # Get results both ways
        bulk_results = member.get_all_forces_array(combo_names, n_points)

        # Compare with traditional method
        for i, combo_name in enumerate(combo_names):
            trad_shear = member.shear_array('Fy', n_points, combo_name)
            trad_moment = member.moment_array('Mz', n_points, combo_name)
            trad_axial = member.axial_array(n_points, combo_name)
            trad_torque = member.torque_array(n_points, combo_name)

            # Check shear
            if not np.allclose(trad_shear[1], bulk_results['shear_y'][i], rtol=1e-10):
                print(f"  MISMATCH: {member_name} - {combo_name} - shear_y")
                return False

            # Check moment
            if not np.allclose(trad_moment[1], bulk_results['moment_z'][i], rtol=1e-10):
                print(f"  MISMATCH: {member_name} - {combo_name} - moment_z")
                return False

            # Check axial
            if not np.allclose(trad_axial[1], bulk_results['axial'][i], rtol=1e-10):
                print(f"  MISMATCH: {member_name} - {combo_name} - axial")
                return False

            # Check torque
            if not np.allclose(trad_torque[1], bulk_results['torque'][i], rtol=1e-10):
                print(f"  MISMATCH: {member_name} - {combo_name} - torque")
                return False

    print("  All results match! Bulk extraction is correct.")
    return True


if __name__ == '__main__':
    test_correctness()
    run_benchmark()
