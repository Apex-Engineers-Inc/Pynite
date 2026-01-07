#!/usr/bin/env python3
"""
Simple test script to verify parallel load combination analysis works correctly.

This script creates a simple beam model with multiple load combinations and
tests the analyze_linear() method with parallel processing enabled.
"""

from Pynite import FEModel3D
from Pynite.ParallelUtils import is_free_threaded
import sys

def create_simple_beam_model():
    """Create a simple supported beam with 8 load combinations."""
    # Create a new model
    model = FEModel3D()

    # Add nodes (simple supported beam)
    model.add_node('N1', 0, 0, 0)
    model.add_node('N2', 10, 0, 0)
    model.add_node('N3', 20, 0, 0)

    # Define a material
    E = 29000  # ksi
    G = 11200  # ksi
    nu = 0.3
    rho = 0.490  # kci (kip/in^3)
    model.add_material('Steel', E, G, nu, rho)

    # Define a section
    Iy = 100  # in^4
    Iz = 150  # in^4
    J = 250   # in^4
    A = 10    # in^2
    model.add_section('W12x26', A, Iy, Iz, J)

    # Add members
    model.add_member('M1', 'N1', 'N2', 'Steel', 'W12x26')
    model.add_member('M2', 'N2', 'N3', 'Steel', 'W12x26')

    # Add supports
    model.def_support('N1', True, True, True, True, False, False)  # Pin
    model.def_support('N3', False, True, True, False, False, False)  # Roller

    # Add load cases
    model.add_member_dist_load('M1', 'Fy', -0.5, -0.5, 0, 10, 'D')  # Dead load
    model.add_member_dist_load('M1', 'Fy', -1.0, -1.0, 0, 10, 'L')  # Live load
    model.add_member_dist_load('M2', 'Fy', -0.5, -0.5, 0, 10, 'D')
    model.add_member_dist_load('M2', 'Fy', -1.0, -1.0, 0, 10, 'L')

    # Add load combinations (8 combos to test parallelization)
    model.add_load_combo('1.4D', {'D': 1.4})
    model.add_load_combo('1.2D+1.6L', {'D': 1.2, 'L': 1.6})
    model.add_load_combo('1.2D+L', {'D': 1.2, 'L': 1.0})
    model.add_load_combo('D', {'D': 1.0})
    model.add_load_combo('D+L', {'D': 1.0, 'L': 1.0})
    model.add_load_combo('0.9D', {'D': 0.9})
    model.add_load_combo('1.2D+0.5L', {'D': 1.2, 'L': 0.5})
    model.add_load_combo('Service', {'D': 1.0, 'L': 0.5})

    return model


def test_parallel_analysis():
    """Test parallel analysis with free-threading detection."""
    print("=" * 60)
    print("Testing Parallel Load Combination Analysis")
    print("=" * 60)
    print()

    # Check if running on free-threaded Python
    print(f"Python version: {sys.version}")
    print(f"Free-threaded mode: {is_free_threaded()}")
    print()

    # Create test model
    print("Creating test model...")
    model = create_simple_beam_model()
    print(f"Model created with {len(model.load_combos)} load combinations")
    print()

    # Test 1: Analyze with parallel=True (auto-detect)
    print("Test 1: analyze_linear(parallel=True, log=True)")
    print("-" * 60)
    model.analyze_linear(log=True, parallel=True)
    print()

    # Check results
    import numpy as np
    print("Results:")
    for combo_name in model.load_combos.keys():
        if combo_name in model._D:
            max_disp = float(np.max(np.abs(model._D[combo_name])))
            print(f"  {combo_name}: max displacement = {max_disp:.6f} in")
    print()

    # Test 2: Analyze with parallel=False (force sequential)
    print("Test 2: analyze_linear(parallel=False)")
    print("-" * 60)
    model2 = create_simple_beam_model()
    model2.analyze_linear(log=False, parallel=False)
    print("Sequential analysis completed successfully")
    print()

    # Verify results match
    print("Verifying results match between parallel and sequential...")
    all_match = True
    for combo_name in model.load_combos.keys():
        if combo_name in model._D and combo_name in model2._D:
            max_diff = float(np.max(np.abs(model._D[combo_name] - model2._D[combo_name])))
            if max_diff > 1e-10:
                print(f"  {combo_name}: MISMATCH (max diff = {max_diff})")
                all_match = False
            else:
                print(f"  {combo_name}: OK")

    if all_match:
        print()
        print("✓ All tests passed! Results match between parallel and sequential analysis.")
    else:
        print()
        print("✗ Tests failed! Results do not match.")
        return False

    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    if is_free_threaded():
        print("✓ Running on free-threaded Python - parallel processing was used")
    else:
        print("✓ Running on standard Python - sequential processing was used")
    print("✓ analyze_linear() works correctly with parallel parameter")
    print("✓ Results are consistent between parallel and sequential execution")

    return True


if __name__ == '__main__':
    success = test_parallel_analysis()
    sys.exit(0 if success else 1)
