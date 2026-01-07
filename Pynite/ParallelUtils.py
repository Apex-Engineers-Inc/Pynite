"""
Utilities for parallel execution of finite element analysis.

This module provides utilities for detecting free-threaded Python (no-GIL)
and parallelizing load combination analysis using ThreadPoolExecutor.
"""

import sys
import os


def is_free_threaded() -> bool:
    """
    Detect if Python is running in free-threaded mode (without the GIL).

    Python 3.13+ with free-threading enabled has the sys._is_gil_enabled() function.
    This function returns False when the GIL is disabled.

    Returns:
        bool: True if Python is running without the GIL (free-threaded mode),
              False otherwise (standard Python with GIL).

    Examples:
        >>> # On Python 3.14t (free-threaded build)
        >>> is_free_threaded()
        True

        >>> # On standard Python 3.13 or earlier
        >>> is_free_threaded()
        False
    """
    # Python 3.13+ with free-threading support has sys._is_gil_enabled
    if hasattr(sys, '_is_gil_enabled'):
        try:
            # If GIL is disabled, we're in free-threaded mode
            # Use getattr to avoid type checker issues with private attribute
            is_gil_enabled = getattr(sys, '_is_gil_enabled')
            return not is_gil_enabled()
        except Exception:
            # If there's any error calling the function, assume GIL is enabled
            return False

    # Older Python versions always have the GIL enabled
    return False


def get_optimal_worker_count(combo_count: int, max_workers: int | None = None) -> int:
    """
    Determine the optimal number of worker threads for parallelizing load combinations.

    Args:
        combo_count: Number of load combinations to analyze
        max_workers: Maximum number of workers to use (None = auto-detect from CPU count)

    Returns:
        int: Optimal number of workers to use

    Notes:
        - Returns 1 (sequential) if combo_count < 4 (overhead not worth it)
        - Otherwise returns min(combo_count, cpu_count, max_workers)
    """
    # Not worth parallelizing for small combo counts
    if combo_count < 4:
        return 1

    # Get CPU count (defaults to 1 if can't determine)
    cpu_count = os.cpu_count() or 1

    # Start with the CPU count
    workers = cpu_count

    # No point having more workers than combos
    workers = min(workers, combo_count)

    # Respect user's max_workers if specified
    if max_workers is not None:
        workers = min(workers, max_workers)

    return workers
