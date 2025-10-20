"""
Helper utilities for optional Numba acceleration across PyNite.

Numba is treated as an optional dependency. This module provides shims that let
the rest of the code base opt into ``@njit``/``prange`` when Numba is available
and the user has not disabled it explicitly. When Numba is missing (or turned
off), the helpers degrade gracefully to lightweight no-op implementations so
the surrounding code can stay unchanged.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Callable, TypeVar

_TFunc = TypeVar("_TFunc", bound=Callable[..., object])


def _should_enable_numba() -> bool:
    """
    Returns ``True`` when Numba should be used.

    Users can force-disable JIT compilation by setting the environment variable
    ``PYNITE_USE_NUMBA`` (or the legacy name ``PYNITEFEA_USE_NUMBA``) to any of
    ``{"0", "false", "no", "off"}``.
    """

    def _flag_value(name: str) -> str | None:
        value = os.getenv(name)
        return value.strip().lower() if value is not None else None

    for env_var in ("PYNITE_USE_NUMBA", "PYNITEFEA_USE_NUMBA"):
        decision = _flag_value(env_var)
        if decision is not None:
            return decision not in {"0", "false", "no", "off"}

    # Default to enabled when the environment is silent.
    return True


try:  # pragma: no cover - import guarded by availability
    from numba import config as _numba_config  # type: ignore
    from numba import njit as _numba_njit  # type: ignore
    from numba import prange as _numba_prange  # type: ignore

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - behaviour verified via fallback
    _numba_config = SimpleNamespace()
    _numba_njit = None  # type: ignore[assignment]
    _numba_prange = range  # type: ignore[assignment]
    NUMBA_AVAILABLE = False


USE_NUMBA = NUMBA_AVAILABLE and _should_enable_numba()

# --- Public shims --------------------------------------------------------- #


def _identity_decorator(
    *decorator_args: object, **decorator_kwargs: object
) -> Callable[[_TFunc], _TFunc] | _TFunc:
    """
    Returns a decorator that leaves the target function untouched.

    Handles both ``@decorator`` and ``@decorator(...)`` styles.
    """

    decorator_args = tuple(decorator_args)

    if (
        decorator_args
        and callable(decorator_args[0])
        and len(decorator_args) == 1
        and not decorator_kwargs
    ):
        # Direct decorator application without parameters, e.g. ``@njit``
        return decorator_args[0]  # type: ignore[return-value]

    def _wrapper(func: _TFunc) -> _TFunc:
        return func

    return _wrapper


if USE_NUMBA:
    njit = _numba_njit  # type: ignore[assignment]
    prange = _numba_prange  # type: ignore[assignment]
else:

    def njit(
        *decorator_args: object, **decorator_kwargs: object
    ) -> Callable[[_TFunc], _TFunc]:
        return _identity_decorator(*decorator_args, **decorator_kwargs)

    def prange(*args: int) -> range:
        return range(*args)


__all__ = ["NUMBA_AVAILABLE", "USE_NUMBA", "njit", "prange"]
