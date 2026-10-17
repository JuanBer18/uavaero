"""Shared skip markers: tests degrade gracefully on a machine missing a heavy optional solver.

The pure-Python schema and arithmetic tests must always run. A skip is not a pass — the reason
string says which install is missing, because the failure mode this file exists to prevent is a
suite that quietly stops testing anything (a path change once turned nine tests from passed to
skipped with "avl binary not found" and nothing failed).
"""
from __future__ import annotations

import importlib.util
import os
import shutil

import pytest


def _missing(module: str) -> bool:
    return importlib.util.find_spec(module) is None


def _binary_missing(resolver) -> bool:
    """A solver binary, not a Python package — find_spec does not apply."""
    exe = str(resolver())
    return shutil.which(exe) is None and not os.path.isfile(exe)


def _xfoil_missing() -> bool:
    from uavaero.solvers.sections.xfoil import xfoil_binary_path

    return _binary_missing(xfoil_binary_path)


requires_openvsp = pytest.mark.skipif(_missing("openvsp"), reason="openvsp not installed")
requires_aerosandbox = pytest.mark.skipif(_missing("aerosandbox"), reason="aerosandbox not installed")
requires_neuralfoil = pytest.mark.skipif(_missing("neuralfoil"), reason="neuralfoil not installed")
requires_build123d = pytest.mark.skipif(_missing("build123d"), reason="build123d not installed")
requires_pyvista = pytest.mark.skipif(_missing("pyvista"), reason="pyvista not installed")
requires_xfoil = pytest.mark.skipif(_xfoil_missing(), reason="xfoil binary not found")
