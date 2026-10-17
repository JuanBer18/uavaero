"""Moving a worker's `cwd` breaks `-m`, so `cwd` and `env` are one decision, not two.

Every OpenVSP entry point runs in a disposable subprocess, and every one of them needs its own
working directory: OpenVSP names its artefacts after the model and writes them beside wherever the
process is standing, so a worker that inherits the app's directory litters the repo.

But `python -m uavaero...` resolves the module *through the working directory*. Setting `cwd`
without putting the app root on `PYTHONPATH` kills the worker before it runs a line:

    ModuleNotFoundError: No module named 'uavaero'

`_worker_environment()` exists to supply exactly that, and this has now been the same mistake
twice — first in the optimiser's parallel workers, then in the `.vsp3` export, where it took out
both "Export" and "Open in OpenVSP" at once because they share one call. It is invisible in review:
the `cwd=` argument looks complete on its own, and the failure is at import time in another
process.

So the pairing is checked mechanically rather than remembered.
"""
from __future__ import annotations

import ast

import pytest

from uavaero.paths import APP_ROOT

#: Every module that launches a worker. Add to this when a new one appears — a launcher nobody
#: listed here is a launcher nobody checks.
LAUNCHER_MODULES = [
    APP_ROOT / "uavaero" / "app" / "main_window.py",
    APP_ROOT / "uavaero" / "solvers" / "vspaero.py",
    APP_ROOT / "uavaero" / "solvers" / "stability.py",
    APP_ROOT / "uavaero" / "app" / "viewport" / "probe.py",
]


def _subprocess_launches(tree: ast.AST) -> list[ast.Call]:
    """`subprocess.run(...)` and `subprocess.Popen(...)` calls that use `-m`."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if not (isinstance(callee, ast.Attribute) and callee.attr in ("run", "Popen")):
            continue
        if not any(
            isinstance(literal, ast.Constant) and literal.value == "-m"
            for argument in node.args
            for literal in ast.walk(argument)
        ):
            continue
        found.append(node)
    return found


def _keywords(call: ast.Call) -> set[str]:
    return {keyword.arg for keyword in call.keywords if keyword.arg}


@pytest.mark.parametrize("module", LAUNCHER_MODULES, ids=lambda p: p.name)
def test_a_worker_given_its_own_directory_is_also_given_its_path(module):
    """The regression itself, in both directions.

    A `-m` launch with `cwd` and no `env` cannot import the package. One with `env` and no `cwd`
    drops OpenVSP's scratch files wherever the parent happened to be standing.
    """
    launches = _subprocess_launches(ast.parse(module.read_text(encoding="utf-8")))
    assert launches, f"{module.name}: expected a worker launch, found none"
    for call in launches:
        keywords = _keywords(call)
        assert ("cwd" in keywords) == ("env" in keywords), (
            f"{module.name}:{call.lineno} passes "
            f"{'cwd without env - the worker cannot import uavaero' if 'cwd' in keywords else 'env without cwd - OpenVSP will litter the parent directory'}"
        )


def test_every_launcher_this_tree_has_is_actually_listed():
    """A launcher missing from LAUNCHER_MODULES is one this file silently stops protecting, which
    is the failure mode the markers file warns about: a check that quietly tests nothing."""
    listed = set(LAUNCHER_MODULES)
    found = set()
    for path in (APP_ROOT / "uavaero").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if _subprocess_launches(ast.parse(path.read_text(encoding="utf-8"))):
            found.add(path)
    assert found == listed, f"unlisted: {sorted(p.name for p in found - listed)}"
