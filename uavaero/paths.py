"""Where things live on disk, resolved once.

Counting `parents[N]` from a module's own `__file__` is a trap. It breaks the moment that module
moves to a different directory depth, and it breaks *quietly*: in the previous generation of this
tool a restructure moved two solver backends one level deeper, their `parents[2]` started pointing
at the package instead of the application root, and the only symptom was nine tests turning from
passed to skipped with the reason "avl binary not found". Nothing failed. The tests just stopped
running.

So nothing outside this module counts parents. Anything that needs a path relative to the tree
imports it from here, and a future move means changing one line.

**On sharing `external/` and `data/` with the previous tree.** Between them they are roughly 450 MB
of solver binaries and downloaded reference data — gitignored, machine-setup artifacts, never
written to at runtime. Duplicating them into this tree would buy nothing and cost half a gigabyte,
so both resolvers fall back to the sibling `Software/` tree when this one has no local copy. That
is a *data* dependency, not a code one: nothing here imports from `Software/`, and the fallback
disappears the moment you drop a local `external/` or `data/` beside this file or set the
environment override.
"""
from __future__ import annotations

import os
from pathlib import Path

#: The `uavaero` package directory itself.
PACKAGE_ROOT = Path(__file__).resolve().parent

#: The application root — the directory holding `uavaero/`, `tests/`, `pyproject.toml`.
APP_ROOT = PACKAGE_ROOT.parent

#: The previous generation of the tool, kept as a read-only donor for the shared binary and data
#: sets. Frozen: never imported, never edited. See this tree's CLAUDE.md.
_LEGACY_ROOT = APP_ROOT.parent / "Software"


def _shared(env_var: str, name: str) -> Path:
    """Local copy if there is one, else the legacy tree's, else the local path regardless.

    Returning the local path even when nothing exists is deliberate — the caller gets a path that
    says where the thing *should* go, and every binary lookup below degrades to PATH anyway. A
    resolver that raised here would turn a missing optional solver into an import error.
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    local = APP_ROOT / name
    if local.is_dir():
        return local
    legacy = _LEGACY_ROOT / name
    if legacy.is_dir():
        return legacy
    return local


#: Bundled external solver binaries (OpenVSP, AVL, XFOIL). Gitignored: a local machine-setup
#: artifact, not shared state. Absent on a fresh clone, which is why every lookup below degrades to
#: PATH rather than raising.
EXTERNAL = _shared("UAVAERO_EXTERNAL", "external")

#: Bulk read-only reference data. Downloaded rather than authored, never written to at runtime —
#: and therefore gitignored, which means it needs its own backup.
DATA = _shared("UAVAERO_DATA", "data")

#: The UIUC/Selig airfoil coordinate set, in Selig `.dat` format.
AIRFOILS = DATA / "airfoils" / "coord_seligFmt"

#: Solver scratch space: meshes, logs, raw solver artefacts. Everything under here is disposable
#: and regenerable — if deleting it loses something, that something was in the wrong place.
RUNS = APP_ROOT / ".runs"

#: The cache database and its array sidecars. Disposable in the same sense as RUNS: losing it costs
#: solve time, never information.
CACHE = APP_ROOT / ".cache"

#: Saved airframes. The unit of work in this tool.
AIRFRAMES = APP_ROOT / "airframes"


def external_binary(tool: str, executable: str) -> Path:
    """Path a bundled solver binary *would* occupy. Existence is the caller's business — the
    resolution order (env override, bundled, PATH) belongs with the backend that knows the tool."""
    return EXTERNAL / tool / executable


def openvsp_gui() -> Path | None:
    """OpenVSP's own GUI, for the deliberate trip out of this app into theirs.

    Nothing in the workflow needs it: the app renders the same `ParametricAirframe` that produces
    the `.vsp3`, and VSPAERO runs headless through the Python package. This is for when you want
    OpenVSP's own tooling on a model — measuring, slicing, the parm browser.

    Returns None rather than raising when it is not installed, because a missing GUI is a button
    that should be disabled, not an error.
    """
    override = os.environ.get("UAVAERO_VSP_GUI")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    for candidate in sorted((EXTERNAL / "openvsp").glob("*/vsp.exe")):
        return candidate
    return None
