"""Permanent regression cases for the three former OpenVSP meshing bugs.

Each case runs through the real worker SUBPROCESS — a native OpenVSP crash (the original
vertical-tail failure mode) becomes a nonzero exit code and a test failure, never a dead pytest
process or a silent hang.

These three cost days to find and are the single most expensive piece of knowledge carried into
this tree. They are the reason the OpenVSP adapter was promoted rather than rewritten: nothing
about the bugs is visible in the geometry that triggers them, and each one fails by producing a
mesh that looks fine and solves to zeros.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.markers import requires_openvsp
from tests.reference_cases import cases

pytestmark = requires_openvsp


def run_worker(airframe, tmp_path: Path) -> dict:
    job_file = tmp_path / f"{airframe.name}.job.json"
    result_file = tmp_path / f"{airframe.name}.result.json"
    job_file.write_text(json.dumps({
        "spec": airframe.to_dict(),
        "vsp3_out": str(tmp_path / f"{airframe.name}.vsp3"),
        "run_compgeom": True,
        "result_out": str(result_file),
    }), encoding="utf-8")

    # `cwd` is the tmp dir, not the repo root. CompGeom writes `<model>_CompGeom.csv` and `.txt`
    # beside wherever the process is standing, so a worker run from the root litters the tree —
    # and three of those files reached a commit before anyone looked. Moving cwd breaks `-m`
    # module resolution, which is what `_worker_environment` exists to put back (CLAUDE.md 21).
    from uavaero.solvers.vspaero import _worker_environment

    proc = subprocess.run(
        [sys.executable, "-m", "uavaero.geometry.adapters.vsp_worker", str(job_file)],
        cwd=str(tmp_path), env=_worker_environment(),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert proc.returncode == 0, (
        f"worker failed (exit {proc.returncode}) — a crash here means an OpenVSP bug regressed:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result["ok"], f"mesh check not ok: {result}"
    return result


@pytest.mark.parametrize("build,expected_components", [
    (cases.vertical_tail_case, 2),   # bug (a): 90°-rotated tail + symmetry flag -> native crash
    (cases.dihedral_twist_case, 1),  # bug (b): dihedral+twist same section -> degenerate mesh
    (cases.thin_tail_case, 2),       # bug (c): thin symmetric tail section -> degenerate mesh
])
def test_former_bug_case_passes_mesh_check(build, expected_components, tmp_path):
    airframe = build()
    result = run_worker(airframe, tmp_path)
    compgeom = result["compgeom"]
    assert compgeom["total_num_tris"] > 0
    assert compgeom["total_wetted_area"] > 0
    assert len(compgeom["per_component"]) == expected_components
    assert all(area > 0 for area in compgeom["per_component"].values())
    assert Path(result["vsp3"]).exists()


def test_vertical_tail_symmetry_flag_is_actually_off(tmp_path):
    """The specific bug-(a) fix: the rotated tail geom must have planar symmetry disabled in the
    written .vsp3 (checked in-process here — build only, no mesher involved)."""
    import openvsp as vsp

    from uavaero.geometry.adapters.openvsp import build_model

    geom_ids = build_model(cases.vertical_tail_case())
    assert vsp.GetParmVal(geom_ids["vtail"], "Sym_Planar_Flag", "Sym") == 0.0
    assert vsp.GetParmVal(geom_ids["vtail"], "X_Rel_Rotation", "XForm") == 90.0
    # and the non-rotated wing keeps its symmetry
    assert vsp.GetParmVal(geom_ids["wing"], "Sym_Planar_Flag", "Sym") != 0.0
