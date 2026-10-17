"""AVL, kept for one job: an independent opinion on the stability derivatives.

Not a drag solver in this tree and not exposed in the UI. AVL exists here because a second,
completely separate code computing the same derivative set is the only cheap way to tell whether
VSPAERO's are being read correctly — and the rate derivatives specifically need that, because
VSPAERO's raw rate output fails a sign check (`Cn_r` comes back positive, which is an aircraft that
drives its own yaw).

AVL computes derivatives internally in a single run, per radian, in about a second and a half. It
is Drela's code, it has been the reference for this class of aircraft for thirty years, and where
it and VSPAERO agree there is real reason to believe both.

Geometry goes through AeroSandbox's `Airplane` because AVL's input format is fiddly and AeroSandbox
already writes it. That is AeroSandbox as a translator and a process wrapper, not as a solver —
its own VLM is deliberately not used anywhere in this tree.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from uavaero.paths import external_binary
from uavaero.schema.airframe import ParametricAirframe
from uavaero.schema.condition import OperatingCondition

#: AVL's own names for the derivatives, mapped onto this tree's. AVL reports per radian already.
DERIVATIVE_NAMES = {
    "CL_alpha": "CLa", "Cm_alpha": "Cma",
    "CY_beta": "CYb", "Cl_beta": "Clb", "Cn_beta": "Cnb",
    "CY_p": "CYp", "CY_r": "CYr",
    "CL_q": "CLq", "Cm_q": "Cmq",
    "Cl_p": "Clp", "Cn_p": "Cnp",
    "Cl_r": "Clr", "Cn_r": "Cnr",
}


def avl_binary_path() -> str:
    """Environment override, then the bundled copy, then whatever is on PATH."""
    override = os.environ.get("UAVAERO_AVL_PATH")
    if override:
        return override
    bundled = external_binary("avl", "avl.exe")
    if bundled.is_file():
        return str(bundled)
    return "avl"


def available() -> bool:
    exe = avl_binary_path()
    return shutil.which(exe) is not None or os.path.isfile(exe)


@dataclass
class AvlDerivatives:
    """AVL's own derivative set, per radian, plus whatever else it reported."""

    values: dict[str, float]
    raw: dict[str, float]

    def get(self, name: str) -> float | None:
        return self.values.get(name)


def run_derivatives(
    airframe: ParametricAirframe,
    condition: OperatingCondition,
    timeout_s: float = 30.0,
) -> AvlDerivatives:
    """One AVL run at the condition's operating point, remapped onto this tree's names.

    Raises rather than returning a sentinel — a cross-check that silently produced nothing would
    let a disagreement pass as agreement.
    """
    import aerosandbox as asb

    from uavaero.geometry.adapters.aerosandbox import to_airplane

    airplane = to_airplane(airframe)
    # Moments must be taken about the same point as VSPAERO's, or the cross-check compares two
    # different aircraft. Left unset, AeroSandbox defaults the reference to the origin, which on
    # this airframe moves Cm_alpha by a factor of 2.5 and the static margin from 0.18 to 0.42 —
    # a disagreement that looks like a solver problem and is entirely bookkeeping.
    airplane.xyz_ref = list(condition.xyz_ref)
    op_point = asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=condition.altitude_m),
        velocity=condition.velocity_mps,
        alpha=condition.alpha_start_deg,
        beta=condition.beta_start_deg,
    )
    raw = asb.AVL(
        airplane, op_point, avl_command=avl_binary_path(), timeout=timeout_s
    ).run()

    values: dict[str, float] = {}
    for ours, theirs in DERIVATIVE_NAMES.items():
        if theirs in raw:
            values[ours] = float(raw[theirs])
    return AvlDerivatives(values=values, raw={k: v for k, v in raw.items() if isinstance(v, float)})


def compare(vspaero, avl: AvlDerivatives, tolerance: float = 0.25) -> dict[str, dict]:
    """Derivative by derivative, how far apart the two codes are.

    Two independent codes on the same geometry should agree on the static derivatives to well
    inside 25%; where they do not, at least one is being read wrong. Returns a row per derivative
    rather than a verdict, because *which* ones disagree is the whole diagnostic — a uniform
    scale-factor discrepancy on the rates and agreement on the angles says something very specific
    about non-dimensionalisation.
    """
    rows: dict[str, dict] = {}
    for name in DERIVATIVE_NAMES:
        theirs = avl.get(name)
        ours = getattr(vspaero, name, None)
        if theirs is None or ours is None:
            continue
        denominator = abs(theirs) if abs(theirs) > 1e-9 else 1.0
        error = (ours - theirs) / denominator
        rows[name] = {
            "vspaero": ours,
            "avl": theirs,
            "relative_error": error,
            "ratio": (theirs / ours) if abs(ours) > 1e-9 else None,
            "agrees": abs(error) <= tolerance,
        }
    return rows
