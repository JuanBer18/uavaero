"""VSPAERO's stability run: the derivative set, straight from the solver.

**This corrects an earlier finding in this tree.** Phase 0 probed `VSPAEROSweep`'s inputs, found no
`StabilityType`, and concluded there was no stability mode — so Phase 3 was planned around
finite-differencing alpha and beta sweeps by hand. That was wrong. The input is called
`UnsteadyType`, it takes the `STABILITY_*` enum, and `STABILITY_DEFAULT` runs exactly the analysis
that was assumed absent. It produces the full derivative set, the neutral point and the static
margin in about the time of an ordinary sweep, and the results group is called `VSPAERO_Stab`
after all. The lesson is the one this codebase already had a rule about: an input that is not
where you looked is not an input that does not exist.

**The output is perturbed values, not derivatives.** `Base_Aero_*` is the unperturbed solve;
`Alpha_*`, `Beta_*`, `Roll__Rate_*` and so on are each a second solve with one variable nudged, and
each carries its own `*_Delta`. The derivative is `(perturbed − base) / delta`, per degree for the
angles. Reading those fields as derivatives directly would give numbers about a hundred times too
small, and — this is the trap — they would still look like a plausible aircraft.

Note the field names: `Roll__Rate_` and `Yaw___Rate_` carry padding underscores so every prefix is
the same width. They are not typos and stripping them breaks the lookup.
"""
from __future__ import annotations

import dataclasses

import json
import subprocess
import sys
from pathlib import Path

from uavaero.schema.airframe import ParametricAirframe
from uavaero.schema.condition import OperatingCondition
from uavaero.schema.stability import StabilityResult, per_radian
from uavaero.solvers.base import SolveFailed
from uavaero.solvers.vspaero import _worker_environment

#: VSPAERO's own prefixes. The padding underscores are part of the name.
BASE = "Base_Aero"
ALPHA = "Alpha"
BETA = "Beta"
ROLL_RATE = "Roll__Rate"
PITCH_RATE = "Pitch_Rate"
YAW_RATE = "Yaw___Rate"


def derivative(results: dict[str, list[float]], prefix: str, quantity: str) -> float:
    """(perturbed − base) / delta, per unit of whatever the delta is in.

    Returns 0.0 rather than raising when a field is absent or the delta is zero: a missing rate
    derivative on a configuration that has no surface to produce it is an ordinary state of
    affairs, and zero is the honest value for it.
    """
    base = results.get(f"{BASE}_{quantity}")
    perturbed = results.get(f"{prefix}_{quantity}")
    delta = results.get(f"{prefix}_Delta")
    if not base or not perturbed or not delta or delta[0] == 0:
        return 0.0
    return (perturbed[0] - base[0]) / delta[0]


def parse(results: dict[str, list[float]], condition: OperatingCondition,
          reference: tuple[float, float, float]) -> StabilityResult:
    """Turn one `VSPAERO_Stab` results group into a derivative set, in per-radian units."""
    sref, bref, cref = reference

    def first(name: str, default: float | None = None) -> float | None:
        values = results.get(name)
        return values[0] if values else default

    # Angle perturbations are in degrees; the state-space model downstream wants per radian.
    result = StabilityResult(
        alpha_deg=condition.alpha_start_deg,
        beta_deg=condition.beta_start_deg,
        velocity_mps=condition.velocity_mps,
        x_cg_m=condition.xyz_ref[0],
        sref_m2=sref, bref_m=bref, cref_m=cref,
        CL=first(f"{BASE}_CL", 0.0) or 0.0,
        CD=first(f"{BASE}_CD", 0.0) or 0.0,
        Cm=first(f"{BASE}_CMm", 0.0) or 0.0,
        CL_alpha=per_radian(derivative(results, ALPHA, "CL")),
        CD_alpha=per_radian(derivative(results, ALPHA, "CD")),
        Cm_alpha=per_radian(derivative(results, ALPHA, "CMm")),
        # CS is VSPAERO's side force; CMl and CMn are the rolling and yawing moments.
        CY_beta=per_radian(derivative(results, BETA, "CS")),
        Cl_beta=per_radian(derivative(results, BETA, "CMl")),
        Cn_beta=per_radian(derivative(results, BETA, "CMn")),
        CL_q=derivative(results, PITCH_RATE, "CL"),
        Cm_q=derivative(results, PITCH_RATE, "CMm"),
        Cl_p=derivative(results, ROLL_RATE, "CMl"),
        Cn_p=derivative(results, ROLL_RATE, "CMn"),
        Cl_r=derivative(results, YAW_RATE, "CMl"),
        Cn_r=derivative(results, YAW_RATE, "CMn"),
        neutral_point_x_m=first("X_np"),
        static_margin_reported=first("SM"),
        solver="vspaero",
    )
    complaint = result.consistency()
    if complaint:
        result.notes.append(complaint)

    rates = result.rate_complaint()
    if rates:
        result.notes.append(rates)
        result.rates_validated = False
    return result


#: How far below the fin root the side-force height may legitimately sit before the derivative
#: pair is called impossible. A fuselage carries side force near z = 0 and drags the weighted
#: average down; a quarter of the root height leaves ample room for that and still catches a value
#: that is essentially zero.
FIN_HEIGHT_FLOOR = 0.25


def run_stability(vsp, airframe: ParametricAirframe, condition: OperatingCondition,
                  workdir: Path) -> StabilityResult:
    """Build, mesh and run one stability point. **Only call this inside the worker process.**

    A stability run is a single angle of attack by construction — it perturbs about one operating
    point — so the sweep's alpha range is ignored and `alpha_start_deg` is taken as the point.
    """
    from uavaero.geometry.adapters import openvsp as vsp_adapter
    from uavaero.solvers.vspaero import (
        LOG_FILENAME, reference_values, flow_conditions, scoped_airframe,
    )

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    solved = scoped_airframe(airframe, condition.scope)
    mesh = condition.fidelity.settings
    vsp_adapter.write_vsp3(solved, workdir / "stability.vsp3", mesh.tess_w, mesh.sect_tess_u)

    sref, bref, cref = reference_values(airframe, condition)
    _rho, mach, reynolds = flow_conditions(condition, cref)
    thick_set = vsp.SET_ALL if condition.thick else vsp.SET_NONE
    thin_set = vsp.SET_NONE if condition.thick else vsp.SET_ALL

    vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [thick_set], 0)
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "ThinGeomSet", [thin_set], 0)
    vsp.ExecAnalysis("VSPAEROComputeGeometry")

    vsp.SetAnalysisInputDefaults("VSPAEROSweep")
    vsp.SetIntAnalysisInput("VSPAEROSweep", "GeomSet", [thick_set], 0)
    vsp.SetIntAnalysisInput("VSPAEROSweep", "ThinGeomSet", [thin_set], 0)
    vsp.SetIntAnalysisInput("VSPAEROSweep", "RefFlag", [0], 0)  # never auto-derive
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "Sref", [sref], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "bref", [bref], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "cref", [cref], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "AlphaStart", [condition.alpha_start_deg], 0)
    vsp.SetIntAnalysisInput("VSPAEROSweep", "AlphaNpts", [1], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "BetaStart", [condition.beta_start_deg], 0)
    vsp.SetIntAnalysisInput("VSPAEROSweep", "BetaNpts", [1], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "MachStart", [mach], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "ReCref", [reynolds], 0)
    vsp.SetIntAnalysisInput("VSPAEROSweep", "WakeNumIter", [max(1, condition.wake_iterations)], 0)

    # The moment reference. Every derivative about a moment axis depends on it, and a static
    # margin quoted about the wrong point is a number that looks entirely normal.
    x, y, z = condition.xyz_ref
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "Xcg", [x], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "Ycg", [y], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "Zcg", [z], 0)

    # The whole point. Not `StabilityType` — that input does not exist.
    vsp.SetIntAnalysisInput("VSPAEROSweep", "UnsteadyType", [vsp.STABILITY_DEFAULT], 0)
    vsp.SetStringAnalysisInput("VSPAEROSweep", "RedirectFile", [str(workdir / LOG_FILENAME)], 0)
    vsp.ExecAnalysis("VSPAEROSweep")

    results_id = vsp.FindLatestResultsID("VSPAERO_Stab")
    if not results_id:
        raise RuntimeError(
            "the stability run produced no VSPAERO_Stab results - UnsteadyType may not have taken"
        )
    values: dict[str, list[float]] = {}
    for name in vsp.GetAllDataNames(results_id):
        try:
            found = vsp.GetDoubleResults(results_id, name)
        except Exception:  # noqa: BLE001 - string and int fields are not derivatives
            continue
        if found:
            values[name] = [float(v) for v in found]
    return parse(values, condition, (sref, bref, cref))


def run_isolated(
    airframe: ParametricAirframe,
    condition: OperatingCondition,
    workdir: Path,
    timeout_s: float = 900.0,
    on_output=None,
) -> tuple[StabilityResult | None, str]:
    """Run the stability point in a disposable subprocess. Returns (result, error message).

    Same reasoning as the sweep: OpenVSP's native crashes kill the interpreter with no Python
    exception, so they happen somewhere we can afford to lose.
    """
    workdir = Path(workdir).resolve()  # see the note in vspaero.run_isolated
    workdir.mkdir(parents=True, exist_ok=True)
    job_path = workdir / "job.json"
    result_path = workdir / "result.json"
    job_path.write_text(json.dumps({
        "mode": "stability",
        "spec": airframe.to_dict(),
        "condition": condition.to_dict(),
        "workdir": str(workdir),
        "result_out": str(result_path),
    }), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-u", "-m", "uavaero.solvers.vspaero_worker", str(job_path)],
        capture_output=True, timeout=timeout_s,
        # See the note in vspaero.run_isolated: never the locale default.
        text=True, encoding="utf-8", errors="replace",
        cwd=str(workdir),  # see the note in vspaero.run_isolated
        env=_worker_environment(),
    )
    if on_output is not None:
        for line in (completed.stdout or "").splitlines():
            on_output(line)

    if not result_path.is_file():
        from uavaero.solvers.vspaero import _worker_error, diagnose

        captured = [
            line for line in ((completed.stdout or "") + "\n" + (completed.stderr or "")).splitlines()
            if line.strip()
        ]
        return None, diagnose(
            _worker_error(completed.stderr, "stability"), captured, workdir,
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        from uavaero.solvers.vspaero import diagnose

        captured = [line for line in (completed.stdout or "").splitlines() if line.strip()]
        return None, diagnose(
            payload.get("error") or "the stability run failed", captured, workdir,
        )
    return StabilityResult.from_dict(payload["result"]), ""


def solve(
    airframe: ParametricAirframe,
    condition: OperatingCondition,
    workdir: Path | None = None,
    timeout_s: float = 900.0,
    on_output=None,
    cross_check: bool = True,
) -> StabilityResult:
    """Stability derivatives for this aircraft at this operating point, or raise `SolveFailed`.

    **Statics from VSPAERO, rates from AVL.** That split is not a preference, it is what the
    measurement supports: VSPAERO's rate output fails a damping sign check and does not correspond
    to AVL's by any single scale factor, while AVL's rates are physically sensible throughout. AVL
    also runs the whole derivative set in about a second and a half, so it costs almost nothing to
    take its opinion on the statics too and record where the two codes part company.

    With `cross_check=False`, or with no AVL on the machine, you get VSPAERO's statics and no rate
    derivatives at all — which is the honest outcome, because the alternative is shipping numbers
    that were already shown not to hold up.
    """
    from uavaero.paths import RUNS
    from uavaero.solvers.vspaero import VSPAeroSolver

    blocked = VSPAeroSolver().unavailable_for(airframe, condition)
    if blocked:
        raise SolveFailed(blocked, "vspaero")
    if workdir is None:
        workdir = RUNS / (airframe.name or "airframe") / "stability"

    result, error = run_isolated(airframe, condition, Path(workdir), timeout_s, on_output)
    if result is None:
        raise SolveFailed(error, "vspaero")

    if cross_check:
        _add_independent_rates(result, airframe, condition, on_output)
    else:
        result.notes.append(
            "No cross-check was run, so the rate derivatives are VSPAERO's and are not usable"
        )
    return result


def fin_height_band(airframe: ParametricAirframe, x_cg_z_m: float = 0.0) -> tuple[float, float] | None:
    """How far above the CG the fin reaches, root to tip. None when there is no fin.

    This is the yardstick the lateral derivatives are checked against: whatever makes the side
    force has to be somewhere the aircraft actually is.
    """
    fin = airframe.vtail
    if fin is None or fin.span() <= 0:
        return None
    root = float(fin.origin[2]) - x_cg_z_m
    return root, root + fin.span()


def implausible_dihedral_effect(
    result: StabilityResult, airframe: ParametricAirframe
) -> str:
    """Why this `Cl_beta` cannot be right, or "" if it can.

    `Cl_beta` and `CY_beta` together place the side force at a height (see
    `StabilityResult.side_force_height_m`). Both the fin and any wing dihedral push that height
    *up* - dihedral adds a stabilising roll increment of the same sign - so there is no mechanism
    that puts it below the fin root on an aircraft with a fin above the CG and non-negative
    dihedral. A value far below the root means the rolling moment is too small for the side force
    the same solve reported.

    The tolerance is generous on purpose: a fuselage carries side force near z = 0 and drags the
    weighted average down, so only a clearly impossible value is called one.
    """
    band = fin_height_band(airframe)
    height = result.side_force_height_m
    if band is None or height is None:
        return ""
    root, tip = band
    if root <= 0:
        return ""
    # A fuselage can pull the average down to a fraction of the fin root, but not to nothing.
    floor = FIN_HEIGHT_FLOOR * root
    if height >= floor:
        return ""
    if min(section.dihedral_deg for section in airframe.wing.sections) < -1e-6:
        # Anhedral rolls the other way and legitimately cancels the fin. Not our case to judge.
        return ""
    return (
        f"Cl_beta {result.Cl_beta:+.4f} against CY_beta {result.CY_beta:+.4f} puts the side force "
        f"at z = {height:.4f} m, below the fin root at {root:.3f} m (the fin reaches {tip:.3f} m). "
        f"Both the fin and the wing's dihedral can only push that height up, so the rolling moment "
        f"is too small for the side force this same solve reported"
    )


def _add_independent_rates(
    result: StabilityResult,
    airframe: ParametricAirframe,
    condition: OperatingCondition,
    on_output=None,
) -> None:
    """Run AVL, take its rate derivatives, and record where the two codes disagree."""
    from uavaero.solvers import avl

    if not avl.available():
        result.notes.append(
            "AVL is not installed, so the rate derivatives are VSPAERO's and are not usable - "
            "put avl.exe under external/avl/ or set $UAVAERO_AVL_PATH"
        )
        return

    if on_output is not None:
        on_output("AVL: cross-checking derivatives...")
    try:
        independent = avl.run_derivatives(airframe, condition)
    except Exception as exc:  # noqa: BLE001 - a failed cross-check is a note, not a lost solve
        result.notes.append(f"AVL cross-check did not run: {type(exc).__name__}: {exc}")
        return

    result.cross_check = avl.compare(result, independent)

    # The rates come across wholesale. Mixing one code's Cl_p with another's Cn_r would produce a
    # derivative set that is internally inconsistent in a way nothing downstream could detect.
    replaced = False
    for name in ("CL_q", "Cm_q", "Cl_p", "Cn_p", "Cl_r", "Cn_r", "CY_p", "CY_r"):
        value = independent.get(name)
        if value is not None:
            setattr(result, name, value)
            replaced = True
    if replaced:
        result.rate_source = "avl"
        result.rates_validated = result.rate_complaint() == ""
        result.notes = [n for n in result.notes if "rate derivatives are not usable" not in n]
        if not result.rates_validated:
            result.notes.append(
                "AVL's rate derivatives also fail the damping sign check - do not use them"
            )

    # Cl_beta, when VSPAERO's own two lateral numbers are geometrically impossible together.
    #
    # This is a *rejection*, not a preference. The rule is not "AVL is better at lateral" - on
    # Cn_beta VSPAERO is the one to trust, because it has a fuselage in the solve and AVL is a
    # vortex lattice with no bodies at all. It is that a Cl_beta which places the side force below
    # the fin root cannot be right whatever produced it, and the substitute is only taken when it
    # survives the same check.
    complaint = implausible_dihedral_effect(result, airframe)
    if complaint:
        candidate = independent.get("Cl_beta")
        substitute = dataclasses.replace(result, Cl_beta=candidate) if candidate else None
        if substitute is not None and not implausible_dihedral_effect(substitute, airframe):
            result.notes.append(
                f"VSPAERO's Cl_beta was rejected and AVL's {candidate:+.4f} used instead - "
                f"{complaint}"
            )
            result.Cl_beta = float(candidate)
            result.dihedral_effect_source = "avl"
        else:
            # Nothing usable to fall back on. Say so rather than quietly keeping a value that has
            # already been shown to be impossible.
            result.notes.append(
                f"{complaint} - and AVL offers no usable replacement, so the lateral modes built "
                f"from this are not trustworthy"
            )

    disagreements = result.disagreements()
    if disagreements:
        detail = ", ".join(
            f"{name} {result.cross_check[name]['vspaero']:+.4f} vs {result.cross_check[name]['avl']:+.4f}"
            for name in disagreements
            if name in ("CL_alpha", "Cm_alpha", "CY_beta", "Cl_beta", "Cn_beta")
        )
        if detail:
            result.notes.append(
                f"VSPAERO and AVL disagree on {detail} - both are reported, neither is corrected"
            )
