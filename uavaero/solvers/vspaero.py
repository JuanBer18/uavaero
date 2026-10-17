"""VSPAERO alpha sweep on a `ParametricAirframe`. Promoted from AEROAPP, not copied from it.

The archived `engine/aero_vsp.py` carried its own `WingParams`/`TailParams` — a second geometry
parametrisation, which is precisely what rule 2 forbids. This takes the canonical spec and builds
through the OpenVSP adapter, so there is exactly one description of the aircraft and the three OpenVSP
meshing fixes apply here for free.

**This module must only ever run inside `vspaero_worker`.** OpenVSP's state is process-global and
its native crashes kill the interpreter with no Python exception (rule 5). `run_isolated` is the
entry point everything outside this file should use.

Three things learned by measurement on OpenVSP 3.51, each of which silently produces wrong or
apparently-degenerate results if you get it wrong:

1. **There is no VLM/Panel switch any more.** No `AnalysisMethod` input, no `vsp.VORTEX_LATTICE`.
   Geometry is assigned to a THIN set (vortex-lattice treatment) or a THICK set (panels) and one
   solve handles both — so a real aircraft is thin lifting surfaces *plus* a thick fuselage rather
   than a choice between two solvers.
2. **The polar is its own result and the lift field is `CLtot`.** `FindLatestResultsID("VSPAERO_Polar")`.
   A field named plain `CL` also exists on the sweep container and returns a scalar `0.0`, which
   reads exactly like the degenerate-mesh failure mode. It is not one.
3. **Reference values must be set explicitly** (`RefFlag = 0`). VSPAERO's own geometry
   auto-derivation was observed in AEROAPP picking 45 m^2 for a 0.16 m^2 wing, which blows up every
   coefficient and diverges the solve.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from uavaero.schema.results import AeroResult, MeshDiagnostics, Polar, ReferenceValues, SpanLoad
from uavaero.schema.airframe import ParametricAirframe
from uavaero.schema.condition import SCOPES, OperatingCondition
from uavaero.solvers.base import SolveFailed
from uavaero.aero.strips import StripIntegration
from uavaero.solvers.parasite import ParasiteResult, run_parasite_drag, sanity_check

#: Dynamic viscosity of air at sea level (Pa s). Only used for the Reynolds number VSPAERO takes as
#: a reference; it is not a solve input in the sense of driving a boundary layer.
MU_SEA_LEVEL = 1.789e-5
SPEED_OF_SOUND = 340.3

DEFAULT_WAKE_ITERATIONS = 5

#: VSPAERO writes its own progress here via the RedirectFile analysis input. Both the worker
#: (which sets it) and run_isolated (which tails it) need the same name.
LOG_FILENAME = "vspaero.log"


@dataclass
class VspAeroResult:
    polar: Polar
    cd_induced: list[float] = field(default_factory=list)
    cd_parasite: list[float] = field(default_factory=list)
    oswald: list[float] = field(default_factory=list)
    lift_to_drag: list[float] = field(default_factory=list)
    mesh: MeshDiagnostics = field(default_factory=MeshDiagnostics)
    span_load: SpanLoad | None = None
    sref_m2: float = 0.0
    bref_m: float = 0.0
    cref_m: float = 0.0
    reynolds: float = 0.0
    mach: float = 0.0
    #: Profile drag per alpha from the 2D section source, integrated over the span load. Empty
    #: when the correction could not run — never silently zero, because zero profile drag is a
    #: number that looks like an answer.
    cd_profile: list[float] = field(default_factory=list)
    #: One span load per alpha, root to tip, folded to one side.
    span_loads: list[SpanLoad] = field(default_factory=list)
    #: Drag Model B: OpenVSP's component parasite buildup. One value for the whole sweep — a
    #: wetted-area estimate has no lift dependence.
    parasite: "ParasiteResult | None" = None
    #: Per-alpha strip detail behind `cd_profile`, and the stations the stall check reads.
    strips: list[Any] = field(default_factory=list)
    #: True when the buildup covered every component *but* the wing, which is what makes
    #: it addable to the strip-integrated wing without double-counting.
    parasite_excludes_wing: bool = False
    #: The buildup over EVERY component including the wing. Not part of the composite -
    #: it is the independent cross-check that gives the comparison an error bar.
    parasite_all: Any = None
    #: Sections the source could not answer for, reported rather than silently dropped.
    section_misses: list[str] = field(default_factory=list)

    # -- the numbers the estimates ledger actually wants ----------------------------------------

    @property
    def cl_alpha_per_rad(self) -> float:
        """Lift-curve slope from a straight-line fit over the linear part of the sweep."""
        alpha = np.radians(np.asarray(self.polar.alpha_deg, dtype=float))
        cl = np.asarray(self.polar.CL, dtype=float)
        if len(alpha) < 2:
            return 0.0
        return float(np.polyfit(alpha, cl, 1)[0])

    @property
    def cd0(self) -> float:
        """Zero-lift drag from a CD = CD0 + k*CL^2 fit across the sweep.

        Fitted rather than read off `CDo`, because `CDo` is VSPAERO's parasite *buildup* and the
        quantity sizing assumed is the intercept of the actual drag polar. Reporting the buildup
        under the name CD0 would compare two different things in the reconciliation table.
        """
        cl = np.asarray(self.polar.CL, dtype=float)
        cd = np.asarray(self.cd_corrected, dtype=float)
        if len(cl) < 3:
            return float(cd[0]) if len(cd) else 0.0
        intercept = float(np.polyfit(cl**2, cd, 1)[1])
        return max(intercept, 0.0)

    #: How negative induced drag is allowed to get before the solve is called unphysical.
    #: Near zero lift a panel solve produces a little numerical noise either side of zero — the
    #: healthy thin solves measure about -0.001 at the bottom of a sweep. A genuinely broken solve
    #: is not close: the case this guard was written for reported -0.14.
    INDUCED_DRAG_NOISE_FLOOR = -0.005

    def implausible(self) -> str:
        """Why this result cannot be believed, or "" if nothing is obviously wrong.

        **A healthy mesh is not the same as a healthy solve.** VSPAERO's thick/panel mode on the
        conventional reference aircraft returns an induced drag of -0.14 while reporting 1116 nodes
        and 28 Kutta nodes — every mesh diagnostic passes and the answer is physically impossible.
        Induced drag is the energy left in the wake; a negative value is a wing that generates
        thrust by producing lift.

        This matters more than most checks because of what consumes it. A human notices a negative
        drag column. An optimiser maximising L/D notices it too, and promotes it.
        """
        cd_induced = [float(v) for v in self.cd_induced]
        if cd_induced and min(cd_induced) < self.INDUCED_DRAG_NOISE_FLOOR:
            return (
                f"induced drag came back negative ({min(cd_induced):.4f}), which is not physically "
                f"possible - the solve converged to something meaningless"
            )
        totals = [float(v) for v in self.polar.CD]
        if totals and min(totals) <= 0.0:
            return f"total drag came back non-positive ({min(totals):.4f})"
        return ""

    @property
    def viscous(self) -> bool:
        return len(self.cd_profile) == len(self.polar.CD) and any(self.cd_profile)

    @property
    def cd_corrected(self) -> list[float]:
        """Induced drag from VSPAERO, profile drag from NeuralFoil.

        Not `CDtot + profile`: that would count the parasite drag twice, because `CDtot` already
        contains VSPAERO's own `CDo`. The point of the correction is to *replace* that buildup —
        a component-count estimate that knows nothing about the boundary layer — with a real 2D
        solve at each strip's own Reynolds number, which at Re ~ 3e5 is a first-order difference
        rather than a refinement. Induced drag is kept from VSPAERO, which is what a panel method
        is actually good at.
        """
        if not self.viscous:
            return [float(v) for v in self.polar.CD]
        return [float(cdi) + float(cdp) for cdi, cdp in zip(self.cd_induced, self.cd_profile)]

    @property
    def lift_to_drag_corrected(self) -> list[float]:
        return [float(cl) / float(cd) if cd > 0 else 0.0
                for cl, cd in zip(self.polar.CL, self.cd_corrected)]

    @property
    def aspect_ratio(self) -> float:
        return self.bref_m**2 / self.sref_m2 if self.sref_m2 > 0 else 0.0

    @property
    def oswald_e(self) -> float:
        """Oswald efficiency, from the same `CD = CD0 + CL^2/(pi AR e)` fit as `cd0`.

        **Deliberately not VSPAERO's own `E` column.** That field returns values well above 1 on a
        healthy solve (3.74 on the conventional reference case), so whatever it measures, it is not
        the Oswald efficiency that sizing assumed and that the reconciliation table compares
        against. `E` is still carried through in `oswald` for reference; it is just not reported
        under a name it does not mean.
        """
        cl = np.asarray(self.polar.CL, dtype=float)
        cd = np.asarray(self.cd_corrected, dtype=float)
        ar = self.aspect_ratio
        if len(cl) < 3 or ar <= 0:
            return 0.0
        k = float(np.polyfit(cl**2, cd, 1)[0])
        if k <= 0:
            return 0.0
        return 1.0 / (np.pi * ar * k)

    @property
    def best_lift_to_drag(self) -> tuple[float, float]:
        """(max L/D, the alpha it happens at), on the corrected polar when there is one."""
        ratios = self.lift_to_drag_corrected if self.viscous else self.lift_to_drag
        if not ratios:
            return 0.0, 0.0
        index = int(np.argmax(ratios))
        return float(ratios[index]), float(self.polar.alpha_deg[index])

    def to_dict(self) -> dict[str, Any]:
        best_ld, best_ld_alpha = self.best_lift_to_drag
        return {
            "polar": self.polar.to_dict(),
            "cd_induced": self.cd_induced,
            "cd_parasite": self.cd_parasite,
            "oswald": self.oswald,
            "lift_to_drag": self.lift_to_drag,
            "mesh": self.mesh.to_dict(),
            "span_load": self.span_load.to_dict() if self.span_load else None,
            "sref_m2": self.sref_m2, "bref_m": self.bref_m, "cref_m": self.cref_m,
            "reynolds": self.reynolds, "mach": self.mach,
            "cd_profile": self.cd_profile,
            "cd_corrected": self.cd_corrected,
            "lift_to_drag_corrected": self.lift_to_drag_corrected,
            "viscous": self.viscous,
            "span_loads": [s.to_dict() for s in self.span_loads],
            # These three cross the worker process boundary as JSON. Omitting them here is not a
            # missing field in a file — it silently loses two of the three drag models between the
            # subprocess that computed them and the parent that reports them.
            "parasite": self.parasite.to_dict() if self.parasite else None,
            "strips": [s.to_dict() for s in self.strips],
            # Must cross the worker boundary or the composite cannot be assembled: the
            # parent has no other way to know the buildup left the wing out.
            "parasite_excludes_wing": self.parasite_excludes_wing,
            "parasite_all": self.parasite_all.to_dict() if self.parasite_all else None,
            "section_misses": list(self.section_misses),
            "aspect_ratio": self.aspect_ratio,
            "derived": {
                "cd0": self.cd0,
                "oswald_e": self.oswald_e,
                "cl_alpha_per_rad": self.cl_alpha_per_rad,
                "best_lift_to_drag": best_ld,
                "best_lift_to_drag_alpha_deg": best_ld_alpha,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VspAeroResult":
        span = payload.get("span_load")
        return cls(
            polar=Polar.from_dict(payload["polar"]),
            cd_induced=payload.get("cd_induced", []),
            cd_parasite=payload.get("cd_parasite", []),
            oswald=payload.get("oswald", []),
            lift_to_drag=payload.get("lift_to_drag", []),
            mesh=MeshDiagnostics(**{
                k: v for k, v in (payload.get("mesh") or {}).items() if k != "ok"}),
            span_load=SpanLoad.from_dict(span) if span else None,
            sref_m2=payload.get("sref_m2", 0.0), bref_m=payload.get("bref_m", 0.0),
            cref_m=payload.get("cref_m", 0.0), reynolds=payload.get("reynolds", 0.0),
            mach=payload.get("mach", 0.0),
            cd_profile=payload.get("cd_profile", []),
            span_loads=[SpanLoad.from_dict(x) for x in payload.get("span_loads") or []],
            parasite=(
                ParasiteResult.from_dict(payload["parasite"]) if payload.get("parasite") else None
            ),
            strips=[StripIntegration.from_dict(x) for x in payload.get("strips") or []],
            parasite_excludes_wing=bool(payload.get("parasite_excludes_wing", False)),
            parasite_all=(
                ParasiteResult.from_dict(payload["parasite_all"])
                if payload.get("parasite_all") else None
            ),
            section_misses=list(payload.get("section_misses") or []),
        )


# ════════ reference values ══════════════════════════════════════════════════════════════════

SCOPES: dict[str, str] = {
    "wing": "Wing only",
    "wing_tail": "Wing + tail",
    "aircraft": "Whole aircraft",
}


def scoped_airframe(airframe: ParametricAirframe, scope: str) -> ParametricAirframe:
    """The airframe reduced to what this scope solves.

    A copy rather than a mutation: the project's aircraft is the one description of the machine
    (rule 2), and a solve scope is a property of the run, not of the aeroplane.
    """
    if scope == "aircraft":
        return airframe
    reduced = ParametricAirframe.from_dict(airframe.to_dict())
    if scope == "wing":
        reduced.htail = None
        reduced.vtail = None
        reduced.fins = []
    reduced.fuselages = []
    return reduced


def missing_for_scope(airframe: ParametricAirframe, scope: str) -> str:
    """Why a scope cannot be run yet, in the user's terms. Empty string when it can.

    These are questions that cannot be asked of this airframe, not solver errors, so they are
    checked before anything expensive starts and answered with a sentence naming what is missing.
    """
    if len(airframe.wing.sections) < 2 or airframe.wing.area() <= 0:
        return "the wing needs at least two sections before it has any span to mesh"
    # Each of these names the scope that *can* be run, because "not available" without a way
    # forward reads as the tool refusing rather than as a question this airframe cannot answer.
    if scope in ("wing_tail", "aircraft") and airframe.htail is None and airframe.vtail is None:
        return (
            "this aircraft has no tail yet - add one on the Airframe screen, or set the scope to "
            "'wing' to solve the wing on its own"
        )
    if scope == "aircraft" and not airframe.fuselages:
        return (
            "this aircraft has no fuselage yet - add one on the Airframe screen, or set the scope "
            "to 'wing_tail' to solve the wing and tails without it"
        )
    return ""


def widest_runnable_scope(airframe: ParametricAirframe, preferred: str = "aircraft") -> str:
    """The most complete scope this airframe can actually be solved at, at or below `preferred`.

    Deleting a component leaves the operating condition pointing at geometry that is gone, and the
    only symptom is a refusal at the moment you press Solve — which reads as the tool being broken
    rather than as a setup that needs one change. Narrowing is *not* silent: the caller says so.

    Only ever narrows. Adding a fuselage back does not widen the scope on its own, because
    `wing_tail` may well be the question you meant to ask.
    """
    order = list(SCOPES)
    if preferred not in order:
        preferred = order[-1]
    for scope in reversed(order[: order.index(preferred) + 1]):
        if not missing_for_scope(airframe, scope):
            return scope
    return order[0]


def reference_values(
    airframe: ParametricAirframe, condition: OperatingCondition
) -> tuple[float, float, float]:
    """Sref / bref / cref, from the wing unless the condition overrides them.

    **`LiftingSurface.span()` is one side; `area()` is already doubled.** That asymmetry is
    documented on both methods and is easy to miss, and missing it here is expensive rather than
    obvious: VSPAERO takes bref as a reference, so a half-span silently quadruples the apparent
    aspect ratio and every quantity derived from it. It was caught by an Oswald efficiency of 3.35,
    which is impossible — had the error been smaller it would have looked like a result.
    """
    wing = airframe.wing
    full_span = wing.span() * (2.0 if wing.symmetric else 1.0)
    sref = condition.sref_m2 if condition.sref_m2 else wing.area()
    bref = condition.bref_m if condition.bref_m else full_span
    cref = condition.cref_m if condition.cref_m else wing.mean_aerodynamic_chord()
    return float(sref), float(bref), float(cref)


def flow_conditions(condition: OperatingCondition, cref_m: float) -> tuple[float, float, float]:
    """(density, Mach, Reynolds at the reference chord).

    Everything comes off the condition's own `Atmosphere` so that the reference Reynolds
    VSPAERO is given and the local Reynolds the strip correction uses cannot drift apart —
    they were two independent hard-coded constants in the previous generation.
    """
    air = condition.atmosphere
    return air.density, air.mach(condition.velocity_mps), air.reynolds(condition.velocity_mps, cref_m)


# ════════ the solve (worker process only) ═══════════════════════════════════════════════════

def run_alpha_sweep(
    airframe: ParametricAirframe, condition: OperatingCondition, workdir: Path
) -> VspAeroResult:
    """Build, mesh and sweep. **Only call this inside `vspaero_worker`.**"""
    import openvsp as vsp

    from uavaero.geometry.adapters import openvsp as vsp_adapter

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    solved_airframe = scoped_airframe(airframe, condition.scope)
    vsp3 = workdir / f"{airframe.name or 'airframe'}.vsp3"
    mesh = condition.fidelity.settings
    # The adapter returns {component name: geom id}. ParasiteDrag needs them so its buildup
    # runs over the real components rather than over SET_ALL, which by then also holds the
    # NGonMesh that VSPAEROComputeGeometry leaves behind.
    geom_ids = vsp_adapter.write_vsp3(solved_airframe, vsp3, mesh.tess_w, mesh.sect_tess_u)

    sref, bref, cref = reference_values(airframe, condition)
    _rho, mach, reynolds = flow_conditions(condition, cref)
    thick_set = vsp.SET_ALL if condition.thick else vsp.SET_NONE
    thin_set = vsp.SET_NONE if condition.thick else vsp.SET_ALL

    vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [thick_set], 0)
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "ThinGeomSet", [thin_set], 0)
    vsp.ExecAnalysis("VSPAEROComputeGeometry")

    log_path = workdir / LOG_FILENAME
    alphas = condition.alphas()

    vsp.SetAnalysisInputDefaults("VSPAEROSweep")
    vsp.SetIntAnalysisInput("VSPAEROSweep", "GeomSet", [thick_set], 0)
    vsp.SetIntAnalysisInput("VSPAEROSweep", "ThinGeomSet", [thin_set], 0)
    vsp.SetIntAnalysisInput("VSPAEROSweep", "RefFlag", [0], 0)  # never auto-derive: see module docs
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "Sref", [sref], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "bref", [bref], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "cref", [cref], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "MachStart", [mach], 0)
    vsp.SetIntAnalysisInput("VSPAEROSweep", "MachNpts", [1], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "ReCref", [reynolds], 0)
    vsp.SetIntAnalysisInput("VSPAEROSweep", "ReCrefNpts", [1], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "AlphaStart", [float(alphas[0])], 0)
    vsp.SetDoubleAnalysisInput("VSPAEROSweep", "AlphaEnd", [float(alphas[-1])], 0)
    vsp.SetIntAnalysisInput("VSPAEROSweep", "AlphaNpts", [len(alphas)], 0)
    vsp.SetIntAnalysisInput("VSPAEROSweep", "WakeNumIter", [max(1, condition.wake_iterations)], 0)
    vsp.SetStringAnalysisInput("VSPAEROSweep", "RedirectFile", [str(log_path)], 0)
    vsp.ExecAnalysis("VSPAEROSweep")

    polar_id = vsp.FindLatestResultsID("VSPAERO_Polar")
    if not polar_id:
        raise RuntimeError("VSPAERO produced no polar result")

    def column(name: str) -> list[float]:
        return [float(v) for v in vsp.GetDoubleResults(polar_id, name)]

    alpha = column("Alpha")
    cl = column("CLtot")          # NOT "CL": that one is a scalar 0.0. See module docstring.
    cd = column("CDtot")
    cm = column("CMytot")
    result = VspAeroResult(
        polar=Polar(
            alpha_deg=np.asarray(alpha), CL=np.asarray(cl), CD=np.asarray(cd),
            CM=np.asarray(cm), source="vspaero",
            meta={"thick": condition.thick, "wake_iterations": condition.wake_iterations,
          "fidelity": condition.fidelity.value, "scope": condition.scope},
        ),
        cd_induced=column("CDi"),
        cd_parasite=column("CDo"),
        oswald=column("E"),
        lift_to_drag=column("L_D"),
        mesh=parse_mesh_diagnostics(
            log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""),
        span_load=extract_span_load(vsp, sref, bref / 2.0, alpha),
        span_loads=extract_span_loads(vsp, sref, bref / 2.0, alpha),
        sref_m2=sref, bref_m=bref, cref_m=cref, reynolds=reynolds, mach=mach,
    )
    result.cd_profile, result.strips, result.section_misses = profile_drag_per_alpha(
        solved_airframe, result, condition
    )

    # The component buildup, off the geometry already standing in this process. Cheap next to the
    # sweep, and running it here rather than in a second worker avoids rebuilding the model.
    #
    # **Everything except the wing.** The wing's viscous drag comes from 2D section polars at each
    # strip's own Reynolds number, which is the method that can see a laminar separation bubble; a
    # flat-plate estimate cannot, and at model-scale Reynolds that is the difference between a
    # section at L/D 27 and the same section at L/D 6. The buildup keeps the fuselage, the booms
    # and the tails, where a wetted-area estimate is the right tool and where the strip integrator
    # has no airfoil to look up. Counting the wing in both would double it.
    # By the wing's own name, never the literal "wing" - an airframe may name it anything.
    wing_name = solved_airframe.wing.name
    everything = list(geom_ids.values())
    others = [gid for name, gid in geom_ids.items() if name != wing_name]
    result.parasite_excludes_wing = bool(others) and len(others) != len(everything)

    def buildup(ids: list[str], label: str):
        """One ParasiteDrag pass, or None with a note saying why not."""
        try:
            value = run_parasite_drag(condition, sref, geom_ids=ids)
        except Exception as exc:  # noqa: BLE001 - one model failing must not lose the others
            result.section_misses.append(f"{label}: {type(exc).__name__}: {exc}")
            return None
        complaint = sanity_check(value, condition, cref)
        if complaint:
            result.section_misses.append(f"{label}: {complaint}")
            return None
        return value

    # Run twice, over different component sets, because they answer different questions.
    #
    # `parasite` covers the BODY ONLY and is the half of the composite the strip integrator cannot
    # do. `parasite_all` covers EVERYTHING including the wing and is not part of the composite at
    # all - it is the independent cross-check, the same total arrived at by a method that is blind
    # to airfoil and to lift. Two methods that disagree are an error bar; one number on its own is
    # a confidence nobody has, which is what a single composite would report.
    #
    # The second pass is nearly free: the geometry is already built and standing in this process.
    result.parasite = buildup(others or everything, "parasite buildup")
    if result.parasite_excludes_wing:
        result.parasite_all = buildup(everything, "parasite cross-check")

    return result


def section_lookup(condition: OperatingCondition, mach: float):
    """The 2D section source the condition asked for, wrapped in its cache."""
    from uavaero.solvers.sections.base import SectionLookup
    from uavaero.solvers.sections.neuralfoil import NeuralFoilSource
    from uavaero.solvers.sections.xfoil import XFoilSource

    source = XFoilSource() if condition.section_source == "xfoil" else NeuralFoilSource()
    return SectionLookup(source, mach=mach)


def wing_sheets(load: "SpanLoad", airframe: ParametricAirframe) -> set[int]:
    """Which of VSPAERO's vortex sheets belong to the main wing.

    Sheet numbering is not a clean map onto geometry order - measured on the reference aircraft,
    the wing came back as one sheet carrying both halves while the horizontal tail split into two.
    So each sheet is matched to a surface by what it actually spans rather than by its index.

    The discriminator is span extent, taken along y for a horizontal surface and z for a fin, and
    compared with each surface's own semi-span. Strips sit at panel centres, so a sheet reaches
    about 90% of the surface it belongs to - which is unambiguous when the candidates differ by
    more than that, and this returns nothing rather than guessing when they do not.

    An empty result means "no tag recorded", and the caller integrates everything, which is the
    behaviour that existed before the tag was read.
    """
    if not load.sheet or len(load.sheet) != len(load.y_m):
        return set()

    surfaces = [(s, s.span()) for s in airframe.lifting_surfaces() if s.span() > 0]
    if not surfaces:
        return set()

    chosen: set[int] = set()
    for sheet in sorted(set(load.sheet)):
        reach = max(
            (abs(y) for y, sh in zip(load.y_m, load.sheet) if sh == sheet), default=0.0
        )
        # Nearest surface by how far its strips reach. A fin's strips are all at y = 0 and so
        # reach nothing in y, which is itself the signature that separates it from a tailplane.
        best = min(surfaces, key=lambda item: abs(reach - item[1]))
        if best[0] is airframe.wing:
            chosen.add(sheet)
    return chosen


def profile_drag_per_alpha(
    airframe: ParametricAirframe, result: "VspAeroResult", condition: OperatingCondition
) -> tuple[list[float], list[Any], list[str]]:
    """Profile drag at every alpha, from the solver's own span loads. Drag Model C.

    Runs as part of every solve rather than as an opt-in: VSPAERO has no 2D mode and its `CDo` is a
    component buildup, so an uncorrected polar is not a cheaper answer — it is a different and, at
    UAS Reynolds numbers, a worse one. With NeuralFoil it costs about a tenth of a second, because
    the polars are cached per (airfoil, Reynolds bucket).

    Returns ([], [], reasons) rather than zeros if it cannot run: zero profile drag is a number
    that looks like an answer, and the caller reports the absence instead.
    """
    from uavaero.aero.strips import integrate_profile_drag

    if not result.span_loads or result.sref_m2 <= 0:
        return [], [], ["no span load to integrate"]
    air = condition.atmosphere
    lookup = section_lookup(condition, result.mach)
    values: list[float] = []
    integrations: list[Any] = []
    # The wing only. Every other surface is covered by the component buildup, which is the method
    # that suits a body or a thin symmetric tail - and the section lookup has no way to know a
    # strip is a tail, so it would hand it the wing's airfoil.
    sheets = wing_sheets(result.span_loads[0], airframe)
    for full in result.span_loads:
        load = full.stations_on(sheets) if sheets else full
        try:
            integration = integrate_profile_drag(
                airframe, load.y_m, load.chord_m, load.cl_local,
                sref_m2=result.sref_m2, density=air.density, mu=air.viscosity,
                velocity_mps=condition.velocity_mps, lookup=lookup,
            )
        except Exception as exc:  # noqa: BLE001 - a failed correction must not lose the solve
            return [], [], [f"{type(exc).__name__}: {exc}"]
        values.append(integration.cd_profile)
        integrations.append(integration)
    if len(values) != len(result.polar.CD):
        return [], [], ["the correction did not cover every angle"]
    return values, integrations, list(dict.fromkeys(lookup.misses))


def parse_mesh_diagnostics(log_text: str) -> MeshDiagnostics:
    """Out of VSPAERO's own solve log — the same text it always prints, captured via RedirectFile
    rather than left to vanish into the worker's stdout."""
    def find_int(pattern: str) -> int | None:
        match = re.search(pattern, log_text)
        return int(match.group(1)) if match else None

    return MeshDiagnostics(
        nodes=find_int(r"Total NumberOfNodes:\s+(\d+)"),
        loops=find_int(r"Total NumberOfLoops:\s+(\d+)"),
        kutta_nodes=find_int(r"Total NumberOfKuttaNodes:\s+(\d+)"),
        surfaces=find_int(r"NumberOfSurfaces_:\s+(\d+)"),
        unresolved_wake_loops=
            find_int(r"There were (\d+) loops that we could not find upwind edges for") or 0,
    )


def extract_span_loads(vsp, area_m2: float, half_span_m: float, alphas: list[float]) -> list[SpanLoad]:
    """Every alpha's strip data, in sweep order.

    The viscous correction needs a span load per point, not just the last one: profile drag varies
    across the polar because the local section CLs do. `VSPAERO_Load` holds one result per alpha
    already — it was simply never all read.
    """
    count = vsp.GetNumResults("VSPAERO_Load")
    loads: list[SpanLoad] = []
    for index in range(count):
        alpha = float(alphas[index]) if index < len(alphas) else 0.0
        load = _span_load_at(vsp, index, area_m2, half_span_m, alpha)
        if load is not None:
            loads.append(load)
    return loads


def extract_span_load(vsp, area_m2: float, half_span_m: float, alphas: list[float]) -> SpanLoad | None:
    """The spanwise strip data VSPAERO already computes and nothing here ever read.

    One `VSPAERO_Load` result per alpha point, in sweep order. The group's own `Angle` field is
    unreliable (observed always 0), so the index is trusted and the last one taken — the highest
    alpha, which is the load case Structures cares about.
    """
    count = vsp.GetNumResults("VSPAERO_Load")
    if not count:
        return None
    return _span_load_at(
        vsp, count - 1, area_m2, half_span_m, float(alphas[-1]) if alphas else 0.0)


def _span_load_at(
    vsp, index: int, area_m2: float, half_span_m: float, alpha_deg: float
) -> SpanLoad | None:
    load_id = vsp.FindResultsID("VSPAERO_Load", index)
    y_positions = [float(v) for v in vsp.GetDoubleResults(load_id, "Yavg")]
    chords = [float(v) for v in vsp.GetDoubleResults(load_id, "Chord")]
    cls_local = [float(v) for v in vsp.GetDoubleResults(load_id, "cl")]
    # Which surface each strip belongs to. An INT column - reading it as a double returns an empty
    # list and silently loses the tag, which is how it went unnoticed until now.
    sheets = [int(v) for v in vsp.GetIntResults(load_id, "VortexSheet")]
    if not y_positions:
        return None
    if len(sheets) != len(y_positions):
        sheets = [0] * len(y_positions)

    # Both halves are reported; fold them together so the distribution is one-sided and monotonic.
    #
    # **Folded within a surface, not across surfaces.** A vertical fin has every strip at y = 0, so
    # a fold keyed on |y| alone collapses the whole fin into one averaged station - and a
    # horizontal tail sits inside the wing's span rather than beyond it, so the two interleave.
    # Keying on (sheet, |y|) keeps each surface's own distribution intact.
    folded: dict[tuple[int, float], list[tuple[float, float]]] = {}
    for y, chord, cl, sheet in zip(y_positions, chords, cls_local, sheets):
        folded.setdefault((sheet, round(abs(y), 6)), []).append((chord, cl))
    stations = sorted(
        (y, float(np.mean([c for c, _ in v])), float(np.mean([l for _, l in v])), sheet)
        for (sheet, y), v in folded.items()
    )

    ys = [s[0] for s in stations]
    chord_m = [s[1] for s in stations]
    cl_local = [s[2] for s in stations]
    sheet_ids = [s[3] for s in stations]

    bending = None
    if len(stations) >= 2 and area_m2 > 0 and half_span_m > 0:
        loads = np.asarray(cl_local) * np.asarray(chord_m)
        bending = float(np.trapezoid(loads * np.asarray(ys), ys) / (area_m2 * half_span_m))

    return SpanLoad(
        alpha_deg=alpha_deg,
        y_m=ys, chord_m=chord_m, cl_local=cl_local,
        root_bending_moment_coefficient=bending,
        sheet=sheet_ids,
    )


# ════════ the entry point everything else uses ══════════════════════════════════════════════

#: VSPAERO prints one of these per operating point it starts. Counting them is the only honest
#: progress signal available: the wake iterations inside a point are not uniform in cost, and the
#: solver reports no percentage of its own.
#: VSPAERO announces its own case count. Exact, and better than counting "Solving..." lines,
#: which drifts whenever the case count is not the alpha count - a stability run perturbs
#: about one angle and still reports seven cases.
_CASE_LINE = re.compile(r"case:\s*(\d+)\s*of\s*(\d+)")
_SOLVING_LINE = re.compile(r"Solving\.\.\..*Alpha:\s*(-?[\d.]+)")
_WAKE_LINE = re.compile(r"Wake Iter:\s*(\d+)\s*/\s*(\d+)")


def parse_progress(line: str, alpha_points: int) -> tuple[float, str] | None:
    """(fraction complete, human label) for one line of solver output, or None if it says nothing.

    Kept in the kernel rather than the screen so the CLI can show the same thing, and so the
    regexes sit next to the module that knows what VSPAERO prints.
    """
    match = _CASE_LINE.search(line)
    if match:
        done, total = int(match.group(1)), max(int(match.group(2)), 1)
        return done / total, f"case {done} of {total}"

    match = _SOLVING_LINE.search(line)
    if match:
        parse_progress.seen = getattr(parse_progress, "seen", 0) + 1  # type: ignore[attr-defined]
        done = min(parse_progress.seen, max(alpha_points, 1))  # type: ignore[attr-defined]
        return done / max(alpha_points, 1), f"alpha {match.group(1)}°  ({done}/{alpha_points})"
    match = _WAKE_LINE.search(line)
    if match:
        return -1.0, f"wake iteration {match.group(1)}/{match.group(2)}"
    return None


def run_isolated(
    airframe: ParametricAirframe,
    condition: OperatingCondition,
    workdir: Path,
    timeout_s: float = 900.0,
    on_output: "Callable[[str], None] | None" = None,
) -> tuple[VspAeroResult | None, str]:
    """Run the sweep in a disposable subprocess. Returns (result, error message).

    A VSPAERO native crash is a non-zero exit code here and nothing more — which is the entire
    reason this indirection exists (rule 5). Never raises for a solver failure: a failed solve is a
    finding the module reports, not an exception the UI has to catch.

    `on_output` receives the worker's stdout **line by line as it arrives**. VSPAERO is chatty and
    slow in a way that looks like a hang, so the caller needs to see it happening rather than a
    frozen panel and a result thirty seconds later.
    """
    import json
    import subprocess
    import sys

    # Resolved, not merely constructed. The worker runs with its own working directory, so a
    # relative path stops meaning what it meant the moment the child starts - which shows up as
    # the worker dying on the job file it was handed rather than as a path error.
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    job_path = workdir / "job.json"
    result_path = workdir / "result.json"
    job_path.write_text(json.dumps({
        "spec": airframe.to_dict(),
        "condition": condition.to_dict(),
        "workdir": str(workdir),
        "result_out": str(result_path),
    }), encoding="utf-8")

    # Reset the per-run progress counter that `parse_progress` accumulates.
    parse_progress.seen = 0  # type: ignore[attr-defined]
    tail_lines: list[str] = []

    # VSPAERO's own output never reaches our pipe: the solver is native code writing to the
    # RedirectFile it was given, so the worker's stdout carries only the worker's own two Python
    # prints (measured: 2 lines in a 17 s solve). The progress that makes the wait bearable is in
    # that file, so it gets tailed here while the solve runs.
    log_path = workdir / LOG_FILENAME
    log_path.unlink(missing_ok=True)
    stop_tailing = threading.Event()

    def tail_log() -> None:
        position = 0
        while not stop_tailing.is_set():
            try:
                if log_path.is_file():
                    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                        handle.seek(position)
                        chunk = handle.read()
                        position = handle.tell()
                    for entry in chunk.splitlines():
                        entry = entry.strip()
                        if entry and on_output is not None:
                            on_output(entry)
            except OSError:
                pass  # the solver holds the file open; try again on the next tick
            stop_tailing.wait(0.15)

    tailer = threading.Thread(target=tail_log, daemon=True) if on_output else None
    if tailer is not None:
        tailer.start()

    try:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "uavaero.solvers.vspaero_worker", str(job_path)],
            # Its own directory. VSPAERO is native code that writes scratch files relative to the
            # working directory, so parallel workers sharing one collide - which presents as an
            # OpenVSP process dying with no log and no result, not as an error message.
            cwd=str(workdir),
            env=_worker_environment(),
            # UTF-8 with replacement, never the locale default. `text=True` alone decodes
            # with the preferred encoding - cp1252 on this machine - which has undefined byte
            # slots that RAISE on decode. One stray byte in a solver's output would then kill
            # the read loop and report the solve as failed for entirely the wrong reason: the
            # plumbing masking the very error it was supposed to be carrying.
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            text=True, encoding="utf-8", errors="replace",
        )
        deadline = time.monotonic() + timeout_s
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                tail_lines.append(line)
                if on_output is not None:
                    on_output(line)
            if time.monotonic() > deadline:
                process.kill()
                return None, f"VSPAERO did not finish within {timeout_s:.0f} s."
        returncode = process.wait()
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not start the VSPAERO worker: {type(exc).__name__}: {exc}"
    finally:
        stop_tailing.set()
        if tailer is not None:
            tailer.join(timeout=2.0)
            tail_log()  # one last pass, so the final lines are not lost to the poll interval

    if not result_path.is_file():
        return None, diagnose(
            f"The VSPAERO worker exited {returncode} without writing a result.",
            tail_lines, workdir,
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        return None, diagnose(
            payload.get("error") or "VSPAERO failed for an unrecorded reason.",
            tail_lines, workdir,
        )
    return VspAeroResult.from_dict(payload["result"]), ""


def diagnose(headline: str, captured: list[str], workdir: Path) -> str:
    """A failure message with the evidence attached.

    The old version reported the last three captured lines. That is almost always the wrong three:
    a VSPAERO failure announces itself part-way up its own output and then keeps printing, so the
    tail is the aftermath rather than the cause. The whole capture goes in, and the run directory
    is named because `job.json`, `vspaero.log` and the `.vsp3` in it are the actual evidence — the
    message is a summary of them.

    Returned as one string with newlines so it survives the worker boundary and the `SolveFailed`
    that carries it; the console splits it back out.
    """
    parts = [headline]
    log_path = Path(workdir) / LOG_FILENAME
    if log_path.is_file():
        try:
            log_lines = [
                line.rstrip()
                for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
        except OSError:
            log_lines = []
        if log_lines:
            parts.append(f"--- {LOG_FILENAME} (last 40 of {len(log_lines)}) ---")
            parts.extend(log_lines[-40:])
    if captured:
        parts.append(f"--- worker output ({len(captured)} lines) ---")
        parts.extend(captured[-40:])
    parts.append(f"--- run directory ---\n{workdir}")
    return "\n".join(parts)


# ════════ the AeroSolver implementation ═════════════════════════════════════════════════════

class VSPAeroSolver:
    """VSPAERO behind the common solver interface.

    Everything expensive still happens in `run_isolated`'s disposable subprocess. This class is the
    translation layer: it turns `VspAeroResult` — which is VSPAERO-shaped and knows about Kutta
    nodes and wake iterations — into the canonical `AeroResult` that the rest of the tool reads,
    and in doing so it registers the first two entries of the drag comparison.

    Those two are genuinely independent estimates of the same quantity and they routinely disagree
    by tens of percent at UAS Reynolds numbers, which is exactly why both are kept.
    """

    name = "vspaero"

    def __init__(self, timeout_s: float = 900.0, runs_dir: Path | None = None) -> None:
        self.timeout_s = timeout_s
        self._runs_dir = runs_dir

    def unavailable_for(
        self, airframe: ParametricAirframe, condition: OperatingCondition
    ) -> str:
        missing = missing_for_scope(airframe, condition.scope)
        if missing:
            return f"the {condition.scope} scope is not available: {missing}"
        return ""

    def evaluate(
        self,
        airframe: ParametricAirframe,
        condition: OperatingCondition,
        *,
        workdir: Path | None = None,
        on_output: "Callable[[str], None] | None" = None,
    ) -> "AeroResult":
        from uavaero.schema.results import AeroResult, DragBreakdown, DragModel

        blocked = self.unavailable_for(airframe, condition)
        if blocked:
            raise SolveFailed(blocked, self.name)

        if workdir is None:
            from uavaero.paths import RUNS
            workdir = (self._runs_dir or RUNS) / (airframe.name or "airframe") / self.name

        started = time.time()
        native, error = run_isolated(
            airframe, condition, Path(workdir), timeout_s=self.timeout_s, on_output=on_output,
        )
        elapsed = time.time() - started
        if native is None:
            raise SolveFailed(error or "VSPAERO produced no result", self.name)

        drag = DragBreakdown(cd_induced=[float(v) for v in native.cd_induced])
        notes: list[str] = list(native.section_misses)

        # THE model: each surface estimated by the method that suits it.
        #
        #   CD = CD_induced (the 3D solve)
        #      + CD_profile (the WING, from 2D section polars strip by strip)
        #      + CD_parasite (EVERY OTHER component, from the wetted-area buildup)
        #
        # The split is not a compromise between two methods, it is each one used where it is
        # right. A flat-plate skin-friction estimate cannot see a laminar separation bubble, and
        # on a model-scale wing that is the difference between a section at L/D 27 and the same
        # section at L/D 6 - so the wing gets real section data at each strip's own Reynolds. A
        # fuselage has no airfoil to look up and no meaningful section polar, so it gets the
        # buildup, which is exactly what that method is good at.
        parasite = native.parasite
        wing_viscous = native.viscous
        if wing_viscous and parasite is not None and native.parasite_excludes_wing:
            body = parasite.cd_parasite
            missing = sum(getattr(s, "missing", 0) for s in native.strips)
            drag.add(DragModel(
                key=f"composite_{condition.section_source}",
                label=f"Composite ({condition.section_source} wing + buildup body)",
                cd=[float(v) + body for v in native.cd_corrected],
                basis=(
                    f"Induced drag from the 3D solve; the wing's profile drag from "
                    f"{condition.section_source}, integrated strip by strip at each strip's own "
                    f"Reynolds number and local lift coefficient; every other component "
                    f"({', '.join(c.name for c in parasite.components)}) from OpenVSP's wetted-area "
                    f"buildup, {body:.5f} over {parasite.wetted_area_m2:.3f} m2. The wing appears "
                    f"in exactly one of those two, so nothing is double-counted."
                ),
            ))
            if missing:
                notes.append(
                    f"{missing} wing strip(s) had no usable section data - the composite drag is "
                    f"an under-estimate over the strips that did solve"
                )
        elif wing_viscous:
            # The buildup could not be restricted to the non-wing components, so adding it would
            # count the wing twice. Report the strip correction alone and say what is missing.
            drag.add(DragModel(
                key=f"strip_{condition.section_source}",
                label=f"Strip-corrected wing ({condition.section_source})",
                cd=[float(v) for v in native.cd_corrected],
                basis=(
                    f"Induced drag from the 3D solve plus the WING's profile drag from "
                    f"{condition.section_source}, strip by strip. The fuselage and tails are NOT "
                    f"included - the component buildup that would cover them could not be run "
                    f"over the non-wing components alone."
                ),
            ))
            notes.append(
                "the body's parasite drag is missing from this result - it is a wing-only figure"
            )
        else:
            notes.append("Strip correction could not run - no airfoil-specific drag in this result")

        # The cross-check, ALWAYS alongside the composite rather than only when it fails. An
        # earlier cut of this offered the comparison only when the composite could not be built,
        # which is exactly backwards: it gave a second opinion precisely when there was no first
        # one, and reported a bare 0.0% spread on every healthy solve.
        cross = native.parasite_all if native.parasite_excludes_wing else parasite
        if cross is not None:
            parasite = cross
            drag.add(DragModel(
                key="parasite_buildup",
                label="Parasite buildup (all components)",
                cd=[float(cdi) + parasite.cd_parasite for cdi in native.cd_induced],
                basis=(
                    f"Induced drag plus OpenVSP's component buildup over every component "
                    f"({parasite.cd_parasite:.5f} over {parasite.wetted_area_m2:.3f} m2 wetted, "
                    f"{parasite.turbulent_equation}). A flat-plate skin-friction estimate per "
                    f"component with a form factor for thickness - blind to both airfoil and lift, "
                    f"and therefore an independent check on the composite rather than a rival."
                ),
            ))
        else:
            notes.append(
                "the component buildup did not run - this result has no independent cross-check"
            )

        stall_payload, stall_notes = self._stall(native, condition)

        return AeroResult(
            polar=native.polar,
            drag=drag,
            span_loads=list(native.span_loads),
            reference=ReferenceValues(native.sref_m2, native.bref_m, native.cref_m),
            mesh=native.mesh,
            reynolds=native.reynolds,
            mach=native.mach,
            solver=self.name,
            fidelity=condition.fidelity.value,
            scope=condition.scope,
            elapsed_s=elapsed,
            # The composite when it could be assembled, the wing-only strip figure when the
            # buildup could not be restricted to the body, and the buildup when there is no
            # section data at all. Never a model that is not in `drag.models`.
            primary_model=next(
                (k for k in (
                    f"composite_{condition.section_source}",
                    f"strip_{condition.section_source}",
                    "parasite_buildup",
                ) if k in drag.models),
                "",
            ),
            notes=notes + stall_notes,
            native={
                "oswald_column": [float(v) for v in native.oswald],
                "cd_parasite_column": [float(v) for v in native.cd_parasite],
                "wake_iterations": condition.wake_iterations,
                "thick": condition.thick,
                "section_source": condition.section_source,
                "parasite": native.parasite.to_dict() if native.parasite else None,
                "stall": stall_payload,
            },
        )

    @staticmethod
    def _stall(native: "VspAeroResult", condition: OperatingCondition):
        """Stall margins at the highest angle solved, which is where the wing runs out first.

        A panel method reports a local lift coefficient of 1.9 on a section that tops out at 1.2,
        and the polar stays perfectly smooth through it. This is the only thing in the solve that
        notices, which is why it runs on every solve rather than on request.
        """
        if not native.strips:
            return None, []
        stations = getattr(native.strips[-1], "stations", [])
        if not stations:
            return None, []

        from uavaero.aero import stall as stall_module

        alpha = float(native.polar.alpha_deg[-1]) if len(native.polar.alpha_deg) else 0.0
        report = stall_module.check(stations, section_lookup(condition, native.mach), alpha_deg=alpha)
        payload = {
            "alpha_deg": report.alpha_deg,
            "ok": report.ok,
            "summary": report.summary(),
            "stations": [
                {
                    "y_m": s.y_m, "span_fraction": s.span_fraction, "cl_local": s.cl_local,
                    "cl_max": s.cl_max, "reynolds": s.reynolds, "airfoil": s.airfoil,
                    "margin": s.margin,
                }
                for s in report.stations
            ],
        }
        notes = [] if report.ok else [f"At {alpha:g} deg: {report.summary()}"]
        return payload, notes


def _worker_environment() -> dict:
    """The environment a worker subprocess needs when it is not started from the app root.

    Workers run with `cwd` set to their own directory so that native scratch files cannot collide
    between parallel solves. That breaks `-m uavaero...`, because the module is found through the
    working directory — so the application root goes on PYTHONPATH explicitly. Fixing the collision
    without this created a second bug that looked exactly like the first: a worker that died
    immediately, with no log and no result.
    """
    import os

    from uavaero.paths import APP_ROOT

    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    root = str(APP_ROOT)
    if root not in existing.split(os.pathsep):
        environment["PYTHONPATH"] = os.pathsep.join(p for p in (root, existing) if p)
    return environment


def _worker_error(stderr: str | None, label: str) -> str:
    """The useful line out of a worker's stderr.

    A traceback's *last* line names the failure; its first names `io.open` or whatever the
    outermost frame happened to be. Reporting the head of it turned a path bug into fifteen
    identical messages reading "return io.open(self" and cost real time to see through.
    """
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    if not lines:
        return f"the {label} worker produced no result and said nothing"
    tail = lines[-1]
    if len(lines) > 1 and ("Error" in tail or "Exception" in tail):
        return f"{tail}  (in the {label} worker)"
    return f"{tail}  (in the {label} worker)"
