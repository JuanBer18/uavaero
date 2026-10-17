"""The speed sweep: drag and power against airspeed, in steady level flight.

An alpha sweep answers "what does this wing do at 6 degrees". It is the wrong question for most
of what you actually want to know — the best cruise speed, how much power the motor has to hold,
where the aircraft stops flying. Those are all questions about *speed*, and at each speed the
incidence is not free: lift equals weight, so the aircraft flies at whatever CL that demands.

**One solve covers the whole range.** VSPAERO's treatment here is incompressible and inviscid, so
its answer depends on the shape and the lift coefficient and not at all on Reynolds number. The
induced drag at CL = 0.4 is the same whether that CL is reached at 12 m/s or 30. What *does* change
with speed is the viscous half — the boundary layer cares about Reynolds a great deal, and at model
scale it cares enormously (a section's L/D can fall threefold between Re 40,000 and Re 24,000).

So the sweep reuses one alpha sweep's inviscid solution and re-integrates the strips at each
speed's own Reynolds number. That is not a shortcut around a solve, it is the physically correct
factorisation, and it is what makes a speed sweep cost one solve instead of twenty.

**Where it stops being honest, it says so.** Below stall the required CL exceeds what the wing can
reach and the point is dropped, not extrapolated. Where the strip integrator loses stations to
missing section data the point carries that count, because a speed whose drag is missing a third of
its viscous contribution is the speed that will look best.

Mass is stated, never sized (rule 18) — a speed sweep divides by weight at every point, and saying
what your aircraft weighs is the same kind of act as saying how fast it is flying.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from uavaero.aero.strips import integrate_profile_drag
from uavaero.schema.airframe import ParametricAirframe
from uavaero.schema.condition import Atmosphere
from uavaero.schema.results import AeroResult
from uavaero.solvers.sections.base import SectionLookup

GRAVITY = 9.80665


@dataclass
class SpeedPoint:
    """One airspeed, trimmed for level flight."""

    velocity_mps: float
    cl_required: float
    alpha_deg: float
    reynolds: float
    cd_induced: float
    cd_profile: float
    cd_total: float
    #: Strips the section source could not answer for. Non-zero means `cd_profile` is a floor.
    missing_strips: int = 0

    @property
    def lift_to_drag(self) -> float:
        return self.cl_required / self.cd_total if self.cd_total > 0 else 0.0

    def drag_newtons(self, density: float, sref_m2: float) -> float:
        return 0.5 * density * self.velocity_mps**2 * sref_m2 * self.cd_total

    def power_watts(self, density: float, sref_m2: float) -> float:
        """Power required to hold level flight: drag times speed. Propulsive, not shaft — no
        propeller or motor efficiency is assumed, because none is known."""
        return self.drag_newtons(density, sref_m2) * self.velocity_mps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpeedPoint":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class SpeedSweep:
    """Level-flight performance across a speed range, and the speeds worth knowing."""

    points: list[SpeedPoint] = field(default_factory=list)
    mass_kg: float = 0.0
    sref_m2: float = 0.0
    altitude_m: float = 0.0
    section_source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def density(self) -> float:
        return Atmosphere(altitude_m=self.altitude_m).density

    def best_range(self) -> SpeedPoint | None:
        """Maximum L/D — the speed that covers the most ground per unit of energy."""
        return max(self.points, key=lambda p: p.lift_to_drag, default=None)

    def best_endurance(self) -> SpeedPoint | None:
        """Minimum power required — the speed that stays up longest. Always slower than best
        range, and confusing the two is how an endurance aircraft ends up cruising too fast."""
        if not self.points:
            return None
        return min(self.points, key=lambda p: p.power_watts(self.density, self.sref_m2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "mass_kg": self.mass_kg, "sref_m2": self.sref_m2,
            "altitude_m": self.altitude_m, "section_source": self.section_source,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpeedSweep":
        return cls(
            points=[SpeedPoint.from_dict(p) for p in payload.get("points") or []],
            mass_kg=payload.get("mass_kg", 0.0),
            sref_m2=payload.get("sref_m2", 0.0),
            altitude_m=payload.get("altitude_m", 0.0),
            section_source=payload.get("section_source", ""),
            notes=list(payload.get("notes") or []),
        )


def _induced_at(result: AeroResult, cl: float) -> float | None:
    """Induced drag at a lift coefficient, from the solved polar.

    Fitted as `CDi = k·CL²` through the solved points rather than interpolated, because the
    relationship is quadratic by theory and a fit uses every point instead of the nearest two.
    Refuses to answer outside the solved CL range: past that the fit is an extrapolation, and an
    extrapolated drag is exactly the kind of invented number the rest of this tree refuses.
    """
    cls = np.asarray(result.polar.CL, dtype=float)
    induced = np.asarray(result.drag.cd_induced, dtype=float)
    if cls.size < 3 or induced.size != cls.size:
        return None
    if not (cls.min() - 1e-9) <= cl <= (cls.max() + 1e-9):
        return None
    # Least squares through CDi = a + k·CL², keeping the offset so a solver that carries a little
    # numerical drag at zero lift is described rather than forced through the origin.
    design = np.column_stack([np.ones_like(cls), cls**2])
    coefficients, *_ = np.linalg.lstsq(design, induced, rcond=None)
    return float(coefficients[0] + coefficients[1] * cl * cl)


def _span_load_at(result: AeroResult, cl: float):
    """The span load whose alpha is nearest the one that gives this CL.

    The *shape* of the loading changes slowly with incidence while its magnitude scales with CL, so
    the nearest solved station distribution rescaled to the target CL is a far better estimate than
    re-solving would be worth. Returns (span_load, alpha_deg, scale).
    """
    cls = np.asarray(result.polar.CL, dtype=float)
    alphas = np.asarray(result.polar.alpha_deg, dtype=float)
    if not result.span_loads or cls.size == 0:
        return None, 0.0, 1.0
    alpha = float(np.interp(cl, cls, alphas))
    index = int(np.argmin(np.abs(np.asarray([s.alpha_deg for s in result.span_loads]) - alpha)))
    load = result.span_loads[index]
    nearest_cl = float(np.interp(load.alpha_deg, alphas, cls))
    scale = (cl / nearest_cl) if abs(nearest_cl) > 1e-6 else 1.0
    return load, alpha, scale


def speed_sweep(
    result: AeroResult,
    airframe: ParametricAirframe,
    mass_kg: float,
    speeds_mps: list[float],
    lookup: SectionLookup,
    altitude_m: float = 0.0,
) -> SpeedSweep:
    """Trim the aircraft at each speed and report what it costs to fly there.

    `result` is one completed alpha sweep — its inviscid solution is reused across the whole speed
    range (see the module docstring), while the profile drag is re-integrated at each speed's own
    Reynolds number.
    """
    air = Atmosphere(altitude_m=altitude_m)
    sref = result.reference.sref_m2 or airframe.wing.area()
    sweep = SpeedSweep(
        mass_kg=mass_kg, sref_m2=sref, altitude_m=altitude_m, section_source=lookup.name,
    )
    if mass_kg <= 0 or sref <= 0:
        sweep.notes.append("a speed sweep needs a stated mass and a reference area")
        return sweep

    weight = mass_kg * GRAVITY
    unreachable: list[float] = []

    for velocity in sorted(float(v) for v in speeds_mps):
        if velocity <= 0:
            continue
        q = 0.5 * air.density * velocity * velocity
        cl_required = weight / (q * sref)

        cd_induced = _induced_at(result, cl_required)
        if cd_induced is None:
            # Past the solved range. At the slow end this is the aircraft running out of lift;
            # at the fast end it is simply outside what was swept. Either way it is not answerable
            # from this solve, and inventing it is worse than omitting it.
            unreachable.append(velocity)
            continue

        load, alpha, scale = _span_load_at(result, cl_required)
        cd_profile, missing = 0.0, 0
        if load is not None and load.y_m:
            strips = integrate_profile_drag(
                airframe,
                y_m=load.y_m,
                chord_m=load.chord_m,
                cl_local=[c * scale for c in load.cl_local],
                sref_m2=sref,
                density=air.density,
                velocity_mps=velocity,
                lookup=lookup,
                mu=air.viscosity,
            )
            cd_profile, missing = strips.cd_profile, strips.missing

        sweep.points.append(SpeedPoint(
            velocity_mps=velocity,
            cl_required=cl_required,
            alpha_deg=alpha,
            reynolds=air.reynolds(velocity, airframe.wing.mean_aerodynamic_chord()),
            cd_induced=cd_induced,
            cd_profile=cd_profile,
            cd_total=cd_induced + cd_profile,
            missing_strips=missing,
        ))

    if unreachable:
        low = ", ".join(f"{v:g}" for v in unreachable[:6])
        sweep.notes.append(
            f"{len(unreachable)} speed(s) need a lift coefficient outside the solved polar and "
            f"were dropped rather than extrapolated ({low}{' ...' if len(unreachable) > 6 else ''} "
            f"m/s) - widen the alpha sweep to reach them"
        )
    starved = sum(1 for p in sweep.points if p.missing_strips)
    if starved:
        sweep.notes.append(
            f"{starved} speed(s) lost strips to missing section data, so their profile drag is a "
            f"floor rather than an estimate - the fastest-looking point may simply be the emptiest"
        )
    return sweep
