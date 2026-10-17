"""Where to put the battery: static margin and the longitudinal modes against CG position.

Every longitudinal number this tool produces depends on one input nobody measures — where the
centre of gravity is. Get it wrong and `Cm_alpha`, the static margin, the short period and the
phugoid are all wrong together, in a way that still looks entirely like an aeroplane. The reference
airframe shipped with `xyz_ref = (0, 0, 0)`, which puts the moment reference at the nose and reports
a static margin of 76% of MAC. That is not a stable aircraft; it is an unset field.

**This costs no solve.** The neutral point is a property of the *shape*, not of where the mass sits
— it is where the lift increment from a change of incidence acts, and moving the CG does not move
it. So one stability run gives `X_np`, and the whole curve of static margin against CG position is
arithmetic:

    SM(x) = (X_np − x) / cref
    Cm_alpha(x) = −CL_alpha · SM(x)

and feeding that `Cm_alpha` back through the longitudinal matrix gives the short period and phugoid
at every candidate CG. So a sweep that would cost twenty solves costs none.

**Two things it does not model, and says so.** `Cm_q` changes with CG as well — the tail's moment
arm is measured from the CG — and so does `Iyy`. Both are held at the solved values, which makes
the short-period damping progressively less trustworthy the further the sweep runs from the CG the
solve was actually run at. The result carries that distance so the caller can report it rather than
discover it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from uavaero.dynamics import modes as mode_analysis
from uavaero.dynamics import statespace
from uavaero.schema.condition import OperatingCondition
from uavaero.schema.mass import MassProperties
from uavaero.schema.stability import StabilityResult

#: The band a conventional aircraft is designed into, as a fraction of MAC. Outside it is not
#: forbidden - a canard or a deliberately relaxed design lives elsewhere - but it is worth saying.
CONVENTIONAL_MARGIN = (0.05, 0.30)


@dataclass
class CgPoint:
    """One candidate CG position, and what the aircraft does there."""

    x_m: float
    #: Position as a fraction of MAC aft of the mean aerodynamic chord's leading edge.
    percent_mac: float
    static_margin: float
    Cm_alpha: float
    short_period_wn: float | None = None
    short_period_damping: float | None = None
    phugoid_wn: float | None = None
    phugoid_damping: float | None = None

    @property
    def stable(self) -> bool:
        """Statically stable in pitch: the neutral point is behind the CG."""
        return self.static_margin > 0.0

    @property
    def conventional(self) -> bool:
        low, high = CONVENTIONAL_MARGIN
        return low <= self.static_margin <= high

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_m": self.x_m, "percent_mac": self.percent_mac,
            "static_margin": self.static_margin, "Cm_alpha": self.Cm_alpha,
            "short_period_wn": self.short_period_wn,
            "short_period_damping": self.short_period_damping,
            "phugoid_wn": self.phugoid_wn, "phugoid_damping": self.phugoid_damping,
        }


@dataclass
class CgSweep:
    """Static margin and the longitudinal modes across a range of CG positions."""

    points: list[CgPoint] = field(default_factory=list)
    neutral_point_x_m: float = 0.0
    mac_m: float = 0.0
    mac_leading_edge_x_m: float = 0.0
    solved_at_x_m: float = 0.0
    notes: list[str] = field(default_factory=list)

    def neutral_point_percent_mac(self) -> float:
        if self.mac_m <= 0:
            return 0.0
        return (self.neutral_point_x_m - self.mac_leading_edge_x_m) / self.mac_m * 100.0

    def conventional_span(self) -> tuple[float, float] | None:
        """The CG range that lands inside the conventional static-margin band, if any."""
        inside = [p.x_m for p in self.points if p.conventional]
        return (min(inside), max(inside)) if inside else None

    def recommended(self) -> CgPoint | None:
        """The point nearest the middle of the conventional band.

        A recommendation, not an answer: it knows nothing about where the battery physically fits,
        what the payload weighs, or what the aircraft is for.
        """
        low, high = CONVENTIONAL_MARGIN
        target = 0.5 * (low + high)
        inside = [p for p in self.points if p.conventional]
        return min(inside, key=lambda p: abs(p.static_margin - target), default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "neutral_point_x_m": self.neutral_point_x_m,
            "mac_m": self.mac_m, "mac_leading_edge_x_m": self.mac_leading_edge_x_m,
            "solved_at_x_m": self.solved_at_x_m, "notes": list(self.notes),
        }


def cg_sweep(
    stability: StabilityResult,
    mass: MassProperties,
    condition: OperatingCondition,
    positions_m: list[float],
    mac_leading_edge_x_m: float = 0.0,
) -> CgSweep:
    """Static margin and the longitudinal modes at each candidate CG. No solver runs."""
    cref = stability.cref_m
    sweep = CgSweep(
        mac_m=cref,
        mac_leading_edge_x_m=mac_leading_edge_x_m,
        solved_at_x_m=stability.x_cg_m,
    )
    if cref <= 0:
        sweep.notes.append("no reference chord on the stability result - cannot express a margin")
        return sweep

    neutral = stability.neutral_point_x_m
    if neutral is None:
        # Recover it from the derivatives: SM = -Cm_alpha / CL_alpha, about the CG it was solved at.
        margin = stability.static_margin_from_derivatives
        if margin is None:
            sweep.notes.append(
                "the solve reported neither a neutral point nor a usable Cm_alpha - "
                "there is nothing to sweep against"
            )
            return sweep
        neutral = stability.x_cg_m + margin * cref
        sweep.notes.append(
            "neutral point recovered from Cm_alpha/CL_alpha - VSPAERO did not report one directly"
        )
    sweep.neutral_point_x_m = float(neutral)

    for x in sorted(float(v) for v in positions_m):
        margin = (neutral - x) / cref
        shifted = copy.deepcopy(stability)
        shifted.x_cg_m = x
        # SM = -Cm_alpha/CL_alpha is the definition, so this is the same statement rearranged
        # rather than a model of how Cm_alpha varies.
        shifted.Cm_alpha = -stability.CL_alpha * margin
        shifted.neutral_point_x_m = float(neutral)
        shifted.static_margin_reported = None

        point = CgPoint(
            x_m=x,
            percent_mac=(x - mac_leading_edge_x_m) / cref * 100.0,
            static_margin=margin,
            Cm_alpha=shifted.Cm_alpha,
        )
        try:
            trim = statespace.trim_level_flight(shifted, mass, condition)
            matrix = statespace.longitudinal_matrix(shifted, mass, trim)
            for mode in mode_analysis.analyse(matrix, ("u", "w", "q", "theta")):
                if mode.name == "short period":
                    point.short_period_wn = mode.natural_frequency
                    point.short_period_damping = mode.damping_ratio
                elif mode.name == "phugoid":
                    point.phugoid_wn = mode.natural_frequency
                    point.phugoid_damping = mode.damping_ratio
        except Exception as exc:  # noqa: BLE001 - a mode that will not solve is not a lost sweep
            sweep.notes.append(f"modes at x={x:.3f} m: {type(exc).__name__}: {exc}")
        sweep.points.append(point)

    furthest = max(
        (abs(p.x_m - stability.x_cg_m) for p in sweep.points), default=0.0
    )
    if cref > 0 and furthest > 0.25 * cref:
        sweep.notes.append(
            f"Cm_q and Iyy are held at the values solved for a CG at x={stability.x_cg_m:.3f} m, "
            f"and this sweep runs {furthest / cref * 100:.0f}% of MAC away from it - the static "
            f"margin stays exact, the short-period damping does not"
        )
    if not any(p.conventional for p in sweep.points):
        sweep.notes.append(
            f"no position in this range lands in the conventional "
            f"{CONVENTIONAL_MARGIN[0]:.0%}-{CONVENTIONAL_MARGIN[1]:.0%} static-margin band; "
            f"the neutral point is at x={neutral:.3f} m"
        )
    return sweep
