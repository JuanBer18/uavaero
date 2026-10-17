"""The design space: a small vector of independent variables, and the aircraft it becomes.

**Independent is the whole point.** Span, area and aspect ratio are three names for two numbers —
S = b²/AR — and handing all three to an optimiser lets it ask for combinations that cannot exist,
which it will, immediately and repeatedly. `Planform` already refuses that by taking span plus
exactly one of area or aspect ratio, and the variables below inherit the refusal: there is a `span`
knob and an `aspect_ratio` knob, and area is what falls out.

The same logic governs the tails, one level further out than it first appears. A tail is span,
chord, position and area, which is again more names than numbers — but tail *area* is not
independent of the wing either, and the optimiser is changing the wing at the same time. What
decides whether a tail works is its volume coefficient, `Vh = S_h·l_h / (S_w·MAC)`, which relates
the two. So the variables are **volume and arm**, and the area is derived from whatever wing the
vector just asked for.

Sampling area directly is what the first version did, and it does not work here: span and aspect
ratio swing the wing area by 23x across the default bounds, `Vh` goes as `S_h·l·b / S_w²`, and the
result is a ~500x swing in a quantity no variable controls. The screen then rejected roughly 30% of
every population for a tail volume nothing had chosen — measured at 71% pass over 3000 samples.
Those rejections were free (screening is microseconds, no solver) but they were a third of the
search spent proposing darts.

**Eight to twenty variables, not eighty.** Every extra dimension costs the optimiser exponentially,
and the aircraft is not more optimal for having its fourth fuselage station exposed. The set below
is the one that changes the answer.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from uavaero.geometry import authoring
from uavaero.geometry.planform import Planform
from uavaero.schema.airframe import ParametricAirframe


@dataclass(frozen=True)
class DesignVariable:
    """One independent knob, with the bounds it may be turned between.

    `apply` receives the working airframe and the value. It mutates rather than returning, because
    a candidate is built by applying a whole vector in sequence and threading copies through would
    make the order-dependence between (say) wing shape and tail placement invisible.
    """

    name: str
    low: float
    high: float
    apply: Callable[[ParametricAirframe, float], None]
    unit: str = ""
    description: str = ""
    #: Whether this knob does anything to a given aircraft. None means "always". A tail variable
    #: on a tailless airframe does not, and a dimension that cannot move the design is worse than
    #: useless: the search still pays for it, and the reported best carries a number for a surface
    #: that is not there.
    applies_to: Callable[[ParametricAirframe], bool] | None = None

    def relevant_for(self, airframe: ParametricAirframe) -> bool:
        return True if self.applies_to is None else bool(self.applies_to(airframe))

    def clip(self, value: float) -> float:
        return float(min(max(value, self.low), self.high))


def _wing_setter(field: str) -> Callable[[ParametricAirframe, float], None]:
    """Reshape the wing by one planform field, holding everything else.

    Goes through `authoring.apply_wing`, which re-places both tails at the arms they currently
    have — the arm is measured from the wing's aerodynamic centre, and a sweep or taper change
    moves it. Without that, an optimiser varying wing sweep would be silently varying tail volume
    at the same time and would find "improvements" that are nothing of the kind.
    """

    def apply(airframe: ParametricAirframe, value: float) -> None:
        planform = authoring.wing_planform(airframe)
        if planform is None:
            raise ValueError("this wing has more than two stations and cannot be varied by planform")
        updated = Planform(
            span=planform.span,
            taper=planform.taper,
            aspect_ratio=planform.resolved_aspect_ratio(),
            sweep_mode=planform.sweep_mode,
            sweep_deg=planform.sweep_deg,
            dihedral_deg=planform.dihedral_deg,
            tip_twist_deg=planform.tip_twist_deg,
            airfoil_root=planform.airfoil_root,
            airfoil_tip=planform.airfoil_tip,
        )
        object.__setattr__(updated, field, value)
        # Span and aspect ratio together fix the area; never set area as well.
        updated.area = None
        authoring.apply_wing(airframe, updated)

    return apply


def _has(which: str) -> Callable[[ParametricAirframe], bool]:
    """The surface exists, so a variable that shapes it can move the design."""

    def present(airframe: ParametricAirframe) -> bool:
        return getattr(airframe, which, None) is not None

    return present


def _tail_volume_setter(which: str) -> Callable[[ParametricAirframe, float], None]:
    """Set the tail's volume coefficient, deriving its area from the wing as it now stands.

    Inverts `authoring.horizontal_tail_volume` / `vertical_tail_volume`, so the two cannot drift:
    the horizontal tail references the wing's MAC, the fin references the *full* wing span (one
    side doubled — the asymmetry that has caused a real bug in this tree before).

    **Order matters, and it is the reason the arm variable is listed before this one.** The area
    follows from the volume and the arm together, so a vector that set the volume and then moved
    the tail would silently change the volume it had just asked for. Applying the arm first makes
    this the last word on both.
    """

    def apply(airframe: ParametricAirframe, value: float) -> None:
        surface = getattr(airframe, which)
        if surface is None:
            return
        vertical = which == "vtail"
        wing = airframe.wing
        sizing = authoring.infer_tail_sizing(airframe, surface, vertical=vertical)
        if sizing.arm_m <= 0:
            return
        reference = (
            wing.span() * (2.0 if wing.symmetric else 1.0)
            if vertical
            else wing.mean_aerodynamic_chord()
        )
        area = value * wing.area() * reference / sizing.arm_m
        if area <= 0:
            return
        authoring.apply_tail(
            airframe,
            type(sizing)(**{**sizing.__dict__, "area_m2": area}),
            which,
        )

    return apply


def _tail_setter(which: str, field: str) -> Callable[[ParametricAirframe, float], None]:
    def apply(airframe: ParametricAirframe, value: float) -> None:
        surface = getattr(airframe, which)
        if surface is None:
            return
        sizing = authoring.infer_tail_sizing(airframe, surface, vertical=which == "vtail")
        authoring.apply_tail(
            airframe,
            type(sizing)(**{**sizing.__dict__, field: value}),
            which,
        )

    return apply


#: The standard set. Twelve variables covering the shape decisions that move L/D and the stability
#: that has to survive them.
def standard_variables(
    span: tuple[float, float] = (1.2, 3.2),
    aspect_ratio: tuple[float, float] = (5.0, 16.0),
    taper: tuple[float, float] = (0.35, 1.0),
    sweep_deg: tuple[float, float] = (-5.0, 20.0),
    dihedral_deg: tuple[float, float] = (0.0, 8.0),
    tip_twist_deg: tuple[float, float] = (-6.0, 1.0),
    htail_arm: tuple[float, float] = (0.5, 1.4),
    htail_volume: tuple[float, float] = (0.30, 1.00),
    htail_aspect: tuple[float, float] = (2.0, 5.0),
    vtail_arm: tuple[float, float] = (0.5, 1.4),
    vtail_volume: tuple[float, float] = (0.02, 0.12),
    vtail_aspect: tuple[float, float] = (1.0, 3.0),
) -> list[DesignVariable]:
    return [
        DesignVariable("span", *span, _wing_setter("span"), "m", "wing span, full"),
        DesignVariable("aspect_ratio", *aspect_ratio, _wing_setter("aspect_ratio"), "",
                       "wing aspect ratio; with span this fixes the area"),
        DesignVariable("taper", *taper, _wing_setter("taper"), "", "tip chord over root"),
        DesignVariable("sweep_deg", *sweep_deg, _wing_setter("sweep_deg"), "deg",
                       "leading-edge sweep"),
        DesignVariable("dihedral_deg", *dihedral_deg, _wing_setter("dihedral_deg"), "deg",
                       "wing dihedral; drives the lateral stability"),
        DesignVariable("tip_twist_deg", *tip_twist_deg, _wing_setter("tip_twist_deg"), "deg",
                       "washout; buys stall margin at the tip"),
        # Arm before volume: the area is derived from the two together, so the arm has to be
        # settled first. See `_tail_volume_setter`.
        DesignVariable("htail_arm", *htail_arm, _tail_setter("htail", "arm_m"), "m",
                       "horizontal tail arm, aerodynamic centre to aerodynamic centre",
                       _has("htail")),
        DesignVariable("htail_volume", *htail_volume, _tail_volume_setter("htail"), "",
                       "horizontal tail volume Vh = S_h·l_h / (S_w·MAC); area follows from it",
                       _has("htail")),
        DesignVariable("htail_aspect", *htail_aspect, _tail_setter("htail", "aspect_ratio"), "",
                       "horizontal tail aspect ratio", _has("htail")),
        DesignVariable("vtail_arm", *vtail_arm, _tail_setter("vtail", "arm_m"), "m", "fin arm",
                       _has("vtail")),
        DesignVariable("vtail_volume", *vtail_volume, _tail_volume_setter("vtail"), "",
                       "fin volume Vv = S_v·l_v / (S_w·b); area follows from it", _has("vtail")),
        DesignVariable("vtail_aspect", *vtail_aspect, _tail_setter("vtail", "aspect_ratio"), "",
                       "fin aspect ratio", _has("vtail")),
    ]


class DesignSpace:
    """A set of variables, and the mapping from a vector to an aircraft."""

    def __init__(self, base: ParametricAirframe, variables: Iterable[DesignVariable]) -> None:
        self.base = base
        offered = list(variables)
        # Drop the knobs that cannot move *this* aircraft. A tail variable on a tailless airframe
        # applies to nothing: the setter returns early, so the dimension is inert. An optimiser
        # still pays for it — differential evolution samples and mutates it every generation, and
        # it lands in the reported best as a number describing a surface that is not there.
        self.variables = [v for v in offered if v.relevant_for(base)]
        #: The names that were dropped, so the caller can say why the space is smaller than asked.
        self.omitted = [v.name for v in offered if not v.relevant_for(base)]
        if not 1 <= len(self.variables) <= 24:
            raise ValueError(
                f"a design space of {len(self.variables)} variables is outside the workable range "
                f"- eight to twenty is the band where this converges in sensible time"
            )

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(v.low, v.high) for v in self.variables]

    def centre(self) -> np.ndarray:
        return np.array([(v.low + v.high) / 2.0 for v in self.variables])

    def build(self, vector: np.ndarray) -> ParametricAirframe:
        """One candidate aircraft.

        The base is deep-copied first: the variables mutate in place, and an optimiser that shared
        one airframe between candidates would accumulate every change it had ever tried.
        """
        candidate = copy.deepcopy(self.base)
        for variable, value in zip(self.variables, np.asarray(vector, dtype=float)):
            variable.apply(candidate, variable.clip(float(value)))
        return candidate

    def describe(self, vector: np.ndarray) -> dict[str, float]:
        return {
            v.name: float(value)
            for v, value in zip(self.variables, np.asarray(vector, dtype=float))
        }
