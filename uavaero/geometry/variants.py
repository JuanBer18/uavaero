"""Variants of one wing at constant area — the comparison that makes a choice mean something.

Changing an airfoil, an aspect ratio or a taper and re-solving tells you very little on its own,
because the wing also got bigger or smaller and most of what moved was area. Holding **area fixed**
is what turns a sweep into an experiment: the lift the wing has to produce at a given speed is then
identical across every variant, so a difference in L/D is a difference in the shape rather than in
the size.

Area is held rather than span because area is what sizing converged on — it came out of the
constraint diagram and the mass fixed point together, and moving it invalidates the take-off mass
the whole design rests on. Aspect ratio then sets the span: `b = sqrt(S * AR)`.

A variant states only what it changes. Everything unstated is inherited from the base wing, so a
variant that swaps an airfoil is one field, and reading a list of them tells you what the study was
actually about.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from uavaero.geometry.planform import Planform, infer_planform, to_surface
from uavaero.schema.airframe import ParametricAirframe


@dataclass
class Variant:
    """One change to try. `None` means "leave the base alone"."""

    label: str = ""
    aspect_ratio: float | None = None
    taper: float | None = None
    airfoil_root: str | None = None
    airfoil_tip: str | None = None
    tip_twist_deg: float | None = None
    dihedral_deg: float | None = None
    sweep_mode: str | None = None
    sweep_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "aspect_ratio": self.aspect_ratio, "taper": self.taper,
            "airfoil_root": self.airfoil_root, "airfoil_tip": self.airfoil_tip,
            "tip_twist_deg": self.tip_twist_deg, "dihedral_deg": self.dihedral_deg,
            "sweep_mode": self.sweep_mode, "sweep_deg": self.sweep_deg,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Variant":
        known = set(cls.__dataclass_fields__)  # noqa: F821
        return cls(**{k: v for k, v in (payload or {}).items() if k in known})

    def changes(self) -> dict[str, Any]:
        """Only the fields this variant actually sets — what the study is about."""
        return {
            key: value for key, value in self.to_dict().items()
            if key != "label" and value is not None
        }

    def describe(self) -> str:
        changed = self.changes()
        if not changed:
            return "unchanged from the base wing"
        parts = []
        for key, value in changed.items():
            name = key.replace("_deg", "").replace("_", " ")
            parts.append(f"{name} {value:g}" if isinstance(value, (int, float))
                         else f"{name} {value}")
        return ", ".join(parts)


def base_planform(airframe: ParametricAirframe) -> Planform | None:
    """The base wing as simple-planform numbers, or None if it is genuinely multi-panel.

    A multi-panel wing has no single taper or aspect ratio to vary, so a variant study on one would
    be quietly changing something other than what it claims.
    """
    return infer_planform(airframe.wing)


def apply(airframe: ParametricAirframe, variant: Variant) -> ParametricAirframe:
    """The base wing with this variant's changes, **at the base wing's area**.

    Raises for a wing the planform model cannot describe rather than silently varying a shape it
    does not understand.
    """
    planform = base_planform(airframe)
    if planform is None:
        raise ValueError(
            "this wing is multi-panel, so it has no single aspect ratio or taper to vary — "
            "edit the stations directly instead")

    area = planform.resolved_area()
    aspect_ratio = (variant.aspect_ratio
                    if variant.aspect_ratio is not None else planform.resolved_aspect_ratio())
    if area <= 0 or aspect_ratio <= 0:
        raise ValueError(f"cannot build a variant from area {area} and aspect ratio {aspect_ratio}")

    varied = Planform(
        # Area is the constant; the span follows from the aspect ratio being asked for.
        span=math.sqrt(area * aspect_ratio),
        area=area,
        taper=variant.taper if variant.taper is not None else planform.taper,
        sweep_mode=variant.sweep_mode if variant.sweep_mode is not None else planform.sweep_mode,
        sweep_deg=variant.sweep_deg if variant.sweep_deg is not None else planform.sweep_deg,
        dihedral_deg=(variant.dihedral_deg if variant.dihedral_deg is not None
                      else planform.dihedral_deg),
        tip_twist_deg=(variant.tip_twist_deg if variant.tip_twist_deg is not None
                       else planform.tip_twist_deg),
        airfoil_root=(variant.airfoil_root if variant.airfoil_root is not None
                      else planform.airfoil_root),
        airfoil_tip=(variant.airfoil_tip if variant.airfoil_tip is not None
                     else planform.airfoil_tip),
        symmetric=planform.symmetric,
    )

    result = ParametricAirframe.from_dict(airframe.to_dict())
    result.name = f"{airframe.name} · {variant.label}" if variant.label else airframe.name
    result.wing = to_surface(varied, name=airframe.wing.name, origin=airframe.wing.origin)
    # Control surfaces are spanwise fractions, so they survive a change of span unchanged.
    result.wing.control_surfaces = list(airframe.wing.control_surfaces)
    return result
