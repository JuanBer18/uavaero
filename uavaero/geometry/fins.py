"""Stability-surface generators for tailless designs (flying wing / future BWB).

`FinSpec` describes one wing-mounted vertical fin the quick way (spanwise position + size);
`generate_wing_fins` turns the list into mirrored pairs of ordinary vertical `LiftingSurface`
objects positioned on the wing — a convenience generator like planform/fuselage_gen (CLAUDE.md
rule 2): the emitted surfaces live in `ParametricAirframe.fins` and remain the single source of
truth for every adapter.

Elevons and winglet control surfaces need no machinery here — they are ordinary `ControlSurface`
entries on the wing, whose span-fraction range can cover the winglet panels.
"""
from __future__ import annotations

from dataclasses import dataclass

from uavaero.schema.airframe import LiftingSurface, WingSection


@dataclass
class FinSpec:
    """One fin pair, described relative to the wing: mounted at `span_fraction` (0=root, 1=tip)
    on BOTH half-wings, trailing-edge aligned with the local chord."""

    span_fraction: float = 0.65
    height: float = 0.12
    root_chord: float = 0.10
    tip_chord: float = 0.06
    sweep_deg: float = 25.0
    airfoil: str = "naca0012"

    def to_dict(self) -> dict:
        return {
            "span_fraction": self.span_fraction, "height": self.height,
            "root_chord": self.root_chord, "tip_chord": self.tip_chord,
            "sweep_deg": self.sweep_deg, "airfoil": self.airfoil,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FinSpec":
        return cls(**payload)


def wing_point_at_fraction(wing: LiftingSurface, fraction: float) -> tuple[tuple[float, float, float], float]:
    """(LE point, chord) on the wing at a span fraction, linear within the panel it falls in —
    the same walk every adapter shares, so fins land exactly on the rendered wing."""
    fraction = min(max(fraction, 0.0), 1.0)
    total = wing.span()
    if total <= 0 or len(wing.sections) < 2:
        return tuple(float(v) for v in wing.origin), wing.sections[0].chord if wing.sections else 0.0

    le_points = wing.le_points()
    running = 0.0
    target = fraction * total
    for i, section in enumerate(wing.sections[1:], start=1):
        if running + section.span >= target or i == len(wing.sections) - 1:
            t = (target - running) / section.span if section.span > 0 else 0.0
            t = min(max(t, 0.0), 1.0)
            inner, outer = le_points[i - 1], le_points[i]
            le = tuple(a + t * (b - a) for a, b in zip(inner, outer))
            chord = wing.sections[i - 1].chord + t * (section.chord - wing.sections[i - 1].chord)
            return le, chord
        running += section.span
    raise AssertionError("unreachable")


def generate_wing_fins(wing: LiftingSurface, specs: list[FinSpec]) -> list[LiftingSurface]:
    """Mirrored fin pairs on the wing. Each fin's LE sits so its trailing edge aligns with the
    local wing trailing edge; the fin grows upward (vertical surface) from the local wing z."""
    fins: list[LiftingSurface] = []
    for i, spec in enumerate(specs):
        (x, y, z), chord = wing_point_at_fraction(wing, spec.span_fraction)
        x_le = x + max(chord - spec.root_chord, 0.0)  # TE-aligned
        sections = [
            WingSection(chord=spec.root_chord, airfoil=spec.airfoil),
            WingSection(chord=spec.tip_chord, airfoil=spec.airfoil,
                        span=spec.height, sweep_deg=spec.sweep_deg),
        ]
        for side, sign in (("r", +1.0), ("l", -1.0)):
            fins.append(LiftingSurface(
                name=f"fin{i}_{side}",
                origin=(x_le, sign * y, z),
                sections=[WingSection(**vars(s)) for s in sections],
                symmetric=False,
                vertical=True,
            ))
    return fins
