"""Simple-planform generator: span/area/aspect-ratio/taper -> spanwise sections.

This is a UI-level CONVENIENCE GENERATOR, not a second parametrization (CLAUDE.md rule 2): it
produces ordinary `WingSection` lists that get written into the one `ParametricAirframe` spec.
The sections remain the single source of truth; these helpers are pure functions from planform
numbers to sections.

Auto-sweep: instead of entering a sweep angle, pick which chord line should be STRAIGHT
(unswept) — leading edge, quarter-chord, or trailing edge. For a tapered panel, holding the line
at chord fraction f straight requires the LE to sweep back by f*(c_root - c_tip) over the panel
span; the equivalent LE sweep angle is computed here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from uavaero.schema.airframe import LiftingSurface, WingSection

#: sweep_mode -> the chord fraction that stays straight (None = user-entered manual sweep).
SWEEP_MODES: dict[str, float | None] = {
    "manual": None,
    "straight_le": 0.0,
    "straight_qc": 0.25,
    "straight_te": 1.0,
}


@dataclass
class Planform:
    """One trapezoidal surface described the quick way. Exactly one of `area` / `aspect_ratio`
    drives the sizing (the other is derived). `span` is the FULL span for symmetric surfaces and
    the height for a single-sided (vertical) surface — matching how each is quoted in practice."""

    span: float
    taper: float = 1.0
    area: float | None = None
    aspect_ratio: float | None = None
    sweep_mode: str = "manual"
    sweep_deg: float = 0.0
    dihedral_deg: float = 0.0
    tip_twist_deg: float = 0.0
    airfoil_root: str = "naca2412"
    airfoil_tip: str = "naca2412"
    symmetric: bool = True

    def resolved_area(self) -> float:
        if self.area is not None:
            return self.area
        if self.aspect_ratio is None or self.aspect_ratio <= 0:
            raise ValueError("Planform needs either area or aspect_ratio")
        return self.span**2 / self.aspect_ratio

    def resolved_aspect_ratio(self) -> float:
        return self.span**2 / self.resolved_area()

    def chords(self) -> tuple[float, float]:
        """(root chord, tip chord) of the trapezoid: S = span * (c_root + c_tip)/2 with
        c_tip = taper * c_root — the full-span/full-area form, which holds one-sided too."""
        if self.span <= 0:
            raise ValueError("Planform span must be positive")
        c_root = 2.0 * self.resolved_area() / (self.span * (1.0 + self.taper))
        return c_root, self.taper * c_root

    def panel_span(self) -> float:
        """Span of the one modeled panel: half-span for symmetric surfaces (the mirror is
        implicit), full value for a single-sided surface."""
        return self.span / 2.0 if self.symmetric else self.span

    def resolved_sweep_deg(self) -> float:
        c_root, c_tip = self.chords()
        return resolve_sweep_deg(self.sweep_mode, self.sweep_deg, c_root, c_tip, self.panel_span())

    def to_sections(self) -> list[WingSection]:
        c_root, c_tip = self.chords()
        return [
            WingSection(chord=c_root, airfoil=self.airfoil_root),
            WingSection(
                chord=c_tip,
                airfoil=self.airfoil_tip,
                span=self.panel_span(),
                sweep_deg=self.resolved_sweep_deg(),
                dihedral_deg=self.dihedral_deg,
                twist_deg=self.tip_twist_deg,
            ),
        ]


def to_surface(
    planform: Planform,
    name: str = "wing",
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertical: bool = False,
) -> LiftingSurface:
    """Planform numbers -> a `LiftingSurface`, the forward direction of `infer_planform`.

    `to_sections()` already does the geometry; this exists so that the one place a planform
    becomes a surface is here rather than open-coded in sizing, in the wing editor, and in
    whatever comes next. `infer_planform(to_surface(p))` round-trips, which is what makes the
    simple editor and the stations editor two views of one thing rather than two sources of truth.
    """
    return LiftingSurface(
        name=name,
        origin=origin,
        sections=planform.to_sections(),
        symmetric=planform.symmetric,
        vertical=vertical,
    )


def resolve_sweep_deg(
    mode: str, manual_sweep_deg: float, root_chord: float, tip_chord: float, panel_span: float
) -> float:
    """LE sweep angle for a sweep mode: manual passes through; the straight-line modes return
    the LE sweep that keeps the line at chord fraction f unswept across the panel."""
    if mode not in SWEEP_MODES:
        raise ValueError(f"Unknown sweep mode '{mode}' (valid: {', '.join(SWEEP_MODES)})")
    fraction = SWEEP_MODES[mode]
    if fraction is None:
        return manual_sweep_deg
    if panel_span <= 0:
        return 0.0
    return math.degrees(math.atan(fraction * (root_chord - tip_chord) / panel_span))


def infer_planform(surface: LiftingSurface) -> Planform | None:
    """Best-effort inverse: recover simple-planform numbers from a plain 2-station surface (or
    2 stations + a winglet tip panel with dihedral > 60 deg), so a reopened project can populate
    the simple editor. Returns None when the surface is genuinely multi-panel — the stations
    editor is the honest view then. The sweep mode is recognized when the stored sweep matches an
    auto mode within a small tolerance, otherwise it comes back as manual."""
    sections = list(surface.sections)
    # strip winglet panels (and their transition panels) off the tip: trailing sections whose
    # panel dihedral reads as "folding up" (>= 30 deg) are winglet structure, not the trapezoid
    while len(sections) > 2 and sections[-1].dihedral_deg >= 30.0:
        sections = sections[:-1]
    if len(sections) != 2:
        return None

    root, tip = sections
    if root.chord <= 0 or tip.span <= 0:
        return None
    span = tip.span * (2.0 if surface.symmetric else 1.0)
    area = 0.5 * (root.chord + tip.chord) * tip.span * (2.0 if surface.symmetric else 1.0)

    sweep_mode, sweep_deg = "manual", tip.sweep_deg
    for mode, fraction in SWEEP_MODES.items():
        if fraction is None:
            continue
        auto = resolve_sweep_deg(mode, 0.0, root.chord, tip.chord, tip.span)
        if math.isclose(tip.sweep_deg, auto, abs_tol=0.05):
            sweep_mode, sweep_deg = mode, auto
            break

    return Planform(
        span=span,
        taper=tip.chord / root.chord,
        area=area,
        sweep_mode=sweep_mode,
        sweep_deg=sweep_deg,
        dihedral_deg=tip.dihedral_deg,
        tip_twist_deg=tip.twist_deg,
        airfoil_root=root.airfoil,
        airfoil_tip=tip.airfoil,
        symmetric=surface.symmetric,
    )
