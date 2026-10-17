"""THE one geometry parametrisation.

`ParametricAirframe` is the only geometry definition in this tool, and in V2 it is also the *unit
of work*: you author one of these directly and everything else is derived from it. Nothing sizes it
for you. Everything downstream (the OpenVSP adapter, CAD export, mesh preview) is a stateless
translation of this spec — never a second parametrisation, never authoritative on its own.

Encoding: each `WingSection` is a spanwise station carrying its own chord/twist/airfoil plus the
panel *leading into it* from the previous station (span/sweep/dihedral — ignored on the root
station). This maps 1:1 onto OpenVSP's per-panel drivers and converts to CAD via one shared
cumulative leading-edge walk (`LiftingSurface.le_points`), so adapters can never disagree about
where a station sits in space.

**The asymmetry that has caused real bugs**: `LiftingSurface.span()` is one side and is NOT
doubled for a symmetric surface, while `area()` IS already doubled. Feeding a half-span to a
solver as `bref` quadruples the apparent aspect ratio, and the only symptom is an Oswald
efficiency above 1. Read both docstrings before using either.

Deliberate non-members: mesh/panel densities (solver run settings, not shape), mass properties
(out of scope in an aero-first tool), anything Qt.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


SPEC_VERSION = 3  # v2: `fuselage` -> `fuselages` list; v3: added `fins` (tailless stability surfaces)



@dataclass
class WingSection:
    """One spanwise station. `span`/`sweep_deg`/`dihedral_deg` describe the trapezoidal panel from
    the previous station to this one, so they are ignored (conventionally 0) on a surface's root
    section. A winglet is nothing special — a tip section with dihedral_deg near 90 and small span."""

    chord: float
    twist_deg: float = 0.0
    airfoil: str = "naca2412"
    span: float = 0.0
    sweep_deg: float = 0.0
    dihedral_deg: float = 0.0


@dataclass
class ControlSurface:
    """Span fractions are relative to the surface's own span (0=root, 1=tip).

    Carried by the spec from the start even though only the derivatives work consumes deflections,
    because a control surface is part of the shape: it changes the OpenVSP build and the CAD
    export whether or not anything deflects it yet."""

    name: str
    kind: str  # "aileron" | "elevator" | "rudder" | "flap" | "ruddervator"
    span_start: float
    span_end: float
    chord_fraction: float
    symmetric: bool = True  # deflect same-sign on both halves (elevator) vs opposite (aileron)


@dataclass
class LiftingSurface:
    """A wing, horizontal tail, vertical tail, or V-tail — all the same thing here.

    `vertical=True` builds the surface in the x-z plane (fin): the LE walk adds 90 deg to every
    panel's dihedral, and the OpenVSP adapter uses X_Rel_Rotation=90 with Sym_Planar_Flag forced
    off (the former hard-crash bug). A V-tail is NOT vertical — it's a symmetric htail whose
    panels carry large dihedral.
    """

    name: str
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sections: list[WingSection] = field(default_factory=list)
    symmetric: bool = True
    vertical: bool = False
    control_surfaces: list[ControlSurface] = field(default_factory=list)

    def span(self) -> float:
        """Sum of panel spans, one side (not doubled for symmetric surfaces)."""
        return sum(s.span for s in self.sections[1:])

    def area(self) -> float:
        """Planform area: trapezoid per panel, doubled when symmetric."""
        one_side = sum(
            0.5 * (inner.chord + outer.chord) * outer.span
            for inner, outer in zip(self.sections, self.sections[1:])
        )
        return one_side * (2.0 if self.symmetric else 1.0)

    def le_points(self) -> list[tuple[float, float, float]]:
        """Leading-edge point of every station — THE shared spatial walk every adapter uses.

        Per panel: dx = span*tan(sweep), dy = span*cos(gamma), dz = span*sin(gamma) with
        gamma = dihedral (+90 deg when the surface is vertical). Sweep is leading-edge sweep.
        """
        points = [tuple(float(v) for v in self.origin)]
        x, y, z = points[0]
        for section in self.sections[1:]:
            gamma = math.radians(section.dihedral_deg + (90.0 if self.vertical else 0.0))
            x += section.span * math.tan(math.radians(section.sweep_deg))
            y += section.span * math.cos(gamma)
            z += section.span * math.sin(gamma)
            points.append((x, y, z))
        return points

    def mean_chord(self) -> float:
        if not self.sections:
            return 0.0
        return sum(s.chord for s in self.sections) / len(self.sections)

    def section_span_fractions(self) -> list[float]:
        """Cumulative span fraction (0 at root, 1 at tip) of every station — THE shared spatial
        fraction every adapter/study maps span-relative quantities (control surfaces, strip-theory
        stations) onto."""
        total = self.span()
        if total <= 0:
            return [0.0] * len(self.sections)
        fractions, running = [0.0], 0.0
        for section in self.sections[1:]:
            running += section.span
            fractions.append(running / total)
        return fractions

    def mean_aerodynamic_chord(self) -> float:
        """True area-weighted MAC — NOT `mean_chord()`'s plain arithmetic average. Formula ported
        from AeroSandbox's own `Wing.mean_aerodynamic_chord()` (verified against the installed
        package): per panel, MAC_panel = (2/3)*c_i*(1+t+t^2)/(1+t) with t = c_o/c_i, then
        area-weighted across panels using the same per-panel trapezoid areas `area()` sums."""
        panels = list(zip(self.sections, self.sections[1:]))
        if not panels:
            return 0.0
        total_area = 0.0
        weighted_mac = 0.0
        for inner, outer in panels:
            panel_area = 0.5 * (inner.chord + outer.chord) * outer.span
            t = outer.chord / inner.chord if inner.chord > 0 else 1.0
            panel_mac = (2.0 / 3.0) * inner.chord * (1 + t + t**2) / (1 + t)
            weighted_mac += panel_mac * panel_area
            total_area += panel_area
        return weighted_mac / total_area if total_area > 0 else 0.0

    def mac_leading_edge(self) -> tuple[float, float, float]:
        """(x, y, z) of the MAC's leading edge, area-weighted across panels the same way
        `mean_aerodynamic_chord()` is — formula ported from AeroSandbox's own
        `Wing.aerodynamic_center()` (verified against the installed package):
        MAC_le = le_i + (le_o - le_i) * (1+2t)/(3+3t), t = c_o/c_i."""
        panels = list(zip(self.sections, self.sections[1:]))
        le = self.le_points()
        le_panels = list(zip(le, le[1:]))
        if not panels:
            return tuple(float(v) for v in self.origin)
        total_area = 0.0
        weighted = [0.0, 0.0, 0.0]
        for (inner, outer), (le_inner, le_outer) in zip(panels, le_panels):
            panel_area = 0.5 * (inner.chord + outer.chord) * outer.span
            t = outer.chord / inner.chord if inner.chord > 0 else 1.0
            frac = (1 + 2 * t) / (3 + 3 * t)
            panel_mac_le = tuple(a + (b - a) * frac for a, b in zip(le_inner, le_outer))
            for i in range(3):
                weighted[i] += panel_mac_le[i] * panel_area
            total_area += panel_area
        if total_area <= 0:
            return le[0]
        return tuple(w / total_area for w in weighted)


@dataclass
class FuselageStation:
    """One cross-section of the fuselage loft: elliptical section of `width` x `height` centered
    at (x, z) relative to the fuselage origin."""

    x: float
    width: float
    height: float
    z: float = 0.0


@dataclass
class Fuselage:
    """One body. An airframe can carry several (pod + tail boom, twin booms, twin fuselages);
    each has its own name and origin — a boom at +y is just a Fuselage with origin y > 0."""

    name: str = "fuselage"
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stations: list[FuselageStation] = field(default_factory=list)

    def length(self) -> float:
        if len(self.stations) < 2:
            return 0.0
        xs = [s.x for s in self.stations]
        return max(xs) - min(xs)


@dataclass
class ParametricAirframe:
    """The whole airframe. Conventional tail = htail (low dihedral) + vtail (vertical=True);
    V-tail = htail with large panel dihedral and vtail=None."""

    name: str = "airframe"
    wing: LiftingSurface = field(default_factory=lambda: LiftingSurface(name="wing"))
    htail: LiftingSurface | None = None
    vtail: LiftingSurface | None = None
    #: Extra stabilizing surfaces (wing-mounted fins on tailless designs, etc.) — full
    #: LiftingSurfaces, so every adapter handles them with zero special cases.
    fins: list[LiftingSurface] = field(default_factory=list)
    fuselages: list[Fuselage] = field(default_factory=list)

    def is_tailless(self) -> bool:
        return self.htail is None and self.vtail is None

    def lifting_surfaces(self) -> list[LiftingSurface]:
        return [s for s in (self.wing, self.htail, self.vtail) if s is not None] + self.fins

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec_version"] = SPEC_VERSION
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParametricAirframe":
        version = payload.get("spec_version", SPEC_VERSION)
        if version > SPEC_VERSION:
            raise ValueError(
                f"Geometry spec version {version} is newer than this app supports ({SPEC_VERSION})"
            )
        if version < 2 and payload.get("fuselage") is not None:
            # v1 -> v2: the single optional `fuselage` became the `fuselages` list
            payload = {**payload, "fuselages": [payload["fuselage"]]}
        return cls(
            name=payload.get("name", "airframe"),
            wing=_surface_from_dict(payload["wing"]),
            htail=_optional(_surface_from_dict, payload.get("htail")),
            vtail=_optional(_surface_from_dict, payload.get("vtail")),
            fins=[_surface_from_dict(f) for f in payload.get("fins") or []],
            fuselages=[_fuselage_from_dict(f) for f in payload.get("fuselages") or []],
        )


def _optional(builder, payload):
    return None if payload is None else builder(payload)


def _surface_from_dict(payload: dict[str, Any]) -> LiftingSurface:
    return LiftingSurface(
        name=payload["name"],
        origin=tuple(payload.get("origin", (0.0, 0.0, 0.0))),
        sections=[WingSection(**s) for s in payload.get("sections", [])],
        symmetric=payload.get("symmetric", True),
        vertical=payload.get("vertical", False),
        control_surfaces=[ControlSurface(**c) for c in payload.get("control_surfaces", [])],
    )


def _fuselage_from_dict(payload: dict[str, Any]) -> Fuselage:
    return Fuselage(
        name=payload.get("name", "fuselage"),
        origin=tuple(payload.get("origin", (0.0, 0.0, 0.0))),
        stations=[FuselageStation(**s) for s in payload.get("stations", [])],
    )


def default_airframe() -> ParametricAirframe:
    """A complete conventional aircraft — wing, htail, vtail and a fuselage — at small fixed-wing
    UAS scale (IMechE UAS Challenge class).

    It is deliberately whole rather than a bare wing. In the previous generation of this tool the
    only automatically-produced airframe was wing-only, which silently blocked every whole-aircraft
    solve scope and made a static margin impossible to ask for. Every reference case in this tree
    starts from something that has all four."""
    return ParametricAirframe(
        name="new-airframe",
        wing=LiftingSurface(
            name="wing",
            origin=(0.0, 0.0, 0.0),
            sections=[
                WingSection(chord=0.30, airfoil="naca2412"),
                WingSection(chord=0.20, airfoil="naca2412", span=1.20, sweep_deg=2.0, dihedral_deg=3.0),
            ],
        ),
        htail=LiftingSurface(
            name="htail",
            origin=(0.90, 0.0, 0.05),
            sections=[
                WingSection(chord=0.16, airfoil="naca0012"),
                WingSection(chord=0.12, airfoil="naca0012", span=0.35),
            ],
        ),
        vtail=LiftingSurface(
            name="vtail",
            origin=(0.90, 0.0, 0.05),
            symmetric=False,
            vertical=True,
            sections=[
                WingSection(chord=0.16, airfoil="naca0012"),
                WingSection(chord=0.11, airfoil="naca0012", span=0.25, sweep_deg=15.0),
            ],
        ),
        fuselages=[Fuselage(
            origin=(-0.25, 0.0, 0.0),
            stations=[
                FuselageStation(x=0.00, width=0.02, height=0.02),
                FuselageStation(x=0.15, width=0.10, height=0.11),
                FuselageStation(x=0.55, width=0.11, height=0.12),
                FuselageStation(x=0.95, width=0.05, height=0.06),
                FuselageStation(x=1.25, width=0.015, height=0.015),
            ],
        )],
    )
