"""Known-answer / known-bug case geometry, shared across the reference-case suite (CLAUDE.md:
permanent regression tests, not one-off validation).

The three former OpenVSP bug cases reproduce, deliberately, the exact configurations that used to
crash or degenerate AEROAPP's meshes — they must keep passing through the new openvsp_adapter.
"""
from __future__ import annotations

from uavaero.schema.airframe import (
    ControlSurface,
    Fuselage,
    FuselageStation,
    LiftingSurface,
    ParametricAirframe,
    WingSection,
)


def _wing(dihedral_deg: float = 3.0, twist_deg: float = 0.0, winglet: bool = False) -> LiftingSurface:
    sections = [
        WingSection(chord=0.30, airfoil="naca2412"),
        WingSection(
            chord=0.20, airfoil="naca2412", span=1.20, sweep_deg=2.0,
            dihedral_deg=dihedral_deg, twist_deg=twist_deg,
        ),
    ]
    if winglet:
        sections.append(
            WingSection(chord=0.10, airfoil="naca2412", span=0.15, sweep_deg=20.0, dihedral_deg=88.0)
        )
    return LiftingSurface(name="wing", sections=sections)


def _htail(dihedral_deg: float = 0.0) -> LiftingSurface:
    return LiftingSurface(
        name="htail",
        origin=(0.90, 0.0, 0.05),
        sections=[
            WingSection(chord=0.16, airfoil="naca0012"),
            WingSection(chord=0.12, airfoil="naca0012", span=0.35, dihedral_deg=dihedral_deg),
        ],
    )


def _vtail(airfoil: str = "naca0012") -> LiftingSurface:
    return LiftingSurface(
        name="vtail",
        origin=(0.90, 0.0, 0.05),
        symmetric=False,
        vertical=True,
        sections=[
            WingSection(chord=0.16, airfoil=airfoil),
            WingSection(chord=0.11, airfoil=airfoil, span=0.25, sweep_deg=15.0),
        ],
    )


def _fuselage() -> Fuselage:
    return Fuselage(
        origin=(-0.25, 0.0, 0.0),
        stations=[
            FuselageStation(x=0.00, width=0.02, height=0.02),
            FuselageStation(x=0.15, width=0.10, height=0.11),
            FuselageStation(x=0.55, width=0.11, height=0.12),
            FuselageStation(x=0.95, width=0.05, height=0.06),
            FuselageStation(x=1.25, width=0.015, height=0.015),
        ],
    )


# --- Archetype configurations (Phase 1 exit criteria 2 and 6) --------------------------------


def conventional() -> ParametricAirframe:
    return ParametricAirframe(name="conventional", wing=_wing(), htail=_htail(), vtail=_vtail())


def vee_tail() -> ParametricAirframe:
    """V-tail = htail with large panel dihedral, no vertical surface at all."""
    return ParametricAirframe(name="vee-tail", wing=_wing(), htail=_htail(dihedral_deg=38.0))


def conventional_winglet() -> ParametricAirframe:
    return ParametricAirframe(
        name="conventional-winglet", wing=_wing(winglet=True), htail=_htail(), vtail=_vtail()
    )


def vee_tail_fuselage() -> ParametricAirframe:
    return ParametricAirframe(
        name="vee-tail-fuselage", wing=_wing(), htail=_htail(dihedral_deg=38.0), fuselages=[_fuselage()]
    )


def full_configuration() -> ParametricAirframe:
    return ParametricAirframe(
        name="full-configuration",
        wing=_wing(winglet=True),
        htail=_htail(),
        vtail=_vtail(),
        fuselages=[_fuselage()],
    )


def twin_boom() -> ParametricAirframe:
    """Auto-generated twin-boom layout: pod + two booms arriving at the h-tail tips."""
    from uavaero.geometry.fuselage_gen import generate_fuselages

    wing, htail, vtail = _wing(), _htail(), _vtail()
    return ParametricAirframe(
        name="twin-boom", wing=wing, htail=htail, vtail=vtail,
        fuselages=generate_fuselages("twin_boom", wing, htail=htail, tail_arm=0.9, tail_z=0.05),
    )


def flying_wing() -> ParametricAirframe:
    """Tailless configuration: swept wing + winglet, elevons, mirrored stabilizing fins on the
    wing, and an automatic pod body — no tail surfaces at all."""
    from uavaero.geometry.fuselage_gen import generate_fuselages
    from uavaero.geometry.fins import FinSpec, generate_wing_fins

    wing = _wing(winglet=True)
    wing.sections[1].sweep_deg = 18.0  # flying wings want real sweep for pitch stability
    wing.control_surfaces.append(ControlSurface(
        name="elevon", kind="elevon", span_start=0.50, span_end=0.90,
        chord_fraction=0.25, symmetric=False,
    ))
    return ParametricAirframe(
        name="flying-wing",
        wing=wing,
        fins=generate_wing_fins(wing, [FinSpec(span_fraction=0.55, height=0.10)]),
        fuselages=generate_fuselages("pod", wing, htail=None, tail_arm=0.45, tail_z=0.0),
    )


def blended_wing_body() -> ParametricAirframe:
    """Tailless BWB: thick airfoil-bodied center sections blending into the outer wing, elevons,
    winglets acting as fins — no separate fuselage at all (the body IS wing sections)."""
    from uavaero.geometry.bwb import CenterBody, bwb_sections

    wing = _wing(winglet=True)
    wing.sections[1].sweep_deg = 22.0
    wing.sections = bwb_sections(
        CenterBody(center_chord=0.85, edge_chord=0.55, halfspan=0.20, blend_span=0.15,
                   sweep_deg=42.0, airfoil="naca0018"),
        wing.sections,
    )
    wing.control_surfaces.append(ControlSurface(
        name="elevon", kind="elevon", span_start=0.55, span_end=0.92,
        chord_fraction=0.25, symmetric=False,
    ))
    return ParametricAirframe(name="blended-wing-body", wing=wing)


def five_config_matrix() -> list[ParametricAirframe]:
    return [conventional(), vee_tail(), conventional_winglet(), vee_tail_fuselage(), full_configuration()]


# --- The three former OpenVSP bug cases (Phase 1 exit criteria 4 and 7) ----------------------


def vertical_tail_case() -> ParametricAirframe:
    """Bug (a): a vertical tail built as a 90-degree-rotated WING used to hard-crash the mesher
    when the geom's planar-symmetry flag stayed on (two superimposed mirrored tails)."""
    return ParametricAirframe(name="bug-vertical-tail", wing=_wing(), vtail=_vtail())


def dihedral_twist_case() -> ParametricAirframe:
    """Bug (b): dihedral and twist together on the same section used to degenerate the mesh
    (0 Kutta nodes, all-zero coefficients)."""
    return ParametricAirframe(name="bug-dihedral-twist", wing=_wing(dihedral_deg=6.0, twist_deg=-4.0))


def thin_tail_case() -> ParametricAirframe:
    """Bug (c): thin (~9% t/c) symmetric tail sections used to degenerate the mesh at small
    model scale."""
    return ParametricAirframe(name="bug-thin-tail", wing=_wing(), vtail=_vtail(airfoil="naca0009"))
