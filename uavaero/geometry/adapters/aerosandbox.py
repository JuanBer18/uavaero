"""ParametricAirframe -> asb.Airplane. Stateless, re-run from scratch on every spec change
(CLAUDE.md rule 2): the spec is authoritative, this is a mechanical export.

Feeds both Tier-2 solvers for free (AeroSandbox native VLM and its asb.AVL wrapper consume the
same Airplane) and the live viewport loop via mesh_preview.py.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from uavaero.geometry.airfoils import get_asb_airfoil
from uavaero.schema.airframe import Fuselage, LiftingSurface, ParametricAirframe

if TYPE_CHECKING:  # pragma: no cover
    import aerosandbox as asb


def to_airplane(
    airframe: ParametricAirframe,
    xyz_ref: tuple[float, float, float] | None = None,
) -> "asb.Airplane":
    """`xyz_ref` is the moment reference point (defaults to AeroSandbox's own [0,0,0]) — pass an
    arbitrary candidate CG to compute derivatives/moments about it (core/studies/cg_sizing.py).
    `s_ref`/`c_ref`/`b_ref` are passed explicitly from the main wing rather than left to
    AeroSandbox's implicit "first wing in the list" default, so nondimensionalization never
    silently depends on `lifting_surfaces()`'s ordering."""
    import aerosandbox as asb  # lazy: keeps uavaero.geometry importable without aerosandbox

    return asb.Airplane(
        name=airframe.name,
        xyz_ref=list(xyz_ref) if xyz_ref is not None else None,
        s_ref=airframe.wing.area(),
        c_ref=airframe.wing.mean_aerodynamic_chord(),
        b_ref=airframe.wing.span() * (2.0 if airframe.wing.symmetric else 1.0),
        wings=[_surface_to_wing(surface) for surface in airframe.lifting_surfaces()],
        fuselages=[_fuselage_to_asb(fuselage) for fuselage in airframe.fuselages],
    )


def _surface_to_wing(surface: LiftingSurface) -> "asb.Wing":
    import aerosandbox as asb

    # The spatial layout (sweep/dihedral/vertical) is entirely encoded by the shared LE walk —
    # AeroSandbox only ever sees explicit xyz_le stations, so it cannot disagree with other adapters.
    panel_cs = _control_surfaces_per_panel(surface)
    return asb.Wing(
        name=surface.name,
        symmetric=surface.symmetric,
        xsecs=[
            asb.WingXSec(
                xyz_le=list(le_point),
                chord=section.chord,
                twist=section.twist_deg,
                airfoil=get_asb_airfoil(section.airfoil),
                control_surfaces=panel_cs[i],
            )
            for i, (section, le_point) in enumerate(zip(surface.sections, surface.le_points()))
        ],
    )


def _control_surfaces_per_panel(surface: LiftingSurface) -> list[list]:
    """Map the spec's span-fraction control surfaces onto asb's per-panel model: an
    asb.ControlSurface on WingXSec i acts over panel i -> i+1, so each spec surface is attached
    to every panel its span range overlaps. Deflections stay 0 here — set at analysis time
    (Phase 2+), not part of the geometry."""
    import aerosandbox as asb

    per_panel: list[list] = [[] for _ in surface.sections]
    if not surface.control_surfaces:
        return per_panel

    fractions = surface.section_span_fractions()
    for cs in surface.control_surfaces:
        for i in range(len(surface.sections) - 1):
            overlap = min(cs.span_end, fractions[i + 1]) - max(cs.span_start, fractions[i])
            if overlap > 1e-6:
                per_panel[i].append(
                    asb.ControlSurface(
                        name=cs.name,
                        symmetric=cs.symmetric,
                        hinge_point=1.0 - cs.chord_fraction,
                    )
                )
    return per_panel


def _fuselage_to_asb(fuselage: Fuselage) -> "asb.Fuselage":
    import aerosandbox as asb

    ox, oy, oz = fuselage.origin
    return asb.Fuselage(
        name=fuselage.name,
        xsecs=[
            asb.FuselageXSec(
                xyz_c=[ox + station.x, oy, oz + station.z],
                width=station.width,
                height=station.height,
            )
            for station in fuselage.stations
        ],
    )
