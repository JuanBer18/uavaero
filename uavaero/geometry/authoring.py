"""Building a whole aircraft out of the numbers an engineer actually quotes.

`ParametricAirframe` remains the only description of the aircraft — this module never introduces a
second one. What it adds is the *forward and inverse translation* between the spec and the handful
of variables you would write on a napkin: a wing quoted as span and aspect ratio, tails quoted as
area and arm, a fuselage quoted as a style.

That is the same arrangement `planform.py` already uses for a single surface, extended to the whole
machine. `to_surface`/`infer_planform` round-trip; so do the functions here. Nothing is stored in
napkin form, so the two can never disagree.

**Tails are quoted as area and arm because that is what determines whether they work.** Tail volume
— area times arm, against the wing — is the number that says whether an aircraft is controllable,
and it is invisible if you are typing root chords and station positions. So the authoring direction
takes area and arm and works out where the surface goes; the inverse reads the arm back off the
geometry.

Arm is measured aerodynamic centre to aerodynamic centre — wing MAC quarter-chord to tail MAC
quarter-chord — which is the convention the tail-volume coefficients below assume.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from uavaero.geometry.fuselage_gen import FUSELAGE_STYLES, WING_POSITIONS, generate_fuselages
from uavaero.geometry.planform import Planform, to_surface
from uavaero.schema.airframe import LiftingSurface, ParametricAirframe

#: Conventional bands for the tail volume coefficients, for the "is this sane" readout. These are
#: not constraints and nothing refuses a value outside them — a canard or a flying wing lives
#: outside on purpose. They exist so a number that is an order of magnitude wrong is visible.
VH_TYPICAL = (0.40, 0.80)
VV_TYPICAL = (0.020, 0.050)


@dataclass(frozen=True)
class TailSizing:
    """A tail the way a tail is quoted: how much area, how far back.

    `arm_m` is aerodynamic centre to aerodynamic centre. `aspect_ratio` and `taper` shape the
    surface; `incidence_deg` sets both stations' twist, which for a tail is the whole point of
    incidence rather than a washout distribution.
    """

    area_m2: float
    aspect_ratio: float
    arm_m: float
    taper: float = 1.0
    incidence_deg: float = 0.0
    airfoil: str = "naca0012"
    sweep_deg: float = 0.0
    dihedral_deg: float = 0.0
    #: Height of the surface's root above the aircraft datum (z = 0), not above the wing. That is
    #: what `_placed` writes into `origin[2]`, and what `infer_tail_sizing` reads back out of it.
    #: A T-tail is a large positive value; a fin sits a little above the boom.
    z_m: float = 0.0

    def planform(self, *, vertical: bool) -> Planform:
        """The surface's own shape, independent of where it ends up sitting.

        A fin is single-sided, so its `span` is its height and its area is not doubled — which is
        also how fin area is quoted, and why `Vv` below divides by wing span rather than MAC.
        """
        span = (self.area_m2 * self.aspect_ratio) ** 0.5
        if vertical:
            # For a single-sided surface AR is conventionally b^2/S with b the height, so the
            # span that satisfies it is the same expression; symmetric=False stops the area
            # being doubled underneath.
            return Planform(
                span=span, area=self.area_m2, taper=self.taper, symmetric=False,
                sweep_mode="manual", sweep_deg=self.sweep_deg,
                airfoil_root=self.airfoil, airfoil_tip=self.airfoil,
            )
        return Planform(
            span=span, area=self.area_m2, taper=self.taper, symmetric=True,
            sweep_mode="manual", sweep_deg=self.sweep_deg, dihedral_deg=self.dihedral_deg,
            airfoil_root=self.airfoil, airfoil_tip=self.airfoil,
        )


def aerodynamic_centre_x(surface: LiftingSurface) -> float:
    """Streamwise position of the surface's MAC quarter-chord.

    The single quantity every arm and every tail volume in this module is measured between. It is
    computed from the surface's own MAC rather than assumed at the root, which matters the moment
    a surface is swept.
    """
    return surface.mac_leading_edge()[0] + 0.25 * surface.mean_aerodynamic_chord()


def tail_arm(airframe: ParametricAirframe, surface: LiftingSurface) -> float:
    """Aerodynamic centre to aerodynamic centre, wing to this surface."""
    return aerodynamic_centre_x(surface) - aerodynamic_centre_x(airframe.wing)


def horizontal_tail_volume(airframe: ParametricAirframe) -> float | None:
    """Vh = S_h · l_h / (S_w · MAC_w). None when there is no horizontal tail."""
    wing, htail = airframe.wing, airframe.htail
    if htail is None:
        return None
    denominator = wing.area() * wing.mean_aerodynamic_chord()
    if denominator <= 0:
        return None
    return htail.area() * tail_arm(airframe, htail) / denominator


def vertical_tail_volume(airframe: ParametricAirframe) -> float | None:
    """Vv = S_v · l_v / (S_w · b_w), referenced to the FULL wing span.

    `LiftingSurface.span()` is one side, so it is doubled here. Getting that wrong doubles Vv and
    makes an under-finned aircraft look adequate — the failure is quiet and in the flattering
    direction, which is the worst kind.
    """
    wing, vtail = airframe.wing, airframe.vtail
    if vtail is None:
        return None
    full_span = wing.span() * (2.0 if wing.symmetric else 1.0)
    denominator = wing.area() * full_span
    if denominator <= 0:
        return None
    return vtail.area() * tail_arm(airframe, vtail) / denominator


def build(
    name: str,
    wing: Planform,
    htail: TailSizing | None = None,
    vtail: TailSizing | None = None,
    fuselage_style: str = "conventional",
    wing_position: str = "mid",
    fuselage_scale: float = 1.0,
    fins: list[LiftingSurface] | None = None,
) -> ParametricAirframe:
    """A complete aircraft from engineering variables.

    The tails are placed so that their *arms come out as asked*: each surface is shaped first, its
    own aerodynamic centre measured, and its origin then solved for. Placing by root leading edge
    instead — the obvious shortcut — makes the arm you get differ from the arm you asked for by a
    quarter of the tail chord, which is a few percent of Vh and drifts as you change tail taper.
    """
    if fuselage_style not in FUSELAGE_STYLES:
        raise ValueError(
            f"Unknown fuselage style '{fuselage_style}' (valid: {', '.join(FUSELAGE_STYLES)})"
        )
    if wing_position not in WING_POSITIONS:
        raise ValueError(
            f"Unknown wing position '{wing_position}' (valid: {', '.join(WING_POSITIONS)})"
        )

    wing_surface = to_surface(wing, name="wing", origin=(0.0, 0.0, 0.0))
    wing_ac = aerodynamic_centre_x(wing_surface)

    airframe = ParametricAirframe(name=name, wing=wing_surface, fins=list(fins or []))
    if htail is not None:
        airframe.htail = _placed(htail, "htail", wing_ac, vertical=False)
    if vtail is not None:
        airframe.vtail = _placed(vtail, "vtail", wing_ac, vertical=True)

    airframe.fuselages = generate_fuselages(
        style=fuselage_style,
        wing=wing_surface,
        wing_position=wing_position,
        tail_arm=htail.arm_m if htail is not None else 0.9,
        tail_z=htail.z_m if htail is not None else 0.05,
        htail=airframe.htail,
        scale=fuselage_scale,
    )
    return airframe


def _placed(sizing: TailSizing, name: str, wing_ac_x: float, *, vertical: bool) -> LiftingSurface:
    """Shape the surface at the origin, then move it so its own AC lands at the requested arm."""
    surface = to_surface(sizing.planform(vertical=vertical), name=name, vertical=vertical)
    local_ac = aerodynamic_centre_x(surface)  # measured from an origin of zero
    origin_x = wing_ac_x + sizing.arm_m - local_ac
    surface.origin = (origin_x, 0.0, sizing.z_m)
    if sizing.incidence_deg:
        surface.sections = [
            dataclasses.replace(section, twist_deg=sizing.incidence_deg)
            for section in surface.sections
        ]
    return surface


def infer_tail_sizing(
    airframe: ParametricAirframe, surface: LiftingSurface, *, vertical: bool
) -> TailSizing:
    """Read the napkin numbers back off a built surface — the inverse of `_placed`.

    This is what lets the editor show area and arm for a tail that was loaded from a file, or
    hand-edited station by station, without keeping a second copy of those numbers anywhere.
    """
    area = surface.area()
    span = surface.span() * (2.0 if surface.symmetric else 1.0)
    chords = [section.chord for section in surface.sections]
    taper = (chords[-1] / chords[0]) if chords and chords[0] > 0 else 1.0
    return TailSizing(
        area_m2=area,
        aspect_ratio=(span**2 / area) if area > 0 else 0.0,
        arm_m=tail_arm(airframe, surface),
        taper=taper,
        incidence_deg=surface.sections[0].twist_deg if surface.sections else 0.0,
        airfoil=surface.sections[0].airfoil if surface.sections else "naca0012",
        sweep_deg=surface.sections[-1].sweep_deg if len(surface.sections) > 1 else 0.0,
        dihedral_deg=surface.sections[-1].dihedral_deg if len(surface.sections) > 1 else 0.0,
        z_m=surface.origin[2],
    )


def summary(airframe: ParametricAirframe) -> dict[str, float | None]:
    """The derived block the screen shows: what the numbers you typed worked out to.

    Everything here is computed from the airframe every time it is asked for. None of it is stored,
    so none of it can go stale — which is the reason the screen can show them beside the inputs
    without a refresh step.
    """
    wing = airframe.wing
    chords = [section.chord for section in wing.sections]
    return {
        "wing_area_m2": wing.area(),
        "wing_span_m": wing.span() * (2.0 if wing.symmetric else 1.0),
        "wing_aspect_ratio": _aspect_ratio(wing),
        "root_chord_m": chords[0] if chords else 0.0,
        "tip_chord_m": chords[-1] if chords else 0.0,
        "mac_m": wing.mean_aerodynamic_chord(),
        "mac_le_x_m": wing.mac_leading_edge()[0],
        "aerodynamic_centre_x_m": aerodynamic_centre_x(wing),
        "htail_arm_m": tail_arm(airframe, airframe.htail) if airframe.htail else None,
        "vtail_arm_m": tail_arm(airframe, airframe.vtail) if airframe.vtail else None,
        "horizontal_tail_volume": horizontal_tail_volume(airframe),
        "vertical_tail_volume": vertical_tail_volume(airframe),
    }


def _aspect_ratio(surface: LiftingSurface) -> float:
    full_span = surface.span() * (2.0 if surface.symmetric else 1.0)
    area = surface.area()
    return full_span**2 / area if area > 0 else 0.0


# ════════ editing one part of an existing aircraft ══════════════════════════════════════════
#
# The screen edits a document. Rebuilding the whole aircraft from napkin numbers on every
# keystroke would work today and would silently discard hand-edited stations the moment a
# station editor exists, so each of these replaces exactly the surface it names and leaves
# everything else — including the other surfaces' arms — alone.
#
# They mutate in place. That is right for a document editor: the airframe *is* the thing being
# edited, and returning copies would only push the assignment back to every call site.


def apply_wing(airframe: ParametricAirframe, wing: Planform) -> None:
    """Reshape the wing, holding its origin.

    Both tails are then re-placed at the arms they currently have, because the arm is measured from
    the wing's aerodynamic centre — which has just moved. Without this, changing wing sweep or
    taper would quietly change every tail volume in the aircraft.
    """
    arms = {
        "htail": tail_arm(airframe, airframe.htail) if airframe.htail else None,
        "vtail": tail_arm(airframe, airframe.vtail) if airframe.vtail else None,
    }
    origin = airframe.wing.origin
    airframe.wing = to_surface(wing, name="wing", origin=origin)

    wing_ac = aerodynamic_centre_x(airframe.wing)
    for name, arm in arms.items():
        surface = getattr(airframe, name)
        if surface is None or arm is None:
            continue
        local_ac = aerodynamic_centre_x(surface) - surface.origin[0]
        surface.origin = (wing_ac + arm - local_ac, surface.origin[1], surface.origin[2])


def apply_tail(airframe: ParametricAirframe, sizing: TailSizing, which: str) -> None:
    """Replace `htail` or `vtail` from napkin numbers, placed at the arm asked for."""
    if which not in ("htail", "vtail"):
        raise ValueError(f"which must be 'htail' or 'vtail', got {which!r}")
    vertical = which == "vtail"
    surface = _placed(sizing, which, aerodynamic_centre_x(airframe.wing), vertical=vertical)
    setattr(airframe, which, surface)


def apply_fuselage(
    airframe: ParametricAirframe,
    style: str,
    wing_position: str = "mid",
    scale: float = 1.0,
) -> None:
    """Regenerate the bodies for the current wing and tail."""
    if style not in FUSELAGE_STYLES:
        raise ValueError(
            f"Unknown fuselage style '{style}' (valid: {', '.join(FUSELAGE_STYLES)})"
        )
    airframe.fuselages = generate_fuselages(
        style=style,
        wing=airframe.wing,
        wing_position=wing_position,
        tail_arm=tail_arm(airframe, airframe.htail) if airframe.htail else 0.9,
        tail_z=airframe.htail.origin[2] if airframe.htail else 0.05,
        htail=airframe.htail,
        scale=scale,
    )


def surface_height(airframe: ParametricAirframe, which: str) -> float | None:
    """The surface's root height above the aircraft datum, or None when it is absent."""
    surface = airframe.wing if which == "wing" else getattr(airframe, which, None)
    return None if surface is None else float(surface.origin[2])


def set_surface_height(airframe: ParametricAirframe, which: str, z_m: float) -> None:
    """Move a lifting surface vertically, relative to the aircraft datum.

    Height is independent of everything else the editors set: the tail arm is measured along x, the
    planform is shaped in its own plane, and `apply_wing` / `apply_tail` both carry `origin[2]`
    through untouched. So this can set the origin directly rather than going back through a
    planform or a `TailSizing` — and a wing moved from low to high does not disturb the tails.
    """
    surface = airframe.wing if which == "wing" else getattr(airframe, which, None)
    if surface is None:
        raise ValueError(f"this aircraft has no {which} to move")
    x, y, _ = surface.origin
    surface.origin = (x, y, float(z_m))


def add_default_component(airframe: ParametricAirframe, kind: str) -> str:
    """Add one optional component and return the selection kind created.

    Defaults scale from the current wing so the Add Component action remains useful for both
    sub-metre models and large aircraft. A V-tail is deliberately an ``htail`` with large
    dihedral, not a separate geometry type.
    """
    facts = summary(airframe)
    wing_area = facts["wing_area_m2"] or 0.5
    mac = facts["mac_m"] or 0.25
    arm = max(2.5 * mac, 0.5)

    if kind in ("htail", "v_tail"):
        if airframe.htail is not None:
            raise ValueError("aircraft already has a horizontal or V-tail")
        if kind == "v_tail" and airframe.vtail is not None:
            raise ValueError("remove the vertical tail before adding a V-tail")
        apply_tail(airframe, TailSizing(
            area_m2=max(0.18 * wing_area, 0.02), aspect_ratio=3.0, arm_m=arm,
            taper=0.65, incidence_deg=0.0, airfoil="naca0012",
            dihedral_deg=38.0 if kind == "v_tail" else 0.0,
        ), "htail")
        return "htail"
    if kind == "vtail":
        if airframe.vtail is not None:
            raise ValueError("aircraft already has a vertical tail")
        apply_tail(airframe, TailSizing(
            area_m2=max(0.07 * wing_area, 0.01), aspect_ratio=1.7, arm_m=arm,
            taper=0.65, airfoil="naca0012",
        ), "vtail")
        return "vtail"
    if kind == "fuselage":
        if airframe.fuselages:
            raise ValueError("aircraft already has a fuselage configuration")
        apply_fuselage(airframe, "conventional")
        return "fuselage"
    raise ValueError(f"unknown optional component kind: {kind!r}")


#: Dihedral at or above this makes a horizontal tail a V-tail. Well clear of the few degrees a
#: conventional tailplane carries, and well below the 30-45 deg a real V-tail uses.
V_TAIL_DIHEDRAL_DEG = 15.0


def is_v_tail(airframe: ParametricAirframe) -> bool:
    """Whether the horizontal tail is really a V-tail.

    A V-tail is stored as one mirrored surface with large dihedral rather than as its own geometry
    type, so nothing downstream has a second configuration to handle. The cost of that choice is
    that it has to be *recognised* to be named, or the screen calls a V-tail a horizontal tail and
    the user reasonably concludes the add action did the wrong thing.
    """
    htail = airframe.htail
    if htail is None:
        return False
    return any(
        abs(getattr(section, "dihedral_deg", 0.0) or 0.0) >= V_TAIL_DIHEDRAL_DEG
        for section in htail.sections
    )


def remove_component(airframe: ParametricAirframe, kind: str) -> None:
    """Remove optional geometry. The main wing is mandatory in the current schema."""
    if kind == "fuselage":
        airframe.fuselages = []
    elif kind in ("htail", "vtail"):
        setattr(airframe, kind, None)
    else:
        raise ValueError(f"component {kind!r} cannot be removed")


def wing_planform(airframe: ParametricAirframe) -> Planform | None:
    """The wing's napkin numbers, read back off the geometry. None for a multi-panel wing, which
    has no single taper or sweep to report and must be edited station by station."""
    from uavaero.geometry.planform import infer_planform

    return infer_planform(airframe.wing)
