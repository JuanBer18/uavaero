"""Automatic fuselage generator: a configuration choice -> ready-made Fuselage bodies, sized from
the wing and tail the airframe already has. Like the planform generator, this is a CONVENIENCE
GENERATOR (CLAUDE.md rule 2): it emits ordinary Fuselage/FuselageStation spec data, which remains
the single source of truth — nothing here is a second parametrization.

Styles (FUSELAGE_STYLES):
- conventional   one body, nose ahead of the wing, tail cone rising to meet the tail.
- pod            short pod wrapped around the wing only — the tailless/flying-wing body.
- pod_and_boom   short pod around the wing + one slim central boom out to the tail.
- twin_boom      pod + two booms that ARRIVE AT THE H-TAIL TIPS (y = htail semispan) — the
                 classic UAS Challenge layout; the booms follow the tail height automatically.
- twin_fuselage  two full bodies at the h-tail tips, no center pod.
- flying_boat    conventional body with a flat-bottomed, deeper hull (keel drop, stepped z).
- none           no body at all.

(A blended-wing-body preset was deliberately removed: a real BWB needs its own design tools —
chord/thickness blending controls, not a one-click body. The tailless configuration in the UI is
where that work will land.)

Everything is sized heuristically from the wing root chord, span, tail arm, and tail height, with
one overall `scale` knob. High/mid/low wing position shifts the body vertically relative to the
wing root LE (which sits at z = 0 in airframe coordinates).
"""
from __future__ import annotations

from uavaero.schema.airframe import Fuselage, FuselageStation, LiftingSurface

FUSELAGE_STYLES = [
    "conventional",
    "pod",
    "pod_and_boom",
    "twin_boom",
    "twin_fuselage",
    "flying_boat",
    "none",
]

#: Styles that make sense without a tail (offered in the tailless configuration).
TAILLESS_STYLES = ["conventional", "pod", "none"]

WING_POSITIONS = ["high", "mid", "low"]

_MIN_DIM = 0.008  # smallest station width/height the mesher and CAD loft stay happy with


def generate_fuselages(
    style: str,
    wing: LiftingSurface,
    wing_position: str = "mid",
    tail_arm: float = 0.9,
    tail_z: float = 0.05,
    htail: LiftingSurface | None = None,
    scale: float = 1.0,
) -> list[Fuselage]:
    """Bodies for the chosen style, sized from the airframe. `tail_arm`/`tail_z` are the tail
    origin relative to the wing root LE; `htail` supplies the semispan the twin styles attach to
    and the root chord the bodies extend under."""
    if style not in FUSELAGE_STYLES:
        raise ValueError(f"Unknown fuselage style '{style}' (valid: {', '.join(FUSELAGE_STYLES)})")
    if wing_position not in WING_POSITIONS:
        raise ValueError(f"Unknown wing position '{wing_position}' (valid: {', '.join(WING_POSITIONS)})")
    if style == "none":
        return []

    chord = wing.sections[0].chord if wing.sections else 0.25
    tail_chord = htail.sections[0].chord if htail and htail.sections else 0.5 * chord
    tail_semispan = htail.span() if htail else 0.35
    tail_end = tail_arm + tail_chord

    width = 0.35 * chord * scale
    height = 0.42 * chord * scale
    # body centerline z relative to the wing root LE (z=0): high wing = body hangs below, etc.
    z_center = {"high": -0.45 * height, "mid": 0.0, "low": +0.45 * height}[wing_position]

    if style == "conventional":
        return [_tapered_body("fuselage", -0.7 * chord * scale, tail_end, width, height,
                              z_center, tail_z)]
    if style == "pod":
        return [_pod("pod", chord, width, height, z_center, scale)]
    if style == "pod_and_boom":
        return [
            _pod("pod", chord, width, height, z_center, scale),
            _boom("boom", 0.0, 0.9 * chord, tail_end, 0.13 * chord * scale, z_center, tail_z),
        ]
    if style == "twin_boom":
        return [
            _pod("pod", chord, width, height, z_center, scale),
            _boom("boom_right", +tail_semispan, 0.25 * chord, tail_end, 0.13 * chord * scale, 0.0, tail_z),
            _boom("boom_left", -tail_semispan, 0.25 * chord, tail_end, 0.13 * chord * scale, 0.0, tail_z),
        ]
    if style == "twin_fuselage":
        return [
            _tapered_body("fuselage_right", -0.7 * chord * scale, tail_end,
                          0.8 * width, 0.8 * height, z_center, tail_z, y=+tail_semispan),
            _tapered_body("fuselage_left", -0.7 * chord * scale, tail_end,
                          0.8 * width, 0.8 * height, z_center, tail_z, y=-tail_semispan),
        ]
    if style == "flying_boat":
        return [_flying_boat_hull(chord, tail_end, width, height, z_center, tail_z, scale)]
    raise AssertionError(style)


# --- body shapes ------------------------------------------------------------------------------


def _tapered_body(
    name: str, x_nose: float, x_tail: float, width: float, height: float,
    z_center: float, tail_z: float, y: float = 0.0,
) -> Fuselage:
    """Classic single body: rounded nose ramp, constant midsection over the wing, tail cone whose
    centerline rises to the tail height so the tail mounts onto it."""
    length = x_tail - x_nose

    def x(fraction: float) -> float:
        return x_nose + fraction * length

    def z(fraction: float, blend_start: float = 0.55) -> float:
        if fraction <= blend_start:
            return z_center
        t = (fraction - blend_start) / (1.0 - blend_start)
        return z_center + t * (tail_z - z_center)

    stations = [
        FuselageStation(x=x(0.00), width=_MIN_DIM, height=_MIN_DIM, z=z_center),
        FuselageStation(x=x(0.10), width=0.75 * width, height=0.80 * height, z=z_center),
        FuselageStation(x=x(0.30), width=width, height=height, z=z_center),
        FuselageStation(x=x(0.55), width=0.90 * width, height=0.92 * height, z=z(0.55)),
        FuselageStation(x=x(0.80), width=0.45 * width, height=0.50 * height, z=z(0.80)),
        FuselageStation(x=x(1.00), width=_MIN_DIM, height=_MIN_DIM, z=z(1.00)),
    ]
    return Fuselage(name=name, origin=(0.0, y, 0.0), stations=stations)


def _pod(name: str, chord: float, width: float, height: float, z_center: float, scale: float) -> Fuselage:
    """Short pod wrapped around the wing: nose ahead of the LE, closing not far behind the TE."""
    x_nose, x_tail = -0.7 * chord * scale, 1.6 * chord * scale
    length = x_tail - x_nose
    stations = [
        FuselageStation(x=x_nose, width=_MIN_DIM, height=_MIN_DIM, z=z_center),
        FuselageStation(x=x_nose + 0.15 * length, width=0.8 * width, height=0.85 * height, z=z_center),
        FuselageStation(x=x_nose + 0.40 * length, width=width, height=height, z=z_center),
        FuselageStation(x=x_nose + 0.75 * length, width=0.55 * width, height=0.6 * height, z=z_center),
        FuselageStation(x=x_tail, width=_MIN_DIM, height=_MIN_DIM, z=z_center),
    ]
    return Fuselage(name=name, origin=(0.0, 0.0, 0.0), stations=stations)


def _boom(
    name: str, y: float, x_start: float, x_end: float, diameter: float,
    z_start: float, tail_z: float,
) -> Fuselage:
    """Slim constant-diameter boom whose centerline runs from the wing/pod height to the tail
    height — automatically following wherever the tail is placed."""
    diameter = max(diameter, _MIN_DIM)
    stations = [
        FuselageStation(x=x_start, width=_MIN_DIM, height=_MIN_DIM, z=z_start),
        FuselageStation(x=x_start + 0.08 * (x_end - x_start), width=diameter, height=diameter, z=z_start),
        FuselageStation(x=x_start + 0.60 * (x_end - x_start), width=diameter, height=diameter,
                        z=z_start + 0.6 * (tail_z - z_start)),
        FuselageStation(x=x_end - 0.05 * (x_end - x_start), width=0.8 * diameter, height=0.8 * diameter,
                        z=tail_z),
        FuselageStation(x=x_end, width=_MIN_DIM, height=_MIN_DIM, z=tail_z),
    ]
    return Fuselage(name=name, origin=(0.0, y, 0.0), stations=stations)


def _flying_boat_hull(
    chord: float, tail_end: float, width: float, height: float,
    z_center: float, tail_z: float, scale: float,
) -> Fuselage:
    """Deeper hull with a dropped keel amidships and a stepped rise aft — the seaplane shape."""
    x_nose = -0.8 * chord * scale
    length = tail_end - x_nose
    hull_height = 1.35 * height
    keel_drop = -0.25 * hull_height

    def x(fraction: float) -> float:
        return x_nose + fraction * length

    stations = [
        FuselageStation(x=x(0.00), width=_MIN_DIM, height=_MIN_DIM, z=z_center),
        FuselageStation(x=x(0.12), width=0.85 * width, height=hull_height, z=z_center + keel_drop / 2),
        FuselageStation(x=x(0.38), width=width, height=hull_height, z=z_center + keel_drop),
        FuselageStation(x=x(0.55), width=0.9 * width, height=0.9 * hull_height, z=z_center + keel_drop / 2),  # step
        FuselageStation(x=x(0.80), width=0.5 * width, height=0.5 * height, z=z_center + 0.7 * (tail_z - z_center)),
        FuselageStation(x=x(1.00), width=_MIN_DIM, height=_MIN_DIM, z=tail_z),
    ]
    return Fuselage(name="hull", origin=(0.0, 0.0, 0.0), stations=stations)
