"""Plan-view outlines of an airframe, as plain coordinate lists.

The drawing is the centre of the interface, so what it draws has to be testable without a window
open. This module turns a `ParametricAirframe` into closed polygons in aircraft coordinates; the Qt
widget's only job is to fit them to a rectangle and stroke them.

Coordinates are (x, y) in metres — x streamwise, positive aft; y spanwise, positive starboard.
Height is discarded, because that is what a plan view is. A vertical surface therefore projects to
a narrow sliver at its own spanwise station, which is correct and is how it appears on a drawing.

A surface is drawn from the leading-edge walk it already owns (`LiftingSurface.le_points`) plus the
chord at each station, so the outline can never disagree with what the solver is given.
"""
from __future__ import annotations

from dataclasses import dataclass

from uavaero.schema.airframe import Fuselage, LiftingSurface, ParametricAirframe

Point = tuple[float, float]


@dataclass(frozen=True)
class Outline:
    """One closed polygon and what it came from."""

    name: str
    kind: str  # "wing" | "htail" | "vtail" | "fin" | "fuselage"
    points: list[Point]


def surface_outline(surface: LiftingSurface) -> list[Point]:
    """Leading edge outboard, trailing edge back, mirrored when the surface is symmetric.

    The mirror is applied here rather than left to the caller because `LiftingSurface.span()` being
    one side is a standing trap in this codebase — a caller that forgets draws half an aeroplane.
    """
    stations = surface.le_points()
    chords = [section.chord for section in surface.sections]
    if len(stations) < 2:
        return []

    leading = [(x, y) for x, y, _z in stations]
    trailing = [(x + chord, y) for (x, y, _z), chord in zip(stations, chords)]

    if not surface.symmetric:
        return leading + list(reversed(trailing))

    # Root outboard to starboard tip, back along the trailing edge, then the mirror image.
    mirrored_leading = [(x, -y) for x, y in leading]
    mirrored_trailing = [(x, -y) for x, y in trailing]
    return (
        list(reversed(mirrored_leading))
        + leading[1:]
        + list(reversed(trailing[1:]))
        + mirrored_trailing
    )


def fuselage_outline(fuselage: Fuselage) -> list[Point]:
    """The body's plan silhouette: half-widths down one side and back up the other."""
    if len(fuselage.stations) < 2:
        return []
    origin_x, origin_y, _ = fuselage.origin
    starboard = [
        (origin_x + station.x, origin_y + station.width / 2.0) for station in fuselage.stations
    ]
    port = [
        (origin_x + station.x, origin_y - station.width / 2.0)
        for station in reversed(fuselage.stations)
    ]
    return starboard + port


def plan_outlines(airframe: ParametricAirframe) -> list[Outline]:
    """Every part of the aircraft, bodies first so surfaces draw over them."""
    outlines: list[Outline] = []
    for fuselage in airframe.fuselages:
        points = fuselage_outline(fuselage)
        if points:
            outlines.append(Outline(fuselage.name, "fuselage", points))

    for surface, kind in _surfaces_with_kind(airframe):
        points = surface_outline(surface)
        if points:
            outlines.append(Outline(surface.name, kind, points))
    return outlines


def _surfaces_with_kind(airframe: ParametricAirframe):
    if airframe.wing is not None:
        yield airframe.wing, "wing"
    if airframe.htail is not None:
        yield airframe.htail, "htail"
    if airframe.vtail is not None:
        yield airframe.vtail, "vtail"
    for fin in airframe.fins:
        yield fin, "fin"


def bounds(outlines: list[Outline]) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) over everything. A zero-extent aircraft returns a unit box
    rather than a degenerate one, so a caller scaling to fit cannot divide by zero."""
    points = [point for outline in outlines for point in outline.points]
    if not points:
        return 0.0, 1.0, -0.5, 0.5
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    if x_max - x_min <= 0:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_max - y_min <= 0:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    return x_min, x_max, y_min, y_max


def mean_aerodynamic_chord_line(surface: LiftingSurface) -> list[Point]:
    """The MAC drawn where it actually sits: leading edge to trailing edge at its spanwise station.

    Shown because the aerodynamic centre is a quarter of the way along it, and every tail arm in
    the tool is measured from there — so the one line that explains the tail volumes is on the
    drawing rather than only in a readout.
    """
    mac = surface.mean_aerodynamic_chord()
    if mac <= 0:
        return []
    x, y, _z = surface.mac_leading_edge()
    return [(x, y), (x + mac, y)]
