"""What the drawing draws.

The drawing is the centre of the interface, so what it contains is tested here rather than left to
be judged by eye in a screenshot. The widget's only remaining job is to fit these coordinates to a
rectangle and stroke them.

The property worth protecting: **a symmetric surface draws both halves**. `LiftingSurface.span()`
is one side, which is the standing trap in this codebase, and a caller that forgets it here draws
half an aeroplane — a mistake that is obvious in a picture and completely invisible in a number.
"""
from __future__ import annotations

import math

import pytest

from uavaero.geometry.authoring import TailSizing, build
from uavaero.geometry.outline import (
    bounds,
    fuselage_outline,
    mean_aerodynamic_chord_line,
    plan_outlines,
    surface_outline,
)
from uavaero.geometry.planform import Planform
from uavaero.schema.airframe import Fuselage, LiftingSurface, ParametricAirframe, WingSection


@pytest.fixture
def aircraft() -> ParametricAirframe:
    return build(
        "test",
        Planform(span=2.40, aspect_ratio=9.60, taper=0.667, sweep_mode="straight_qc",
                 dihedral_deg=3.0),
        htail=TailSizing(0.098, 2.50, 0.880, taper=0.75),
        vtail=TailSizing(0.034, 1.85, 0.880, taper=0.70),
    )


def test_a_symmetric_wing_draws_both_halves(aircraft):
    points = surface_outline(aircraft.wing)
    ys = [y for _, y in points]
    assert min(ys) < 0 < max(ys)
    assert min(ys) == pytest.approx(-max(ys)), "the two halves must be mirror images"


def test_a_flat_wing_draws_its_whole_span(aircraft):
    """The trap, stated as an assertion. `span()` is one side; the drawing shows both."""
    flat = build("flat", Planform(span=2.40, aspect_ratio=9.60, taper=0.667, dihedral_deg=0.0))
    ys = [y for _, y in surface_outline(flat.wing)]
    assert max(ys) - min(ys) == pytest.approx(2.0 * flat.wing.span(), rel=1e-9)
    assert max(ys) - min(ys) == pytest.approx(2.40, rel=1e-9)


def test_dihedral_foreshortens_the_drawn_span(aircraft):
    """This is a true plan view, not a schematic: a wing with dihedral projects narrower than its
    span by exactly cos(dihedral). Drawing the full span regardless would be a diagram of an
    aircraft rather than a view of one, and the difference shows up the moment anyone measures a
    tip position off the screen.
    """
    ys = [y for _, y in surface_outline(aircraft.wing)]
    projected = 2.0 * aircraft.wing.span() * math.cos(math.radians(3.0))
    assert max(ys) - min(ys) == pytest.approx(projected, rel=1e-9)
    assert max(ys) - min(ys) < 2.0 * aircraft.wing.span()


def test_a_single_sided_surface_is_not_mirrored(aircraft):
    """A fin has one half. Mirroring it would draw a second fin that does not exist."""
    points = surface_outline(aircraft.vtail)
    ys = [y for _, y in points]
    assert max(ys) == pytest.approx(min(ys), abs=1e-9), "a fin projects to a sliver in plan"


def test_a_vertical_surface_still_has_streamwise_extent(aircraft):
    """Its height is discarded by the projection, but its chord is not — a fin that collapsed to a
    point would vanish from the drawing entirely."""
    xs = [x for x, _ in surface_outline(aircraft.vtail)]
    assert max(xs) - min(xs) > 0.05


def test_the_outline_is_closed_and_has_no_repeated_first_point(aircraft):
    """The painter closes the polygon itself. A duplicated first point draws a zero-length segment
    that shows up as a blob at the root on a thick pen."""
    points = surface_outline(aircraft.wing)
    assert points[0] != points[-1]
    assert len(points) >= 4


def test_a_surface_with_one_station_draws_nothing(aircraft):
    """Rather than raising. A half-built aircraft has to be drawable, because you are looking at
    the drawing while you build it."""
    stub = LiftingSurface(name="stub", sections=[WingSection(chord=0.3)])
    assert surface_outline(stub) == []


def test_every_part_of_the_aircraft_is_drawn(aircraft):
    kinds = [o.kind for o in plan_outlines(aircraft)]
    assert kinds.count("wing") == 1
    assert kinds.count("htail") == 1
    assert kinds.count("vtail") == 1
    assert kinds.count("fuselage") >= 1


def test_bodies_are_drawn_before_surfaces(aircraft):
    """A wing root sits over the fuselage. Drawing the body last would put it on top of the wing."""
    kinds = [o.kind for o in plan_outlines(aircraft)]
    assert kinds.index("fuselage") < kinds.index("wing")


def test_a_fuselage_is_symmetric_about_its_own_centreline():
    body = Fuselage(name="boom", origin=(0.0, 0.35, 0.0), stations=[])
    from uavaero.schema.airframe import FuselageStation

    body.stations = [FuselageStation(x=0.0, width=0.02, height=0.02),
                     FuselageStation(x=0.5, width=0.10, height=0.10),
                     FuselageStation(x=1.0, width=0.02, height=0.02)]
    ys = [y for _, y in fuselage_outline(body)]
    assert max(ys) == pytest.approx(0.35 + 0.05)
    assert min(ys) == pytest.approx(0.35 - 0.05), "an offset boom stays centred on its own origin"


def test_bounds_cover_every_part(aircraft):
    outlines = plan_outlines(aircraft)
    x_min, x_max, y_min, y_max = bounds(outlines)
    for outline in outlines:
        for x, y in outline.points:
            assert x_min <= x <= x_max
            assert y_min <= y <= y_max


def test_bounds_of_nothing_are_a_unit_box_rather_than_degenerate():
    """The view divides by these to fit. A zero extent would be a division by zero on an empty
    document, which is the state the app starts in."""
    x_min, x_max, y_min, y_max = bounds([])
    assert x_max > x_min
    assert y_max > y_min


def test_the_mac_line_is_one_chord_long_at_its_own_station(aircraft):
    """It is drawn because the aerodynamic centre — a quarter along it — is what every tail arm in
    the tool is measured from."""
    line = mean_aerodynamic_chord_line(aircraft.wing)
    assert len(line) == 2
    (x0, y0), (x1, y1) = line
    assert y0 == pytest.approx(y1), "the MAC is streamwise"
    assert x1 - x0 == pytest.approx(aircraft.wing.mean_aerodynamic_chord())
    assert (x0, y0) == pytest.approx(aircraft.wing.mac_leading_edge()[:2])
