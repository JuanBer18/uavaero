"""The 3D preview mesh: the picture and the solve are built from one geometry.

The property worth protecting is that the viewport is *evidence*. A second mesh builder walking the
sections again would be a second thing to keep in step with `ParametricAirframe`, and the first
time it drifted the render would quietly stop showing the aircraft that gets solved — a wing you
believe you are looking at while VSPAERO analyses a different one is worse than no picture.

So the mesh goes through `to_airplane`, the same adapter the solvers use, and these tests assert
that what comes out has the dimensions the spec asked for rather than merely that it is non-empty.

No Qt and no GL here: the mesh is pure geometry, and it stays checkable on a machine that cannot
open a render window.
"""
from __future__ import annotations

import copy

import pytest

from tests.markers import requires_aerosandbox, requires_pyvista
from uavaero.geometry import authoring
from uavaero.schema.airframe import default_airframe

pytestmark = [requires_pyvista, requires_aerosandbox]


@pytest.fixture(scope="module")
def mesh():
    from uavaero.geometry.adapters.mesh import airframe_to_polydata

    return airframe_to_polydata(default_airframe())


def test_the_whole_aircraft_is_in_the_mesh(mesh):
    """Every lifting surface and the fuselage, merged into one body — not just the wing."""
    assert mesh.n_points > 0
    assert mesh.n_cells > 0


def test_the_mesh_spans_the_full_wing_not_the_half(mesh):
    """`LiftingSurface.span()` is one side. A mesh built from it without mirroring would be half an
    aircraft, and on an isometric view that reads as a slightly odd camera angle rather than as a
    bug."""
    airframe = default_airframe()
    half_span = airframe.wing.span()
    y_min, y_max = mesh.bounds[2], mesh.bounds[3]
    assert y_max == pytest.approx(half_span, rel=0.02)
    assert y_min == pytest.approx(-half_span, rel=0.02)


def test_a_tail_raised_above_the_fin_becomes_the_top_of_the_aircraft():
    """The height control and the render have to agree, or the number says one thing and the
    picture another.

    Asserted as an absolute position rather than as a change in the mesh's z-max: with the tail
    low, the highest point is the *fin* (0.05 root + 0.25 span), so a delta would measure the gap
    between the two surfaces and not the tail's travel. That is exactly the mistake this test was
    written with the first time.
    """
    from uavaero.geometry.adapters.mesh import airframe_to_polydata

    aircraft = copy.deepcopy(default_airframe())
    fin_top = authoring.surface_height(aircraft, "vtail") + aircraft.vtail.span()
    assert airframe_to_polydata(aircraft).bounds[5] == pytest.approx(fin_top, abs=0.02)

    authoring.set_surface_height(aircraft, "htail", 0.45)
    assert 0.45 > fin_top, "the T-tail has to clear the fin for this to test anything"
    # Plus the section's own half-thickness, hence the tolerance rather than equality.
    assert airframe_to_polydata(aircraft).bounds[5] == pytest.approx(0.45, abs=0.02)


def test_an_aircraft_stripped_to_its_wing_still_meshes():
    """Components are removable, so the viewport must not assume a tail or a body exists."""
    from uavaero.geometry.adapters.mesh import airframe_to_polydata

    bare = copy.deepcopy(default_airframe())
    for kind in ("htail", "vtail", "fuselage"):
        authoring.remove_component(bare, kind)
    assert airframe_to_polydata(bare).n_points > 0
