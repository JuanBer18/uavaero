"""`ParametricAirframe` -> a surface mesh, for the 3D viewport.

Another stateless translation of the one parametrisation (rule 2), alongside the OpenVSP and
AeroSandbox adapters. It goes through `to_airplane` rather than walking the sections again, so the
shape you rotate on screen is built from the same geometry the solvers are handed — a second mesh
builder would be a second thing to keep in step, and the first time it drifted the picture would
stop being evidence.

Resolutions are arguments, never part of the spec: the viewport wants a picture in under a tenth of
a second, the `.vsp3` export wants fidelity, and those are run settings rather than properties of
the aircraft.

No Qt here. This module is importable and testable with no event loop and no display, which is what
lets the mesh be checked without a graphics context.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from uavaero.geometry.adapters.aerosandbox import to_airplane
from uavaero.schema.airframe import ParametricAirframe

if TYPE_CHECKING:  # pragma: no cover
    import pyvista as pv

#: Preview quality. The viewport redraws on every field edit, so this is tuned for latency.
PREVIEW_CHORDWISE = 20
PREVIEW_TANGENTIAL = 24


def _to_polydata(points, faces) -> "pv.PolyData":
    """AeroSandbox hands back (points, quad indices); PyVista wants a flat count-prefixed array."""
    import numpy as np
    import pyvista as pv

    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if faces.size == 0:
        return pv.PolyData(points)
    prefixed = np.hstack(
        [np.full((faces.shape[0], 1), faces.shape[1], dtype=int), faces]
    ).ravel()
    return pv.PolyData(points, prefixed)


def airframe_to_polydata(
    airframe: ParametricAirframe,
    chordwise_resolution: int = PREVIEW_CHORDWISE,
    fuselage_tangential_resolution: int = PREVIEW_TANGENTIAL,
) -> "pv.PolyData":
    """One merged surface mesh for the whole aircraft.

    Spanwise panels follow the spec's own stations - AeroSandbox meshes panel-per-section - so the
    picture shows the wing you authored rather than a resampled approximation of it.
    """
    import pyvista as pv

    airplane = to_airplane(airframe)
    meshes = []
    for wing in airplane.wings:
        points, faces = wing.mesh_body(
            method="quad", chordwise_resolution=chordwise_resolution
        )
        meshes.append(_to_polydata(points, faces))
    for fuselage in airplane.fuselages:
        points, faces = fuselage.mesh_body(
            method="quad", tangential_resolution=fuselage_tangential_resolution
        )
        meshes.append(_to_polydata(points, faces))

    if not meshes:
        return pv.PolyData()
    merged = meshes[0]
    if len(meshes) > 1:
        merged = merged.merge(meshes[1:])
    return merged
