"""ParametricAirframe -> OpenVSP model. Stateless translation (CLAUDE.md rule 2); the three known
OpenVSP meshing bugs are fixed HERE, permanently, not worked around by the user (rule 4):

(a) Vertical tail via 90-degree X rotation with the WING geom's planar-symmetry flag still on
    creates two superimposed mirrored tails and hard-crashes the mesher -> every `vertical`
    surface gets `Sym_Planar_Flag = 0`, always.
(b) Dihedral + twist on the same section used to degenerate the panel mesh -> twist is applied
    about the leading edge (`Twist_Location = 0`) with a forced `Update()` between the rotation
    parm groups.
(c) Thin (~9% t/c) symmetric sections used to degenerate the mesh at small model scale -> wing
    surfaces get a raised tessellation (`Tess_W`, per-section `SectTess_U`) so thin sections keep
    enough mesh resolution. (Tessellation is a solver-run setting, so it lives here as build
    defaults, not in the spec.)

Porting trap inherited from AEROAPP: `vsp.Update()` is MANDATORY immediately after
`ChangeXSecShape(..., XS_FOUR_SERIES)` and before setting ThickChord/Camber — omitting it fails
silently ("Can't Find Parm") and degenerates the solve.

This module must only ever run inside the disposable worker process (`uavaero.geometry.adapters.vsp_worker`) —
OpenVSP state is process-global and its native crashes kill the process (CLAUDE.md rule 5).
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from uavaero.geometry.airfoils import parse_naca4, write_openvsp_dat
from uavaero.schema.airframe import Fuselage, LiftingSurface, ParametricAirframe

#: Chordwise/spanwise tessellation defaults — bug fix (c): generous enough that thin symmetric
#: tail sections at UAS model scale keep a healthy mesh.
DEFAULT_TESS_W = 33
DEFAULT_SECT_TESS_U = 12


@dataclass
class CompGeomResult:
    ok: bool
    total_wetted_area: float = 0.0
    total_theo_area: float = 0.0
    num_degen_tris_removed: int = 0
    total_num_tris: int = 0
    per_component: dict[str, float] = field(default_factory=dict)  # name -> wetted area
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "total_wetted_area": self.total_wetted_area,
            "total_theo_area": self.total_theo_area,
            "num_degen_tris_removed": self.num_degen_tris_removed,
            "total_num_tris": self.total_num_tris,
            "per_component": self.per_component,
            "messages": self.messages,
        }


def build_model(
    airframe: ParametricAirframe,
    tess_w: int = DEFAULT_TESS_W,
    sect_tess_u: int = DEFAULT_SECT_TESS_U,
) -> dict[str, str]:
    """Build the airframe into OpenVSP's (process-global) model from a clean slate.
    Returns {surface name: geom id}.

    The tessellation defaults are the measured ones from bug fix (c) and are what a
    normal solve should use. They are parameters only so the coarse-to-fine ladder can
    drop them for screening runs — going below the defaults is a deliberate accuracy
    trade, never a convenience."""
    import openvsp as vsp

    vsp.VSPRenew()
    geom_ids: dict[str, str] = {}
    for surface in airframe.lifting_surfaces():
        geom_ids[surface.name] = _build_lifting_surface(vsp, surface, tess_w, sect_tess_u)
    for fuselage in airframe.fuselages:
        geom_ids[fuselage.name] = _build_fuselage(vsp, fuselage)
    vsp.Update()
    return geom_ids


def write_vsp3(
    airframe: ParametricAirframe,
    path: Path,
    tess_w: int = DEFAULT_TESS_W,
    sect_tess_u: int = DEFAULT_SECT_TESS_U,
) -> dict[str, str]:
    import openvsp as vsp

    geom_ids = build_model(airframe, tess_w, sect_tess_u)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vsp.WriteVSPFile(str(path))
    return geom_ids


def run_compgeom(expected_components: int | None = None) -> CompGeomResult:
    """OpenVSP's CompGeom mesh check on the currently built model — the Phase-1 'geometry/mesh
    check' criterion. ok = nonzero triangles and wetted area overall and per component."""
    import openvsp as vsp

    vsp.SetAnalysisInputDefaults("CompGeom")
    vsp.SetIntAnalysisInput("CompGeom", "Set", [vsp.SET_ALL], 0)
    # No txt/csv sidecar files — results are read via the results API.
    vsp.SetIntAnalysisInput("CompGeom", "WriteCSVFlag", [0], 0)
    vsp.SetIntAnalysisInput("CompGeom", "WriteTXTFlag", [0], 0)
    rid = vsp.ExecAnalysis("CompGeom")

    names = list(vsp.GetStringResults(rid, "Comp_Name"))
    wet_areas = list(vsp.GetDoubleResults(rid, "Wet_Area"))
    result = CompGeomResult(
        ok=True,
        total_wetted_area=_first(vsp.GetDoubleResults(rid, "Total_Wet_Area")),
        total_theo_area=_first(vsp.GetDoubleResults(rid, "Total_Theo_Area")),
        num_degen_tris_removed=int(_first(vsp.GetIntResults(rid, "Num_Degen_Tris_Removed"))),
        total_num_tris=int(_first(vsp.GetIntResults(rid, "Total_Num_Tris"))),
        per_component=dict(zip(names, wet_areas)),
    )

    if result.total_num_tris <= 0:
        result.ok = False
        result.messages.append("CompGeom produced zero triangles — degenerate mesh")
    if result.total_wetted_area <= 0.0:
        result.ok = False
        result.messages.append("CompGeom total wetted area is zero — degenerate mesh")
    for name, area in result.per_component.items():
        if area <= 0.0:
            result.ok = False
            result.messages.append(f"Component '{name}' has zero wetted area")
    if expected_components is not None and len(names) != expected_components:
        result.ok = False
        result.messages.append(
            f"Expected {expected_components} mesh components, CompGeom found {len(names)}: {names}"
        )
    return result


def _first(values, default=0.0):
    return values[0] if values else default


# --- lifting surfaces -------------------------------------------------------------------------


def _build_lifting_surface(
    vsp, surface: LiftingSurface,
    tess_w: int = DEFAULT_TESS_W, sect_tess_u: int = DEFAULT_SECT_TESS_U,
) -> str:
    gid = vsp.AddGeom("WING", "")
    vsp.SetGeomName(gid, surface.name)

    n_panels = len(surface.sections) - 1
    if n_panels < 1:
        raise ValueError(f"Surface '{surface.name}' needs at least 2 sections, got {len(surface.sections)}")
    for _ in range(n_panels - 1):  # default WING already has one panel (XSec_1)
        vsp.InsertXSec(gid, 1, vsp.XS_FOUR_SERIES)
    vsp.Update()

    # Panels: span/chords/sweep/dihedral/twist per section, driven by SPAN+ROOTC+TIPC.
    # ORDER MATTERS (found empirically on 3.51.0): touching the XForm/Sym parms before
    # SetDriverGroup silently breaks the driver switch — Span/chord values then get recomputed
    # from the old drivers. Section drivers/parms first, placement/rotation/symmetry after.
    vsp.SetParmVal(gid, "Tess_W", "Shape", float(tess_w))  # bug fix (c)
    for i in range(1, len(surface.sections)):
        inner, outer = surface.sections[i - 1], surface.sections[i]
        group = f"XSec_{i}"
        vsp.SetDriverGroup(
            gid, i, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER
        )
        vsp.SetParmVal(gid, "Span", group, outer.span)
        vsp.SetParmVal(gid, "Root_Chord", group, inner.chord)
        vsp.SetParmVal(gid, "Tip_Chord", group, outer.chord)
        vsp.SetParmVal(gid, "Sweep", group, outer.sweep_deg)
        vsp.SetParmVal(gid, "Sweep_Location", group, 0.0)  # spec sweep is leading-edge sweep
        vsp.SetParmVal(gid, "Dihedral", group, outer.dihedral_deg)
        # Bug fix (b): twist about the leading edge, with an Update() between the rotation parm
        # groups (dihedral above, twist here) — the dihedral+twist combination used to degenerate
        # the mesh when both hit the same section in one update.
        vsp.Update()
        vsp.SetParmVal(gid, "Twist", group, outer.twist_deg)
        vsp.SetParmVal(gid, "Twist_Location", group, 0.0)
        vsp.SetParmVal(gid, "SectTess_U", group, float(sect_tess_u))  # bug fix (c)
        vsp.Update()

    # Root-station twist lives on XSec_0 (no panel parms there).
    _set_parm_if_exists(vsp, gid, "Twist", "XSec_0", surface.sections[0].twist_deg)
    _set_parm_if_exists(vsp, gid, "Twist_Location", "XSec_0", 0.0)
    vsp.Update()

    # Placement + symmetry — AFTER the driver-group work above (see ORDER MATTERS note).
    # Bug fix (a): a rotated (vertical) geom must NEVER keep planar symmetry on — that
    # superimposes two mirrored copies and hard-crashes the mesher.
    ox, oy, oz = surface.origin
    vsp.SetParmVal(gid, "X_Rel_Location", "XForm", ox)
    vsp.SetParmVal(gid, "Y_Rel_Location", "XForm", oy)
    vsp.SetParmVal(gid, "Z_Rel_Location", "XForm", oz)
    if surface.vertical:
        vsp.SetParmVal(gid, "X_Rel_Rotation", "XForm", 90.0)
        vsp.SetParmVal(gid, "Sym_Planar_Flag", "Sym", 0)
    elif not surface.symmetric:
        vsp.SetParmVal(gid, "Sym_Planar_Flag", "Sym", 0)
    vsp.Update()

    # Airfoils, one per station.
    xsurf = vsp.GetXSecSurf(gid, 0)
    for i, section in enumerate(surface.sections):
        _apply_airfoil(vsp, gid, xsurf, i, section.airfoil)
    vsp.Update()
    return gid


def _apply_airfoil(vsp, gid: str, xsurf: str, index: int, airfoil: str) -> None:
    naca = parse_naca4(airfoil)
    if naca is not None:
        m, p, tt = naca
        vsp.ChangeXSecShape(xsurf, index, vsp.XS_FOUR_SERIES)
        vsp.Update()  # MANDATORY before the parms below exist (silent-failure trap)
        group = f"XSecCurve_{index}"
        vsp.SetParmVal(gid, "ThickChord", group, tt)
        vsp.SetParmVal(gid, "Camber", group, m)
        if m > 0:
            vsp.SetParmVal(gid, "CamberLoc", group, p)
        return

    # File airfoil: TE-closed .dat (blunt trailing edges degenerate VSPAERO meshes at small scale).
    dat_path = Path(tempfile.gettempdir()) / "uavdesigner_airfoils" / f"{airfoil}.dat"
    if not dat_path.exists():
        write_openvsp_dat(airfoil, dat_path)
    vsp.ChangeXSecShape(xsurf, index, vsp.XS_FILE_AIRFOIL)
    vsp.Update()
    vsp.ReadFileAirfoil(vsp.GetXSec(xsurf, index), str(dat_path))


def _set_parm_if_exists(vsp, gid: str, name: str, group: str, value: float) -> None:
    if vsp.FindParm(gid, name, group):
        vsp.SetParmVal(gid, name, group, value)


# --- fuselage ---------------------------------------------------------------------------------


def _build_fuselage(vsp, fuselage: Fuselage) -> str:
    """Elliptical-section loft through the spec stations. End stations become POINT sections so
    the loft is closed (watertight nose/tail) regardless of their small spec widths."""
    stations = sorted(fuselage.stations, key=lambda s: s.x)
    if len(stations) < 2:
        raise ValueError("Fuselage needs at least 2 stations")

    gid = vsp.AddGeom("FUSELAGE", "")
    vsp.SetGeomName(gid, fuselage.name)
    xsurf = vsp.GetXSecSurf(gid, 0)

    # Match OpenVSP's xsec count to the spec's station count.
    while vsp.GetNumXSec(xsurf) < len(stations):
        vsp.InsertXSec(gid, 1, vsp.XS_ELLIPSE)
    while vsp.GetNumXSec(xsurf) > len(stations):
        vsp.CutXSec(gid, 1)
    vsp.Update()

    x_min = stations[0].x
    length = max(stations[-1].x - x_min, 1e-6)
    ox, oy, oz = fuselage.origin
    vsp.SetParmVal(gid, "Length", "Design", length)
    vsp.SetParmVal(gid, "X_Rel_Location", "XForm", ox + x_min)
    vsp.SetParmVal(gid, "Y_Rel_Location", "XForm", oy)
    vsp.SetParmVal(gid, "Z_Rel_Location", "XForm", oz)
    vsp.Update()

    last = len(stations) - 1
    for i, station in enumerate(stations):
        xsec_id = vsp.GetXSec(xsurf, i)
        if i in (0, last):
            vsp.ChangeXSecShape(xsurf, i, vsp.XS_POINT)
        else:
            vsp.ChangeXSecShape(xsurf, i, vsp.XS_ELLIPSE)
        vsp.Update()
        xsec_id = vsp.GetXSec(xsurf, i)  # re-fetch: ChangeXSecShape replaces the xsec
        _set_xsec_parm(vsp, xsec_id, "XLocPercent", (station.x - x_min) / length)
        _set_xsec_parm(vsp, xsec_id, "ZLocPercent", station.z / length)
        if i not in (0, last):
            _set_xsec_parm(vsp, xsec_id, "Ellipse_Width", station.width)
            _set_xsec_parm(vsp, xsec_id, "Ellipse_Height", station.height)
    vsp.Update()
    return gid


def _set_xsec_parm(vsp, xsec_id: str, name: str, value: float) -> None:
    parm_id = vsp.GetXSecParm(xsec_id, name)
    if parm_id:
        vsp.SetParmVal(parm_id, value)
