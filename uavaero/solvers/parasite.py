"""Drag Model B: OpenVSP's component parasite-drag buildup.

A wetted-area estimate. Each component gets a flat-plate skin-friction coefficient at its own
Reynolds number, multiplied by a form factor for its thickness and by its wetted area. It knows
nothing about the boundary layer, nothing about the airfoil, and nothing about lift — which is
exactly why it is worth keeping alongside the other two. It is a genuinely independent method with
genuinely different assumptions, and where it agrees with the strip correction you have some reason
to believe both.

**Three things about it that silently produce wrong numbers, or the wrong quantity.**

The first is units, and it is severe. `ParasiteDrag` defaults to *imperial*: it reads a model in
metres as feet and a velocity in m/s as ft/s, and reports a Reynolds number an order of magnitude
low without complaining. Measured on the conventional reference aircraft at 20 m/s, the same
geometry gives:

    imperial default   CD 0.0325    wing Re     32 220
    PD_UNITS_METRIC    CD 0.0605    wing Re      3 468   (it means CENTIMETRES)
    LEN_M              CD 0.0191    wing Re    346 817   (correct)

A factor of three, from a setting nobody would think to check, on an answer that looks entirely
plausible at every one of those values. Note the middle row in particular — `PD_UNITS_METRIC` is
not metres, and reaching for it because it says "metric" makes things worse rather than better.
`LengthUnit` must be `vsp.LEN_M`.

The second is the set it is given. `VSPAEROComputeGeometry` leaves an `NGonMesh` geom called
`MeshGeom_NGon` — the analysis mesh — behind in the model, and it lands in `SET_ALL`. Handing
`SET_ALL` to ParasiteDrag therefore hands it a mesh artefact it cannot build a wetted area for,
and it says so:

    Warning: Geom ID ZCBOVONKQY not included in Parasite Drag calculation

That warning is benign in effect — every real component was still included — but it is a warning
about the wrong thing being asked, and it went unnoticed for two phases because nothing was
displaying the solver's output. So the components are now named explicitly through a user set
rather than swept up with `SET_ALL`, which also means the buildup cannot silently acquire whatever
else a future analysis leaves lying in the model.

The third is that this is a *parasite* buildup, not a total. It carries no induced drag and does
not vary with lift. Reported on its own it would sit far below the other two models at every angle
and look like a different quantity, because it is one. So Model B as compared here is
`CDi(VSPAERO) + CD_parasite`, which is the same construction the plan's §9 uses and makes all three
models totals of the same thing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from uavaero.schema.condition import OperatingCondition


@dataclass
class ComponentDrag:
    """One component's contribution to the buildup."""

    name: str
    wetted_area_m2: float
    reference_length_m: float
    reynolds: float
    form_factor: float
    #: Equivalent flat-plate area, m². The buildup is the sum of these over Sref.
    flat_plate_area_m2: float
    share: float  # fraction of the total parasite drag

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ComponentDrag":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class ParasiteResult:
    """The buildup, and enough of its working to argue with."""

    cd_parasite: float
    components: list[ComponentDrag] = field(default_factory=list)
    laminar_equation: str = ""
    turbulent_equation: str = ""
    sref_m2: float = 0.0
    velocity_mps: float = 0.0
    altitude_m: float = 0.0

    @property
    def wetted_area_m2(self) -> float:
        return sum(c.wetted_area_m2 for c in self.components)

    @property
    def wetted_area_ratio(self) -> float:
        """Total wetted area over reference area. A conventional light aircraft sits around 4;
        a number far from that usually means a component was double-counted or missed."""
        return self.wetted_area_m2 / self.sref_m2 if self.sref_m2 > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cd_parasite": self.cd_parasite,
            "components": [c.to_dict() for c in self.components],
            "laminar_equation": self.laminar_equation,
            "turbulent_equation": self.turbulent_equation,
            "sref_m2": self.sref_m2, "velocity_mps": self.velocity_mps,
            "altitude_m": self.altitude_m,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParasiteResult":
        return cls(
            cd_parasite=payload.get("cd_parasite", 0.0),
            components=[ComponentDrag.from_dict(c) for c in payload.get("components") or []],
            laminar_equation=payload.get("laminar_equation", ""),
            turbulent_equation=payload.get("turbulent_equation", ""),
            sref_m2=payload.get("sref_m2", 0.0),
            velocity_mps=payload.get("velocity_mps", 0.0),
            altitude_m=payload.get("altitude_m", 0.0),
        )


#: The user set the components are collected into. Sets 0-2 are All/Shown/Not_Shown; the user
#: sets start at 3 and this tree uses the first of them for nothing else.
COMPONENT_SET_NAME = "uavaero_parasite"


def component_set(vsp, geom_ids: list[str]) -> int:
    """Put exactly these geoms in a set of their own and return its index.

    Named rather than inherited. `SET_ALL` also contains whatever the VSPAERO geometry pass left
    behind — an `NGonMesh` it builds for the solver — and a buildup should be over the components
    you meant, not over everything currently in the model.
    """
    index = vsp.SET_FIRST_USER
    vsp.SetSetName(index, COMPONENT_SET_NAME)
    for gid in vsp.FindGeoms():
        vsp.SetSetFlag(gid, index, gid in geom_ids)
    return index


def run_parasite_drag(
    condition: OperatingCondition,
    sref_m2: float,
    geom_set: int | None = None,
    geom_ids: list[str] | None = None,
) -> ParasiteResult:
    """Run OpenVSP's ParasiteDrag on the currently built model.

    **Only call this inside a worker process**, after the geometry has been built — it reads
    OpenVSP's process-global model, exactly like the VSPAERO sweep does.

    Pass `geom_ids` — the components the adapter actually built — so the buildup runs over those
    and not over whatever else is in the model. `SET_ALL` remains the fallback for a caller that
    does not know, but it will warn about the analysis mesh.
    """
    import openvsp as vsp

    air = condition.atmosphere
    if geom_set is not None:
        target_set = geom_set
    elif geom_ids:
        target_set = component_set(vsp, list(geom_ids))
    else:
        target_set = vsp.SET_ALL

    vsp.SetAnalysisInputDefaults("ParasiteDrag")
    vsp.SetIntAnalysisInput("ParasiteDrag", "GeomSet", [target_set], 0)
    vsp.SetIntAnalysisInput("ParasiteDrag", "RefFlag", [0], 0)  # never auto-derive; see rule 4
    vsp.SetDoubleAnalysisInput("ParasiteDrag", "Sref", [float(sref_m2)], 0)

    # The units block. Omitting it is not a rounding error — see the module docstring.
    # `LEN_M` is metres; `PD_UNITS_METRIC` is centimetres and is the wrong constant here.
    vsp.SetIntAnalysisInput("ParasiteDrag", "LengthUnit", [vsp.LEN_M], 0)
    vsp.SetIntAnalysisInput("ParasiteDrag", "AltLengthUnit", [vsp.PD_UNITS_METRIC], 0)
    vsp.SetIntAnalysisInput("ParasiteDrag", "VelocityUnit", [vsp.V_UNIT_M_S], 0)
    vsp.SetIntAnalysisInput("ParasiteDrag", "TempUnit", [vsp.TEMP_UNIT_K], 0)
    vsp.SetIntAnalysisInput("ParasiteDrag", "PresUnit", [vsp.PRES_UNIT_PA], 0)

    vsp.SetDoubleAnalysisInput("ParasiteDrag", "Vinf", [float(condition.velocity_mps)], 0)
    vsp.SetDoubleAnalysisInput("ParasiteDrag", "Altitude", [float(condition.altitude_m)], 0)

    results_id = vsp.ExecAnalysis("ParasiteDrag")
    if not results_id:
        raise RuntimeError("OpenVSP's ParasiteDrag analysis returned no results")

    def doubles(name: str) -> list[float]:
        return [float(v) for v in vsp.GetDoubleResults(results_id, name)]

    def strings(name: str) -> list[str]:
        return [str(v) for v in vsp.GetStringResults(results_id, name)]

    labels = strings("Comp_Label")
    components = [
        ComponentDrag(
            name=name,
            wetted_area_m2=swet,
            reference_length_m=lref,
            reynolds=reynolds,
            form_factor=form,
            flat_plate_area_m2=flat,
            share=share,
        )
        for name, swet, lref, reynolds, form, flat, share in zip(
            labels, doubles("Comp_Swet"), doubles("Comp_Lref"), doubles("Comp_Re"),
            doubles("Comp_FFOut"), doubles("Comp_f"), doubles("Comp_PercTotalCD"),
        )
    ]

    total = doubles("Total_CD_Total")
    return ParasiteResult(
        cd_parasite=total[0] if total else 0.0,
        components=components,
        laminar_equation=(strings("LamCfEqnName") or [""])[0],
        turbulent_equation=(strings("TurbCfEqnName") or [""])[0],
        sref_m2=float(sref_m2),
        velocity_mps=condition.velocity_mps,
        altitude_m=condition.altitude_m,
    )


def sanity_check(result: ParasiteResult, condition: OperatingCondition, mac_m: float) -> str:
    """Why this buildup should not be believed, or "" if nothing looks wrong.

    The unit trap is invisible in the drag coefficient — every setting produces a plausible number
    — but it is glaring in the Reynolds number, which is a quantity we can compute independently.
    So that is what gets checked.
    """
    if not result.components:
        return "the buildup produced no components"
    wing = next((c for c in result.components if c.name.lower() == "wing"), result.components[0])
    if wing.reference_length_m <= 0:
        return "the wing component reported no reference length"
    expected = condition.atmosphere.reynolds(condition.velocity_mps, wing.reference_length_m)
    if expected <= 0:
        return ""
    ratio = wing.reynolds / expected
    if not 0.9 < ratio < 1.1:
        return (
            f"Reynolds disagrees with the flow condition by {ratio:.2f}x "
            f"({wing.reynolds:,.0f} against {expected:,.0f}) — this is the signature of "
            f"ParasiteDrag running in the wrong length unit"
        )
    return ""
