"""What a solve returns, independent of who solved it.

`Polar` is the raw sweep. `AeroResult` is the canonical result every solver produces and every
consumer — plots, dynamics, the optimiser — reads. `DragBreakdown` is the centrepiece.

**On DragBreakdown.** This tool deliberately does not collapse drag to one number. Several
independent methods estimate the same quantity, they disagree, and the disagreement is
information: it is the honest error bar on a preliminary-design prediction. So every method is kept
side by side under its own name, and a later CFD run or wind-tunnel campaign is *added as another
entry* rather than used to overwrite the predictions it is judging. Once both a prediction and a
reference are present, the error between them can be attributed — this method is reliable in
cruise, that one degrades at high lift, this one falls apart below Re 100k — which is a thing you
can only learn if nothing was ever silently replaced.

Nothing here computes physics. These are containers with the arithmetic of comparison on them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class Polar:
    """One sweep, regardless of which solver produced it.

    `source` records which backend produced it; `meta` carries solver-specific extras (Reynolds,
    Mach, and the raw per-alpha result dicts needed for pressure/streamline overlays).
    """

    alpha_deg: np.ndarray
    CL: np.ndarray
    CD: np.ndarray
    CM: np.ndarray
    beta_deg: np.ndarray | None = None  # None for a 2D section polar
    CY: np.ndarray | None = None
    Cl: np.ndarray | None = None  # rolling-moment coefficient (vs. CL = lift coefficient)
    Cn: np.ndarray | None = None
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-plain: numpy arrays -> lists, so this survives persistence."""

        def _arr(a: np.ndarray | None) -> list[float] | None:
            return None if a is None else np.asarray(a, dtype=float).tolist()

        return {
            "alpha_deg": _arr(self.alpha_deg), "CL": _arr(self.CL), "CD": _arr(self.CD),
            "CM": _arr(self.CM), "beta_deg": _arr(self.beta_deg), "CY": _arr(self.CY),
            "Cl": _arr(self.Cl), "Cn": _arr(self.Cn), "source": self.source, "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Polar":
        def _arr(v: list[float] | None) -> np.ndarray | None:
            return None if v is None else np.asarray(v, dtype=float)

        return cls(
            alpha_deg=_arr(payload["alpha_deg"]), CL=_arr(payload["CL"]), CD=_arr(payload["CD"]),
            CM=_arr(payload["CM"]), beta_deg=_arr(payload.get("beta_deg")),
            CY=_arr(payload.get("CY")), Cl=_arr(payload.get("Cl")), Cn=_arr(payload.get("Cn")),
            source=payload.get("source", ""), meta=payload.get("meta", {}),
        )


@dataclass
class SpanLoad:
    """Spanwise lift distribution at one alpha, root to tip, folded onto one side.

    Not a plot for its own sake: `cl_local` at each station's own chord and Reynolds number is what
    the strip-theory viscous correction integrates, and comparing `cl_local` against the section's
    Cl_max at that Reynolds is what detects tip stall.
    """

    alpha_deg: float = 0.0
    y_m: list[float] = field(default_factory=list)
    chord_m: list[float] = field(default_factory=list)
    cl_local: list[float] = field(default_factory=list)
    root_bending_moment_coefficient: float | None = None
    #: Which surface each station belongs to, as VSPAERO's own `VortexSheet` id. Empty on a result
    #: written before this was read, and on any solver that does not report one.
    #:
    #: Load carries every lifting surface, not just the wing, and without this there is no way to
    #: tell a tail strip from a wing strip: they interleave by span, and a horizontal tail sits
    #: *inside* the wing's y range rather than beyond it. The viscous correction integrates only
    #: the wing (the other surfaces are handled by the component buildup), so it needs to know.
    sheet: list[int] = field(default_factory=list)

    def stations_on(self, sheets: set[int]) -> "SpanLoad":
        """This load narrowed to one set of surfaces. Everything when no sheet ids were recorded,
        which keeps an older result readable rather than silently emptying it."""
        if not self.sheet or len(self.sheet) != len(self.y_m):
            return self
        keep = [i for i, s in enumerate(self.sheet) if s in sheets]
        return SpanLoad(
            alpha_deg=self.alpha_deg,
            y_m=[self.y_m[i] for i in keep],
            chord_m=[self.chord_m[i] for i in keep],
            cl_local=[self.cl_local[i] for i in keep],
            root_bending_moment_coefficient=self.root_bending_moment_coefficient,
            sheet=[self.sheet[i] for i in keep],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpanLoad":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class MeshDiagnostics:
    """Panel-mesh health, read out of the solver's own log.

    `kutta_nodes == 0` is the one unambiguous red flag: it is the signature of every degenerate
    mesh, and it produces a solve that returns zeros rather than failing — which reads as a
    spectacularly low-drag aircraft rather than as an error.
    """

    nodes: int | None = None
    loops: int | None = None
    kutta_nodes: int | None = None
    surfaces: int | None = None
    unresolved_wake_loops: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.kutta_nodes) and bool(self.nodes)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MeshDiagnostics":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class DragModel:
    """One independent estimate of total drag across a sweep.

    `basis` is a single sentence saying how the number was arrived at, and it is not decoration:
    when two models disagree by 30% the first question is always what each one actually did, and
    an answer that has to be reconstructed from source is an answer nobody checks.

    `reference=True` marks a measurement rather than a prediction — CFD, wind tunnel, flight test.
    References are what predictions get scored against, and they are never averaged into them.
    """

    key: str
    label: str
    cd: list[float]
    basis: str = ""
    reference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DragModel":
        return cls(**payload)


@dataclass
class DragBreakdown:
    """Every drag estimate for one sweep, kept side by side.

    Insertion-ordered: models appear in the order they were added, which is the order they are
    reported in. Adding a model never removes one.
    """

    cd_induced: list[float] = field(default_factory=list)
    models: dict[str, DragModel] = field(default_factory=dict)

    def add(self, model: DragModel) -> None:
        self.models[model.key] = model

    def cd(self, key: str) -> list[float]:
        return self.models[key].cd

    def predictions(self) -> list[DragModel]:
        return [m for m in self.models.values() if not m.reference]

    def references(self) -> list[DragModel]:
        return [m for m in self.models.values() if m.reference]

    def lift_to_drag(self, key: str, cl: list[float] | np.ndarray) -> list[float]:
        return [float(c) / float(d) if d > 0 else 0.0 for c, d in zip(cl, self.cd(key))]

    def spread(self, index: int) -> float:
        """Max minus min across the *predictions* at one sweep point.

        References are excluded on purpose: the spread is meant to say how much the methods
        disagree with each other, and folding a measurement in would turn a disagreement between
        models into a model-versus-truth error, which is a different quantity with a different use.
        """
        values = [m.cd[index] for m in self.predictions() if index < len(m.cd)]
        return max(values) - min(values) if len(values) > 1 else 0.0

    def relative_spread(self, index: int) -> float:
        """Spread as a fraction of the mean prediction. This is the number worth putting a
        threshold on — an absolute spread of 0.004 is negligible on a brick and enormous on a
        sailplane."""
        values = [m.cd[index] for m in self.predictions() if index < len(m.cd)]
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return (max(values) - min(values)) / mean if mean > 0 else 0.0

    def error_against(self, reference_key: str, index: int) -> dict[str, float]:
        """Relative error of every prediction against one reference, at one sweep point.

        (CD_predicted - CD_reference) / CD_reference, so positive means the model over-predicts
        drag. This is the whole point of keeping references alongside rather than instead of.
        """
        ref = self.models[reference_key].cd[index]
        if ref <= 0:
            return {}
        return {m.key: (m.cd[index] - ref) / ref for m in self.predictions() if index < len(m.cd)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "cd_induced": list(self.cd_induced),
            "models": [m.to_dict() for m in self.models.values()],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DragBreakdown":
        breakdown = cls(cd_induced=list(payload.get("cd_induced") or []))
        for entry in payload.get("models") or []:
            breakdown.add(DragModel.from_dict(entry))
        return breakdown


@dataclass
class ReferenceValues:
    """What the coefficients are referenced to.

    Carried on the result rather than assumed, because a coefficient without its reference area is
    not a number you can compare with anything. `bref` is the FULL span — the doubling that
    `LiftingSurface.span()` does not do.
    """

    sref_m2: float = 0.0
    bref_m: float = 0.0
    cref_m: float = 0.0

    @property
    def aspect_ratio(self) -> float:
        return self.bref_m**2 / self.sref_m2 if self.sref_m2 > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "aspect_ratio": self.aspect_ratio}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceValues":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class AeroResult:
    """The canonical result. Every solver produces one of these; nothing downstream knows which
    solver ran except by reading `solver`."""

    polar: Polar
    drag: DragBreakdown = field(default_factory=DragBreakdown)
    span_loads: list[SpanLoad] = field(default_factory=list)
    reference: ReferenceValues = field(default_factory=ReferenceValues)
    mesh: MeshDiagnostics = field(default_factory=MeshDiagnostics)
    reynolds: float = 0.0
    mach: float = 0.0
    #: Provenance. A result that cannot say what produced it, at what fidelity, over which
    #: components is not a result — it is a rumour, and it will be compared against something it
    #: should never have been compared with.
    solver: str = ""
    fidelity: str = ""
    scope: str = ""
    elapsed_s: float = 0.0
    #: Solver-specific extras, JSON-plain. The escape hatch that keeps this dataclass from growing
    #: a field per backend.
    native: dict[str, Any] = field(default_factory=dict)
    #: Things the reader has to be told: a drag model that could not run, a section the polar
    #: source could not answer for, a station over its stall limit. A result that quietly omits a
    #: model looks identical to one where that model happened to agree.
    notes: list[str] = field(default_factory=list)

    #: Which model `cd` and `lift_to_drag` report by default.
    primary_model: str = ""

    @property
    def cd(self) -> list[float]:
        if self.primary_model and self.primary_model in self.drag.models:
            return self.drag.cd(self.primary_model)
        return [float(v) for v in self.polar.CD]

    @property
    def lift_to_drag(self) -> list[float]:
        return [float(cl) / float(cd) if cd > 0 else 0.0 for cl, cd in zip(self.polar.CL, self.cd)]

    @property
    def best_lift_to_drag(self) -> tuple[float, float]:
        """(best L/D, the alpha it happens at). (0, 0) on an empty sweep rather than an
        IndexError — a caller formatting a result should not have to guard."""
        ratios = self.lift_to_drag
        if not ratios:
            return 0.0, 0.0
        i = int(np.argmax(ratios))
        return float(ratios[i]), float(self.polar.alpha_deg[i])

    @property
    def cl_alpha_per_rad(self) -> float:
        """Lift-curve slope from a straight-line fit across the sweep."""
        alpha = np.radians(np.asarray(self.polar.alpha_deg, dtype=float))
        cl = np.asarray(self.polar.CL, dtype=float)
        return float(np.polyfit(alpha, cl, 1)[0]) if len(alpha) >= 2 else 0.0

    def fit_cd0_e(self) -> tuple[float, float]:
        """(CD0, Oswald e) from a `CD = CD0 + k CL^2` fit over the primary model.

        Fitted rather than read off a solver's own parasite column, because that column is a
        component buildup and CD0 as used everywhere else is the intercept of the actual drag
        polar. Reporting the buildup under the name CD0 compares two different quantities.
        """
        cl = np.asarray(self.polar.CL, dtype=float)
        cd = np.asarray(self.cd, dtype=float)
        ar = self.reference.aspect_ratio
        if len(cl) < 3:
            return (float(cd[0]) if len(cd) else 0.0), 0.0
        slope, intercept = np.polyfit(cl**2, cd, 1)
        cd0 = max(float(intercept), 0.0)
        e = 1.0 / (np.pi * ar * float(slope)) if ar > 0 and slope > 0 else 0.0
        return cd0, float(e)

    def to_dict(self) -> dict[str, Any]:
        return {
            "polar": self.polar.to_dict(), "drag": self.drag.to_dict(),
            "span_loads": [s.to_dict() for s in self.span_loads],
            "reference": self.reference.to_dict(), "mesh": self.mesh.to_dict(),
            "reynolds": self.reynolds, "mach": self.mach, "solver": self.solver,
            "fidelity": self.fidelity, "scope": self.scope, "elapsed_s": self.elapsed_s,
            "native": self.native, "primary_model": self.primary_model,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AeroResult":
        return cls(
            polar=Polar.from_dict(payload["polar"]),
            drag=DragBreakdown.from_dict(payload.get("drag") or {}),
            span_loads=[SpanLoad.from_dict(s) for s in payload.get("span_loads") or []],
            reference=ReferenceValues.from_dict(payload.get("reference") or {}),
            mesh=MeshDiagnostics.from_dict(payload.get("mesh") or {}),
            reynolds=payload.get("reynolds", 0.0), mach=payload.get("mach", 0.0),
            solver=payload.get("solver", ""), fidelity=payload.get("fidelity", ""),
            scope=payload.get("scope", ""), elapsed_s=payload.get("elapsed_s", 0.0),
            native=payload.get("native") or {}, primary_model=payload.get("primary_model", ""),
            notes=list(payload.get("notes") or []),
        )
