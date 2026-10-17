"""Strip-theory viscous correction: 3D span loading plus 2D section polars. Drag Model C.

A vortex-lattice or panel solve is inviscid in the sense that matters here — its drag is induced
drag plus whatever empirical buildup the solver bolts on, and neither one knows what airfoil you
picked. Without this correction, airfoil choice has *no effect* on any 3D performance number.

The method: take the spanwise loading the 3D solver already reports (VSPAERO writes exactly this in
its `VSPAERO_Load` group — station y, local chord, local cl), and for each strip look up a 2D
profile drag coefficient at that strip's *own* Reynolds number and *own* local lift coefficient.
Integrate across the span, reference to Sref, double for the mirrored half.

Two details that are easy to get wrong and both silently bias the answer:

- **Strip width comes from the midpoints between stations, not from differencing them.** The
  stations a solver reports are strip *centres*; differencing them directly drops half a strip at
  each end, which is where the chord is largest at the root and the loading most nonlinear at the tip.
- **Reynolds is per strip.** On a tapered wing the tip can sit a factor of two below the root, and
  that is exactly where a low-Re section starts behaving badly. A single reference Reynolds hides
  the one thing this correction exists to reveal.

**A strip with no usable section data is dropped and counted, never zeroed.** Zero profile drag on
a strip is a number that looks like an answer, and a wing quietly missing a third of its viscous
drag reads as a remarkably efficient one. `StripIntegration.missing` is what the caller reports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from uavaero.schema.airframe import LiftingSurface, ParametricAirframe
from uavaero.solvers.sections.base import SectionLookup


@dataclass
class StripStation:
    """One spanwise strip's own conditions and what the section did in them."""

    y_m: float
    chord_m: float
    width_m: float
    reynolds: float
    cl_local: float
    airfoil: str
    cd_profile: float | None  # None when the section could not be evaluated here

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "StripStation":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class StripIntegration:
    """The integrated profile drag, and enough detail to explain it."""

    cd_profile: float
    stations: list[StripStation] = field(default_factory=list)
    #: Strips the section source could not answer for. Non-zero means `cd_profile` is an
    #: under-estimate over the strips that did work, not a complete integral.
    missing: int = 0
    source: str = ""

    @property
    def complete(self) -> bool:
        return self.missing == 0 and bool(self.stations)

    def to_dict(self) -> dict:
        return {
            "cd_profile": self.cd_profile,
            "stations": [s.to_dict() for s in self.stations],
            "missing": self.missing,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "StripIntegration":
        return cls(
            cd_profile=payload.get("cd_profile", 0.0),
            stations=[StripStation.from_dict(s) for s in payload.get("stations") or []],
            missing=payload.get("missing", 0),
            source=payload.get("source", ""),
        )


def integrate_profile_drag(
    airframe: ParametricAirframe,
    y_m: list[float],
    chord_m: list[float],
    cl_local: list[float],
    sref_m2: float,
    density: float,
    velocity_mps: float,
    lookup: SectionLookup,
    mu: float,
) -> StripIntegration:
    """Profile drag for a wing, from one-sided strip data plus a 2D section source.

    Takes the strip data a solver already reports, so the same correction serves any solver whose
    internals we never see. `mu` and `density` come from the run's own `Atmosphere`.

    Returns the profile drag coefficient referenced to `sref_m2`, doubled for the mirrored half.
    """
    if not y_m or sref_m2 <= 0:
        return StripIntegration(cd_profile=0.0, source=lookup.name)

    ys = np.asarray(y_m, dtype=float)
    chords = np.asarray(chord_m, dtype=float)
    cls = np.asarray(cl_local, dtype=float)

    # Strip widths from the midpoints between stations: the stations are strip centres, so
    # differencing them directly would drop half a strip at each end.
    edges = np.concatenate(([ys[0]], 0.5 * (ys[:-1] + ys[1:]), [ys[-1]]))
    widths = np.diff(edges)

    total = 0.0
    missing = 0
    stations: list[StripStation] = []
    for y, chord, cl, width in zip(ys, chords, cls, widths):
        if chord <= 0 or width <= 0:
            continue
        airfoil = airfoil_at(airframe.wing, float(y))
        reynolds = density * velocity_mps * float(chord) / mu
        cd = lookup.cd(airfoil, reynolds, float(cl))
        stations.append(StripStation(
            y_m=float(y), chord_m=float(chord), width_m=float(width), reynolds=reynolds,
            cl_local=float(cl), airfoil=airfoil, cd_profile=cd,
        ))
        if cd is None:
            missing += 1
            continue
        total += cd * float(chord) * float(width)

    # One side was integrated; the mirror doubles it.
    return StripIntegration(
        cd_profile=2.0 * total / sref_m2, stations=stations, missing=missing, source=lookup.name,
    )


def airfoil_at(surface: LiftingSurface, y_m: float) -> str:
    """Which section owns the strip at this spanwise station.

    The outboard section owns the panel leading into it, matching `WingSection`'s own convention,
    so a wing whose root and tip differ corrects each end with its own airfoil.
    """
    station = 0.0
    for section in surface.sections[1:]:
        station += section.span
        if y_m <= station or section is surface.sections[-1]:
            return section.airfoil
    return surface.sections[-1].airfoil
