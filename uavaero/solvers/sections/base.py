"""One interface over the 2D section solvers, and the lookup the strip integrator drives.

NeuralFoil and XFOIL answer the same question — what does this airfoil do at this Reynolds number
— with completely different failure behaviour. NeuralFoil always returns a number; XFOIL converges
on some angles and not others, and at low Reynolds it can fail across a whole band. Everything
above this line has to work identically with either, so the difference is absorbed here.

**The rule that matters: never interpolate across a hole.** When XFOIL fails to converge between
CL 0.6 and CL 1.1, `numpy.interp` will happily draw a straight line across the gap and hand back a
number that looks exactly like a measurement. That is the single most dangerous thing a polar
database can do, because the invented value is smooth, plausible, and wrong in the direction that
makes a bad airfoil look acceptable. `SectionLookup.cd` returns `None` instead, and the caller
reports the absence rather than integrating a fiction.

`SectionLookup` also owns the caching. A wing whose root and tip share an airfoil costs one or two
section solves per evaluation rather than one per strip, because Reynolds is bucketed before it is
used as a key — local chord varies continuously along a tapered panel, so unbucketed keys would
miss on every single strip.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from uavaero.schema.results import Polar

#: Fixed alpha grid every source is sampled over. Wide enough to bracket the CL range a wing
#: actually reaches, fine enough that linear interpolation between neighbours is honest.
ALPHA_GRID_DEG = np.linspace(-8.0, 16.0, 49)

#: Reynolds numbers are rounded to this before being used as a cache key.
RE_BUCKET = 25_000.0

#: The widest CL gap between two converged points that may still be interpolated across.
#:
#: Below this a linear interpolation is a fair reading of a smooth polar. Above it, the two
#: bracketing points are not describing the same flow regime any more — that is what the gap
#: means — and joining them invents a drag value across a stall or a separation bubble.
MAX_CL_GAP = 0.25


@runtime_checkable
class SectionSource(Protocol):
    """A 2D airfoil polar, from whichever solver."""

    #: Stable key recorded on results so a drag model can say what produced it.
    name: str

    def polar(
        self, airfoil: str, alpha_deg: np.ndarray, reynolds: float, mach: float = 0.0
    ) -> Polar:
        """Sweep this airfoil. May return fewer points than requested — a solver that did not
        converge everywhere must drop the failures rather than pad them."""
        ...


def cd_at_cl(polar: Polar, cl: float, max_gap: float = MAX_CL_GAP) -> float | None:
    """Profile drag at a lift coefficient, or None if the polar cannot honestly answer.

    Read off CL rather than root-finding alpha per strip: the strip already knows its own local
    lift coefficient from the 3D solution, and going back through alpha would re-solve something
    that is already known.

    Returns None when `cl` is outside the polar's range, or when the two points bracketing it are
    further apart in CL than `max_gap` — the hole a non-converged region leaves behind.
    """
    cl_values = np.asarray(polar.CL, dtype=float)
    cd_values = np.asarray(polar.CD, dtype=float)
    if len(cl_values) < 2:
        return None

    order = np.argsort(cl_values)
    cl_sorted, cd_sorted = cl_values[order], cd_values[order]

    # Outside the data entirely. Extrapolating profile drag past the last converged point is how a
    # stalled section comes back reporting cruise drag.
    if cl < cl_sorted[0] or cl > cl_sorted[-1]:
        return None

    upper = int(np.searchsorted(cl_sorted, cl, side="left"))
    if upper == 0:
        return float(cd_sorted[0])
    lower = upper - 1
    if upper >= len(cl_sorted):
        return float(cd_sorted[-1])

    if cl_sorted[upper] - cl_sorted[lower] > max_gap:
        return None
    return float(np.interp(cl, cl_sorted, cd_sorted))


def cl_max_of(polar: Polar) -> float | None:
    """The highest lift coefficient the section reached before dropping off.

    Taken as the maximum of the polar rather than by looking for the slope to reverse: a polar that
    was cut short by non-convergence has its real Cl_max somewhere above the last converged point,
    and the honest reading of that is "at least this", which is what the stall check wants.
    """
    cl_values = np.asarray(polar.CL, dtype=float)
    return float(np.max(cl_values)) if len(cl_values) else None


class SectionLookup:
    """A section source, its polar cache, and honest interpolation over it."""

    def __init__(
        self,
        source: SectionSource,
        mach: float = 0.0,
        alpha_grid: np.ndarray | None = None,
        re_bucket: float = RE_BUCKET,
        max_cl_gap: float = MAX_CL_GAP,
    ) -> None:
        self.source = source
        self.mach = mach
        self.alpha_grid = ALPHA_GRID_DEG if alpha_grid is None else alpha_grid
        self.re_bucket = re_bucket
        self.max_cl_gap = max_cl_gap
        self._polars: dict[tuple[str, float], Polar | None] = {}
        #: What could not be evaluated, so the caller can report it rather than guess.
        self.misses: list[str] = []

    @property
    def name(self) -> str:
        return self.source.name

    def bucket(self, reynolds: float) -> float:
        return max(self.re_bucket, round(reynolds / self.re_bucket) * self.re_bucket)

    def polar(self, airfoil: str, reynolds: float) -> Polar | None:
        """Cached polar, or None when the source could not produce one at all.

        A source failure is cached too. Without that, an airfoil XFOIL cannot handle would be
        retried once per strip — turning one failure into fifty slow ones.
        """
        key = (airfoil, self.bucket(reynolds))
        if key in self._polars:
            return self._polars[key]
        try:
            result = self.source.polar(airfoil, self.alpha_grid, key[1], self.mach)
        except Exception as exc:  # noqa: BLE001 - a section that will not solve is a finding
            self.misses.append(f"{airfoil} at Re {key[1]:,.0f}: {type(exc).__name__}: {exc}")
            result = None
        self._polars[key] = result
        return result

    def cd(self, airfoil: str, reynolds: float, cl: float) -> float | None:
        polar = self.polar(airfoil, reynolds)
        if polar is None:
            return None
        value = cd_at_cl(polar, cl, self.max_cl_gap)
        if value is None:
            self.misses.append(
                f"{airfoil} at Re {self.bucket(reynolds):,.0f}: no usable data at CL {cl:.3f}"
            )
        return value

    def cl_max(self, airfoil: str, reynolds: float) -> float | None:
        polar = self.polar(airfoil, reynolds)
        return None if polar is None else cl_max_of(polar)
