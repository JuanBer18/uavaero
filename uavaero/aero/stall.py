"""Where the wing runs out of lift first, and by how much.

A panel method has no idea what stall is. It will happily report a local lift coefficient of 1.9 on
a section whose 2D data tops out at 1.2, and the resulting polar looks perfectly smooth — which is
how a design that drops a wing at the tip passes an aerodynamic review.

The check is deliberately simple and deliberately *not* a stall model: for each strip, compare the
local lift coefficient the 3D solve produced against the section's own Cl_max at that strip's
Reynolds number. Where the margin goes negative, the section is being asked for more than it can
give. That is a warning and, later, an optimiser constraint — not a prediction of what the aircraft
then does, which would need a coupled viscous-inviscid solve this tool does not have.

**Where it happens matters more than whether it happens.** A root stall is docile and gives warning;
a tip stall on a tapered wing takes the ailerons with it and rolls. So the report says which station,
not just that a station exists.
"""
from __future__ import annotations

from dataclasses import dataclass

from uavaero.aero.strips import StripStation
from uavaero.solvers.sections.base import SectionLookup

#: Below this fraction of Cl_max a station is worth mentioning even though it has not gone over.
#: Stall margin collapses quickly with incidence, so a wing at 0.95 at cruise has no reserve.
WATCH_FRACTION = 0.90


@dataclass
class StationMargin:
    """One strip's stall margin."""

    y_m: float
    span_fraction: float
    cl_local: float
    cl_max: float | None
    reynolds: float
    airfoil: str

    @property
    def margin(self) -> float | None:
        """Cl_max − Cl_local. Negative means the section is over its 2D limit."""
        return None if self.cl_max is None else self.cl_max - self.cl_local

    @property
    def fraction(self) -> float | None:
        """How much of the section's capability is being used."""
        if self.cl_max is None or self.cl_max <= 0:
            return None
        return self.cl_local / self.cl_max

    @property
    def exceeded(self) -> bool:
        return self.margin is not None and self.margin < 0


@dataclass
class StallReport:
    """The wing's stall margins at one angle of attack."""

    alpha_deg: float
    stations: list[StationMargin]

    @property
    def worst(self) -> StationMargin | None:
        """The station with the least margin. None when nothing could be evaluated."""
        rated = [s for s in self.stations if s.margin is not None]
        return min(rated, key=lambda s: s.margin) if rated else None

    @property
    def exceeded(self) -> list[StationMargin]:
        return [s for s in self.stations if s.exceeded]

    @property
    def ok(self) -> bool:
        return not self.exceeded

    def summary(self) -> str:
        """One line, phrased for someone deciding whether to care.

        Names the station because a root stall and a tip stall are different aircraft.
        """
        worst = self.worst
        if worst is None:
            return "No section data — stall margin unknown"
        where = f"{worst.span_fraction * 100:.0f}% semispan"
        if worst.exceeded:
            return (
                f"Section stall limit exceeded at {where}: local CL {worst.cl_local:.2f} against "
                f"a 2D Cl_max of {worst.cl_max:.2f} at Re {worst.reynolds:,.0f} "
                f"(margin {worst.margin:+.2f})"
            )
        if worst.fraction is not None and worst.fraction >= WATCH_FRACTION:
            return (
                f"Little stall margin at {where}: {worst.fraction * 100:.0f}% of the section's "
                f"Cl_max ({worst.cl_local:.2f} of {worst.cl_max:.2f})"
            )
        return (
            f"Stall margin {worst.margin:+.2f} at its tightest, at {where} "
            f"({worst.fraction * 100:.0f}% of Cl_max)"
        )


def check(
    stations: list[StripStation], lookup: SectionLookup, alpha_deg: float = 0.0
) -> StallReport:
    """Compare every strip's local lift against its own section's 2D limit.

    The lookup is the same one the drag integration used, so the Cl_max compared against comes from
    the same polar as the drag — comparing against a different source's Cl_max would make the
    margin and the drag disagree about what the section is doing.
    """
    span = max((s.y_m for s in stations), default=0.0)
    margins = [
        StationMargin(
            y_m=s.y_m,
            span_fraction=(s.y_m / span) if span > 0 else 0.0,
            cl_local=s.cl_local,
            cl_max=lookup.cl_max(s.airfoil, s.reynolds),
            reynolds=s.reynolds,
            airfoil=s.airfoil,
        )
        for s in stations
    ]
    return StallReport(alpha_deg=alpha_deg, stations=margins)
