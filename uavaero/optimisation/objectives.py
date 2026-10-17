"""What we are asking for, and the gate that stops the optimiser cheating to get it.

**The gate is the important half of this file.**

An optimiser is an adversary. It does not find good aircraft; it finds high scores, and it will
take any route to one. Every guard built in the previous four phases is a route if a violation
scores well:

  - a degenerate mesh returns zeros, and zero drag is an infinite L/D
  - the thick-panel solve returns negative induced drag, which is better than zero
  - a strip with no section data is dropped, so a wing outside the polar range loses viscous drag
  - a missing parasite model leaves the comparison with only its two optimistic members
  - a stalled section is still reported by a panel method as producing lift

Every one of those is *already detected* — `implausible()`, `StripIntegration.missing`,
`AeroResult.notes`, the stall check. What Phase 5 adds is that detection now decides the score.
`trust` refuses anything less than a complete, plausible result, and a refused candidate is worth
nothing at all rather than worth whatever its broken numbers say.

The rule is deliberately blunt: **if the result is not fully trustworthy, it does not compete.** A
softer rule — penalise a bit, score the rest — leaves a gradient pointing towards the broken region,
and a differential-evolution population will follow it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from uavaero.schema.results import AeroResult
from uavaero.schema.stability import StabilityResult

#: What a rejected candidate scores. Large and finite rather than infinite: SciPy's differential
#: evolution handles a big number gracefully and can produce NaN internally from an infinity, which
#: turns "this candidate is bad" into "the whole run is broken".
REJECTED_SCORE = 1e6


@dataclass
class Trust:
    """Whether a solved result may be scored at all."""

    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


def trust(result: AeroResult, require_all_models: bool = True) -> Trust:
    """Refuse anything an optimiser could exploit.

    `require_all_models` is on by default and is not fussiness: the comparison's whole value is
    that the three methods disagree, and a candidate whose expensive, pessimistic model failed to
    run is a candidate scored on its two optimistic ones. Over a population that is a systematic
    bias towards shapes that break the strip correction.
    """
    if not result.polar.CL.size:
        return Trust(False, "the solve produced no polar")
    if not result.mesh.ok:
        return Trust(False, "the mesh was degenerate")

    cd = np.asarray(result.cd, dtype=float)
    if cd.size == 0 or not np.all(np.isfinite(cd)):
        return Trust(False, "the drag came back non-finite")
    if np.any(cd <= 0):
        return Trust(False, f"drag came back non-positive ({cd.min():.5f})")

    induced = np.asarray(result.drag.cd_induced, dtype=float)
    if induced.size and np.any(induced < -0.005):
        return Trust(False, f"induced drag came back negative ({induced.min():.4f})")

    if require_all_models and len(result.drag.predictions()) < 3:
        present = ", ".join(m.key for m in result.drag.predictions())
        return Trust(
            False,
            f"only {len(result.drag.predictions())} drag model(s) ran ({present}) - scoring on a "
            f"subset rewards shapes that break the model that is missing",
        )

    for note in result.notes:
        lowered = note.lower()
        if "under-estimate" in lowered or "no usable section data" in lowered:
            return Trust(False, f"the drag is known to be an under-estimate: {note}")

    stall = result.native.get("stall")
    if stall and not stall.get("ok", True):
        return Trust(False, f"a section is past its stall limit: {stall.get('summary', '')}")

    return Trust(True)


# ════════ objectives ════════════════════════════════════════════════════════════════════════
#
# Each returns a value to MINIMISE, so a maximisation is negated at source rather than by a flag
# threaded through the driver.


def _index_at_cl(result: AeroResult, target_cl: float) -> int:
    """The sweep point nearest a lift coefficient.

    Comparing candidates at a fixed *angle* would be comparing them at different lift coefficients,
    and the one that happens to make more lift at 4 degrees wins on nothing but incidence. A cruise
    comparison is at a fixed CL — that is what "the same flight condition" means.
    """
    return int(np.argmin(np.abs(np.asarray(result.polar.CL, dtype=float) - target_cl)))


def max_lift_to_drag(target_cl: float = 0.5) -> Callable[[AeroResult], float]:
    """Best L/D at a cruise lift coefficient. The classic single objective."""

    def objective(result: AeroResult) -> float:
        index = _index_at_cl(result, target_cl)
        cd = result.cd[index]
        cl = float(result.polar.CL[index])
        return -(cl / cd) if cd > 0 else REJECTED_SCORE

    objective.__name__ = f"max L/D at CL {target_cl:g}"
    return objective


def min_drag(target_cl: float = 0.5) -> Callable[[AeroResult], float]:
    """Least drag coefficient at a cruise lift coefficient."""

    def objective(result: AeroResult) -> float:
        return float(result.cd[_index_at_cl(result, target_cl)])

    objective.__name__ = f"min CD at CL {target_cl:g}"
    return objective


def max_endurance_factor(target_cl: float = 0.7) -> Callable[[AeroResult], float]:
    """Maximise CL^1.5 / CD — the quantity that sets endurance for a propeller aircraft.

    Not the same optimum as L/D: endurance wants a bigger, slower wing than best glide does, and an
    optimiser told to maximise L/D will hand back an aircraft that stays up for less time.
    """

    def objective(result: AeroResult) -> float:
        index = _index_at_cl(result, target_cl)
        cl, cd = float(result.polar.CL[index]), result.cd[index]
        return -(cl**1.5 / cd) if cd > 0 and cl > 0 else REJECTED_SCORE

    objective.__name__ = f"max endurance factor at CL {target_cl:g}"
    return objective


OBJECTIVES = {
    "max_lift_to_drag": max_lift_to_drag,
    "min_drag": min_drag,
    "max_endurance_factor": max_endurance_factor,
}


# ════════ post-solve constraints ════════════════════════════════════════════════════════════


@dataclass
class StabilityConstraints:
    """What a candidate has to satisfy once it has been flown, not merely drawn.

    Applied after the stability solve rather than as a screen, because they cannot be known from
    the geometry alone. A candidate that fails one is rejected outright for the same reason an
    untrustworthy result is: a penalty leaves a gradient towards the forbidden region.
    """

    static_margin: tuple[float, float] = (0.05, 0.25)
    require_directional: bool = True
    require_lateral: bool = True

    def check(self, derivatives: StabilityResult) -> Trust:
        margin = derivatives.static_margin
        if margin is None:
            return Trust(False, "no static margin could be computed")
        low, high = self.static_margin
        if not low <= margin <= high:
            return Trust(
                False,
                f"static margin {margin * 100:.1f}% is outside [{low * 100:.0f}, {high * 100:.0f}]%",
            )
        if not derivatives.longitudinally_stable:
            return Trust(False, "longitudinally unstable")
        if self.require_directional and not derivatives.directionally_stable:
            return Trust(False, "directionally unstable (Cn_beta negative)")
        if self.require_lateral and not derivatives.laterally_stable:
            return Trust(False, "laterally unstable (Cl_beta positive)")
        complaint = derivatives.consistency()
        if complaint:
            return Trust(False, f"the derivative set is internally inconsistent: {complaint}")
        return Trust(True)


@dataclass
class Candidate:
    """One evaluated design."""

    vector: np.ndarray
    values: dict[str, float]
    score: float
    accepted: bool
    reason: str = ""
    lift_to_drag: float | None = None
    cd: float | None = None
    static_margin: float | None = None
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "values": dict(self.values), "score": self.score, "accepted": self.accepted,
            "reason": self.reason, "lift_to_drag": self.lift_to_drag, "cd": self.cd,
            "static_margin": self.static_margin, "elapsed_s": self.elapsed_s,
            "notes": list(self.notes),
        }
