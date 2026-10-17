"""Rejecting a candidate before the solver runs.

An optimiser generates a great many aircraft that should not exist, and a solve costs tens of
seconds. Screening is what makes an optimisation finish: a check that runs in microseconds and
throws away the tip chord of two millimetres before OpenVSP is asked to mesh it.

**It is also a safety device, not only a speed one.** Some impossible geometry does not fail to
solve — it solves, and returns an attractive number. A wing with a 5 mm tip chord has a local
Reynolds number in the thousands where the section polars have no data, so the strip correction
drops those strips, the profile drag comes out low and the candidate scores well. That is precisely
the failure the plan warns about: an optimiser exploiting a gap in the model rather than finding a
better aircraft. Screening closes it at the front.

Every check answers with a sentence, and the reasons are counted over a run. A population that is
90% rejected for one reason is telling you the bounds are wrong, and that is worth knowing long
before the objective history is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from uavaero.geometry import authoring
from uavaero.schema.airframe import ParametricAirframe
from uavaero.schema.condition import OperatingCondition

#: The smallest chord that still means something at model scale. Below this the section is outside
#: the range any 2D polar covers, the strip correction has no data, and the drag comes out
#: flattering — so it is a trap rather than merely a small wing.
MIN_CHORD_M = 0.04

#: The lowest local Reynolds number the section data is trustworthy at. NeuralFoil's training set
#: thins out below this and XFOIL stops converging; either way the profile drag is a guess.
MIN_REYNOLDS = 40_000.0

#: Tail volume bounds far wider than the conventional band. These reject the absurd, not the
#: unconventional — a canard or a tailless design is meant to live outside `VH_TYPICAL`, but a Vh
#: of 0.001 is an aircraft with no tail at all.
VH_LIMITS = (0.05, 2.0)
VV_LIMITS = (0.002, 0.30)


@dataclass
class Screen:
    """The verdict on one candidate before anything expensive happens."""

    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class Limits:
    """The manufacturable and physical envelope a candidate has to sit inside."""

    max_span_m: float = 4.0
    min_chord_m: float = MIN_CHORD_M
    min_reynolds: float = MIN_REYNOLDS
    max_aspect_ratio: float = 20.0
    vh_limits: tuple[float, float] = VH_LIMITS
    vv_limits: tuple[float, float] = VV_LIMITS
    #: The tail has to be behind the wing. An optimiser handed an arm range that reaches zero will
    #: put the tail on the wing and report a delightful drag.
    min_tail_arm_m: float = 0.15


def screen(
    airframe: ParametricAirframe,
    condition: OperatingCondition,
    limits: Limits | None = None,
) -> Screen:
    """Whether this aircraft is worth solving. Microseconds, no solver."""
    limits = limits or Limits()
    wing = airframe.wing

    chords = [s.chord for s in wing.sections]
    if not chords or min(chords) <= 0:
        return Screen(False, "a chord came out non-positive")
    if min(chords) < limits.min_chord_m:
        return Screen(
            False,
            f"tip chord {min(chords) * 1000:.0f} mm is below {limits.min_chord_m * 1000:.0f} mm, "
            f"where the section data runs out and the drag would flatter it",
        )

    facts = authoring.summary(airframe)
    span = facts["wing_span_m"]
    if span > limits.max_span_m:
        return Screen(False, f"span {span:.2f} m exceeds the {limits.max_span_m:.2f} m limit")
    if facts["wing_aspect_ratio"] > limits.max_aspect_ratio:
        return Screen(
            False,
            f"aspect ratio {facts['wing_aspect_ratio']:.1f} is above "
            f"{limits.max_aspect_ratio:.0f} and would not be structurally real",
        )
    if facts["wing_area_m2"] <= 0:
        return Screen(False, "the wing has no area")

    # Local Reynolds at the tip, which is the lowest on the wing and the first to leave the data.
    tip_reynolds = condition.atmosphere.reynolds(condition.velocity_mps, min(chords))
    if tip_reynolds < limits.min_reynolds:
        return Screen(
            False,
            f"tip Reynolds {tip_reynolds:,.0f} is below {limits.min_reynolds:,.0f}, where the "
            f"section polars have no data and the strip correction would silently drop strips",
        )

    for name, arm_key, volume_key, bounds in (
        ("horizontal tail", "htail_arm_m", "horizontal_tail_volume", limits.vh_limits),
        ("fin", "vtail_arm_m", "vertical_tail_volume", limits.vv_limits),
    ):
        arm = facts[arm_key]
        if arm is None:
            continue
        if arm < limits.min_tail_arm_m:
            return Screen(
                False,
                f"the {name} arm is {arm:.3f} m - it has to be behind the wing to do anything",
            )
        volume = facts[volume_key]
        if volume is None:
            continue
        low, high = bounds
        if not low <= volume <= high:
            return Screen(
                False,
                f"{name} volume {volume:.4f} is outside [{low}, {high}] - that is not a tail",
            )

    return Screen(True)


@dataclass
class RejectionLedger:
    """Why candidates were thrown away, counted.

    The single most useful diagnostic in an optimisation run. A population 90% rejected for one
    reason means the bounds are wrong, and that shows up here long before it shows up in a flat
    objective history — where it looks like slow convergence rather than a search confined to a
    corner it cannot leave.
    """

    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0

    def record(self, reason: str) -> None:
        self.total += 1
        key = reason.split(" - ")[0].split(",")[0][:80]
        self.counts[key] = self.counts.get(key, 0) + 1

    @property
    def worst(self) -> tuple[str, int] | None:
        if not self.counts:
            return None
        return max(self.counts.items(), key=lambda kv: kv[1])

    def summary(self) -> str:
        if not self.total:
            return "nothing was rejected"
        lines = [f"{self.total} candidate(s) rejected:"]
        for reason, count in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {count:>5}x  {reason}")
        return "\n".join(lines)
