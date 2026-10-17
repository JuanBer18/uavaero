"""A mode as motion: the eigenvector turned into an attitude you can watch.

A damping ratio of 0.29 and a period of 0.85 s is a correct description of Dutch roll and it tells
you almost nothing about what the aircraft *does*. The characteristic behaviour of that mode is
that the roll leads the yaw by most of a quarter cycle, so the aeroplane wallows — and that lives
entirely in the **phase** between the eigenvector's components, which the mode table throws away.

`participation` keeps the magnitudes because that is all that is needed to *name* a mode. This
keeps the phases too, because that is what is needed to *draw* one.

**The amplitude is arbitrary and says so.** An eigenvector has no natural scale: it is a direction
in state space, and multiplying it by anything is the same mode. So the states are normalised to
the largest angular excursion and scaled to a stated display amplitude, and nothing here should be
read as "the aircraft rolls this far" — only as "the aircraft rolls this much *relative to* how
much it yaws".

**The decay is real, though.** Playing a mode at constant amplitude makes a divergent spiral and a
well-damped roll subsidence look identical, which is exactly the distinction worth seeing. The
envelope is `exp(sigma·t)` with the mode's own real part, so a stable mode visibly dies out and an
unstable one visibly runs away.

Pure geometry and arithmetic: no Qt, no VTK. The viewport consumes what this produces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from uavaero.dynamics.modes import Mode

#: Peak attitude excursion in the animation, degrees. Chosen to read clearly at a normal viewport
#: zoom without the aircraft leaving the frame; it is a display choice, not a physical amplitude.
DISPLAY_AMPLITUDE_DEG = 12.0

#: Stop sampling a decaying mode once its envelope falls below this. Past it the aircraft is
#: visibly still, and a loop padded with stillness reads as a stutter rather than as damping.
VISIBLE_FLOOR = 0.06

#: Which state carries which rotation. Everything else in the state vector (u, v, w) is a velocity
#: perturbation, which shows as attitude only through the angles.
ROTATION_STATES = {
    "phi": "roll",
    "theta": "pitch",
    "psi": "yaw",
    # Rates rotate the aircraft too, and on a mode where the angle barely moves they are the whole
    # story - a short period is mostly q, and pitch rate integrated over a cycle is what you see.
    "p": "roll",
    "q": "pitch",
    "r": "yaw",
}


@dataclass
class Pose:
    """The aircraft's attitude at one instant of a mode animation, in degrees."""

    time_s: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    #: The decay envelope at this instant, 1.0 at t = 0. Below 1 the mode is dying out; above 1 it
    #: is running away, which is the whole point of animating an unstable mode.
    envelope: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s, "roll_deg": self.roll_deg, "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg, "envelope": self.envelope,
        }


@dataclass
class ModeShape:
    """One mode, sampled as a sequence of attitudes."""

    name: str
    poses: list[Pose] = field(default_factory=list)
    period_s: float | None = None
    damping_ratio: float | None = None
    stable: bool = True
    #: What the animation is showing, in words, because an arbitrary amplitude needs saying.
    caption: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "period_s": self.period_s,
            "damping_ratio": self.damping_ratio, "stable": self.stable,
            "caption": self.caption, "poses": [p.to_dict() for p in self.poses],
        }


def _rotation_components(mode: Mode) -> dict[str, complex]:
    """The eigenvector's contribution to each rotation axis.

    A rate and its angle both drive the same axis - on a short period the pitch angle barely moves
    while q dominates, and animating only theta would show almost nothing. The rate is integrated
    (divided by the eigenvalue) so it is expressed as the angle it produces over the mode's own
    timescale rather than added to an angle as if it were one.
    """
    axes: dict[str, complex] = {"roll": 0j, "pitch": 0j, "yaw": 0j}
    lam = mode.eigenvalue
    for state, component in mode.vector.items():
        axis = ROTATION_STATES.get(state)
        if axis is None:
            continue
        if state in ("p", "q", "r"):
            # x_dot = lambda x for an eigenmode, so the angle is the rate over lambda.
            component = component / lam if abs(lam) > 1e-12 else 0j
        axes[axis] += component
    return axes


def _default_duration(mode: Mode) -> float:
    """How long to sample, so the loop is mostly the part worth watching.

    Three periods is the obvious answer and it is wrong for a well-damped mode: Dutch roll on the
    reference aircraft has a 0.85 s period and a 0.3 s half-life, so three periods spends most of
    its length showing a stationary aeroplane and the loop reads as a stutter.

    So the window is driven by the **decay**, not by the period: it runs a little past the point
    where the motion reaches `VISIBLE_FLOOR`, and is capped at three periods so a lightly damped
    mode still terminates. Half a period is the lower bound, enough to read the sense of the motion.

    Forcing a period and a half - the first thing tried here - is wrong for exactly the mode people
    most want to see: a short period at zeta 0.56 dies in a third of a cycle, and padding it out to
    a period and a half misrepresents a damped snap as an oscillation that happens to stop.

    A divergent mode has no decay to bound it and is capped by periods alone; watching it run away
    is the point.
    """
    if mode.period:
        window = 3.0 * mode.period
        if mode.eigenvalue.real < -1e-9:
            decayed = math.log(VISIBLE_FLOOR) / mode.eigenvalue.real
            window = min(window, max(1.4 * decayed, 0.5 * mode.period))
        return float(window)
    if mode.time_constant:
        return float(4.0 * abs(mode.time_constant))
    return 5.0


def mode_shape(
    mode: Mode,
    duration_s: float | None = None,
    samples: int = 180,
    amplitude_deg: float = DISPLAY_AMPLITUDE_DEG,
) -> ModeShape:
    """Sample one mode's motion over time.

    `duration_s` defaults to something that shows the mode's character: a few periods for an
    oscillation, a few time constants for a pure convergence or divergence.
    """
    axes = _rotation_components(mode)
    peak = max((abs(v) for v in axes.values()), default=0.0)

    if duration_s is None:
        duration_s = _default_duration(mode)
    duration_s = max(float(duration_s), 1e-3)

    scale = (amplitude_deg / peak) if peak > 1e-12 else 0.0
    sigma, omega = mode.eigenvalue.real, mode.eigenvalue.imag

    poses: list[Pose] = []
    for index in range(max(int(samples), 2)):
        t = duration_s * index / (max(int(samples), 2) - 1)
        # Re{ v * exp(lambda t) } is the physical motion of the mode. Written with the envelope
        # separated so it can be reported: an animation that hides its own decay makes a divergent
        # spiral look like a stable one.
        envelope = math.exp(sigma * t)
        rotor = complex(math.cos(omega * t), math.sin(omega * t))
        values = {
            axis: scale * envelope * (component * rotor).real
            for axis, component in axes.items()
        }
        poses.append(Pose(
            time_s=t, envelope=envelope,
            roll_deg=values["roll"], pitch_deg=values["pitch"], yaw_deg=values["yaw"],
        ))

    return ModeShape(
        name=mode.name,
        poses=poses,
        period_s=mode.period,
        damping_ratio=mode.damping_ratio,
        stable=mode.stable,
        caption=_caption(mode, axes, amplitude_deg),
    )


def _caption(mode: Mode, axes: dict[str, complex], amplitude_deg: float) -> str:
    """What the viewer is looking at, including the fact that the amplitude is made up."""
    ordered = sorted(axes.items(), key=lambda kv: -abs(kv[1]))
    leading = [name for name, value in ordered if abs(value) > 1e-9]
    if not leading:
        return f"{mode.name}: no attitude motion - this mode moves speed, not the aircraft's pose."

    motion = leading[0]
    lead_phase = ""
    if len(leading) > 1 and abs(axes[leading[1]]) > 0.15 * abs(axes[leading[0]]):
        degrees = math.degrees(
            math.atan2(axes[leading[0]].imag, axes[leading[0]].real)
            - math.atan2(axes[leading[1]].imag, axes[leading[1]].real)
        )
        degrees = (degrees + 180.0) % 360.0 - 180.0
        lead_phase = (
            f", with {motion} leading {leading[1]} by {abs(degrees):.0f} deg"
            if abs(degrees) > 15.0 else f", {motion} and {leading[1]} together"
        )

    fate = (
        f"decaying to half in {mode.time_to_half:.1f} s" if mode.stable and mode.time_to_half
        else f"doubling in {mode.time_to_double:.1f} s" if mode.time_to_double
        else "neutrally stable"
    )
    return (
        f"{mode.name}: mostly {motion}{lead_phase}, {fate}. "
        f"Amplitude is normalised to {amplitude_deg:.0f} deg for display - the *ratio* between "
        f"axes is the physical part, the absolute size is not."
    )


def shapes_for(modes: list[Mode], **kwargs) -> list[ModeShape]:
    """Every mode that actually moves the aircraft's attitude, in the order given."""
    out: list[ModeShape] = []
    for mode in modes:
        if not mode.vector:
            continue
        shape = mode_shape(mode, **kwargs)
        if any(
            abs(p.roll_deg) + abs(p.pitch_deg) + abs(p.yaw_deg) > 1e-6 for p in shape.poses
        ):
            out.append(shape)
    return out
