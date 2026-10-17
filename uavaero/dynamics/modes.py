"""Eigenvalues into named modes.

Five modes, and naming them correctly is the part that is easy to get wrong. The tempting rule is
"the faster longitudinal pair is the short period" — and it fails, because on a light airframe with
a large static margin the phugoid can rise until the two pairs overlap in frequency, at which point
frequency-ordering labels them backwards and everything downstream inherits the swap silently.

So classification is by **participation factor**: which states actually move in each eigenvector.
The short period is the mode where w and q dominate; the phugoid is where u and θ do. Dutch roll is
oscillatory with sideslip in it; roll subsidence is the fast real root dominated by p; spiral is the
slow real root dominated by φ. That is what the modes *are*, rather than where they usually sit, and
it holds when the frequencies do not.

`Mode.name` is therefore derived from the eigenvector, never from the order the eigenvalues came
back in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Below this the eigenvalue is a rigid-body integrator (heading, position) rather than a mode.
NEGLIGIBLE = 1e-8

#: Damping-ratio floors from the handling-qualities literature, for the "is this acceptable"
#: readout. Not constraints — an unstable spiral is normal and flyable — but a Dutch roll below
#: 0.08 is genuinely unpleasant and worth colouring.
ACCEPTABLE_DAMPING = {
    "short period": 0.30,
    "dutch roll": 0.08,
    "phugoid": 0.04,
}


@dataclass
class Mode:
    """One eigenvalue, named and described."""

    eigenvalue: complex
    name: str = "unclassified"
    participation: dict[str, float] = field(default_factory=dict)
    #: The eigenvector itself, per state, complex. Participation is its magnitudes normalised —
    #: enough to *name* a mode but not to draw one, because the phase between states is what makes
    #: a Dutch roll roll-then-yaw rather than roll-and-yaw together.
    vector: dict[str, complex] = field(default_factory=dict)
    #: The state names, in the order the matrix used them.
    states: tuple[str, ...] = ()

    @property
    def real(self) -> float:
        return float(self.eigenvalue.real)

    @property
    def imag(self) -> float:
        return float(abs(self.eigenvalue.imag))

    @property
    def oscillatory(self) -> bool:
        return self.imag > NEGLIGIBLE

    @property
    def stable(self) -> bool:
        return self.real < 0

    @property
    def natural_frequency(self) -> float:
        """rad/s. For a real root this is just its magnitude, which is the convention that makes
        the two cases comparable on one plot."""
        return float(abs(self.eigenvalue))

    @property
    def damping_ratio(self) -> float | None:
        """None for a real root — a non-oscillatory mode does not have one, and reporting 1.0
        would invite the reader to conclude it is critically damped, which is a different claim."""
        if not self.oscillatory:
            return None
        magnitude = self.natural_frequency
        return float(-self.real / magnitude) if magnitude > 0 else None

    @property
    def period(self) -> float | None:
        """Seconds. None for a real root."""
        return float(2.0 * math.pi / self.imag) if self.oscillatory else None

    @property
    def time_constant(self) -> float | None:
        """Seconds to decay to 1/e, for a real root. None for an oscillatory one."""
        if self.oscillatory or abs(self.real) < NEGLIGIBLE:
            return None
        return float(-1.0 / self.real)

    @property
    def time_to_half(self) -> float | None:
        """Seconds to halve. Negative-valued for an unstable mode is meaningless, so an unstable
        one reports its doubling time through `time_to_double` instead."""
        if self.real >= 0 or abs(self.real) < NEGLIGIBLE:
            return None
        return float(math.log(2.0) / -self.real)

    @property
    def time_to_double(self) -> float | None:
        """Seconds to double, for an unstable mode. This is the number that decides whether an
        unstable spiral matters: forty seconds is a non-event, four is not."""
        if self.real <= 0 or abs(self.real) < NEGLIGIBLE:
            return None
        return float(math.log(2.0) / self.real)

    @property
    def acceptable(self) -> bool | None:
        """Against the handling-qualities floor for this mode, where one applies."""
        floor = ACCEPTABLE_DAMPING.get(self.name)
        if floor is None or self.damping_ratio is None:
            return None
        return self.damping_ratio >= floor

    def summary(self) -> str:
        if self.oscillatory:
            state = "stable" if self.stable else "UNSTABLE"
            return (
                f"{self.name}: {state}, damping {self.damping_ratio:.3f}, "
                f"period {self.period:.2f} s, wn {self.natural_frequency:.3f} rad/s"
            )
        if self.stable:
            return f"{self.name}: stable, time constant {self.time_constant:.2f} s"
        doubling = self.time_to_double
        return (
            f"{self.name}: UNSTABLE, doubling time "
            f"{doubling:.1f} s" if doubling else f"{self.name}: UNSTABLE"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "real": self.real, "imag": float(self.eigenvalue.imag), "name": self.name,
            "oscillatory": self.oscillatory, "stable": self.stable,
            "natural_frequency": self.natural_frequency, "damping_ratio": self.damping_ratio,
            "period": self.period, "time_constant": self.time_constant,
            "time_to_half": self.time_to_half, "time_to_double": self.time_to_double,
            "participation": self.participation, "acceptable": self.acceptable,
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Mode":
        mode = cls(
            eigenvalue=complex(payload["real"], payload["imag"]),
            name=payload.get("name", "unclassified"),
            participation=payload.get("participation") or {},
        )
        return mode


def _participation(vector: np.ndarray, states: tuple[str, ...]) -> dict[str, float]:
    """How much each state moves in this eigenvector, normalised to sum to one.

    Magnitudes, not components: the phase between states is real information but it is not what
    decides which mode this is, and taking magnitudes keeps the classification from depending on
    an arbitrary eigenvector scaling.
    """
    magnitudes = np.abs(vector)
    total = float(np.sum(magnitudes))
    if total <= 0:
        return {name: 0.0 for name in states}
    return {name: float(m / total) for name, m in zip(states, magnitudes)}


def _classify_longitudinal(modes: list[Mode]) -> None:
    """Short period is w-and-q dominated; phugoid is u-and-theta dominated.

    Assigned by ranking rather than thresholding: whichever oscillatory pair has more of its motion
    in w and q is the short period, and the other is the phugoid. That works even when the two sit
    at the same frequency, which is exactly the case a frequency test gets backwards.
    """
    oscillatory = [m for m in modes if m.oscillatory]
    if not oscillatory:
        for mode in modes:
            mode.name = "longitudinal real root"
        return

    def fastness(mode: Mode) -> float:
        return mode.participation.get("w", 0.0) + mode.participation.get("q", 0.0)

    ranked = sorted(oscillatory, key=fastness, reverse=True)
    ranked[0].name = "short period"
    if len(ranked) > 1:
        ranked[1].name = "phugoid"
    for mode in modes:
        if mode.name == "unclassified":
            mode.name = "longitudinal real root"


def _classify_lateral(modes: list[Mode]) -> None:
    """Dutch roll oscillates; of the real roots, the fast one is roll subsidence and the slow one
    is the spiral.

    The real roots are separated by speed rather than by participation because they are frequently
    both dominated by the same state — a heavily-damped spiral and a roll subsidence can look alike
    in the eigenvector while differing by two orders of magnitude in time constant, which is
    unambiguous.
    """
    for mode in modes:
        if mode.oscillatory:
            mode.name = "dutch roll"

    real_roots = [m for m in modes if not m.oscillatory and abs(m.real) > NEGLIGIBLE]
    real_roots.sort(key=lambda m: abs(m.real), reverse=True)
    if real_roots:
        real_roots[0].name = "roll subsidence"
    if len(real_roots) > 1:
        real_roots[-1].name = "spiral"
    for mode in modes:
        if mode.name == "unclassified":
            mode.name = "lateral integrator"


def analyse(matrix: np.ndarray, states: tuple[str, ...], lateral: bool = False) -> list[Mode]:
    """Eigenvalues of A, named by which states move in them.

    Conjugate pairs are collapsed to one entry: they are one physical mode, and listing both halves
    makes a four-state model look like it has six behaviours.
    """
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=float))

    modes: list[Mode] = []
    seen: list[complex] = []
    for index, value in enumerate(values):
        if any(abs(value - np.conj(other)) < 1e-9 and abs(value.imag) > NEGLIGIBLE
               for other in seen):
            continue
        seen.append(complex(value))
        modes.append(Mode(
            eigenvalue=complex(value),
            participation=_participation(vectors[:, index], states),
            vector={
                name: complex(component)
                for name, component in zip(states, vectors[:, index])
            },
            states=tuple(states),
        ))

    if lateral:
        _classify_lateral(modes)
    else:
        _classify_longitudinal(modes)
    return sorted(modes, key=lambda m: -m.natural_frequency)
