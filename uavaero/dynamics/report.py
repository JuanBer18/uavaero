"""The dynamics result: both models, all five modes, and what to distrust about them.

One object because the two models are not independent findings — they are two halves of one
aircraft at one flight condition, and reading the Dutch roll without knowing the trim mismatch or
the estimated inertia behind it is how a number gets quoted out of the context that qualifies it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from uavaero.dynamics import statespace
from uavaero.dynamics.modes import Mode, analyse
from uavaero.dynamics.statespace import (
    LATERAL_STATES,
    LONGITUDINAL_STATES,
    TrimPoint,
    lateral_matrix,
    longitudinal_matrix,
    phugoid_approximation,
    short_period_approximation,
)
from uavaero.schema.condition import OperatingCondition
from uavaero.schema.mass import MassProperties
from uavaero.schema.stability import StabilityResult

#: How far an eigenvalue may sit from its closed-form approximation before it is worth reporting.
#: The approximations are two-degree-of-freedom reductions of a four-state model, so they are not
#: expected to be exact — this is a "something is structurally wrong" threshold, not a tolerance.
APPROXIMATION_TOLERANCE = 0.35


@dataclass
class DynamicsReport:
    """Every mode of one aircraft at one flight condition."""

    trim: TrimPoint
    longitudinal: list[Mode] = field(default_factory=list)
    lateral: list[Mode] = field(default_factory=list)
    a_longitudinal: np.ndarray | None = None
    a_lateral: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def modes(self) -> list[Mode]:
        return self.longitudinal + self.lateral

    def named(self, name: str) -> Mode | None:
        return next((m for m in self.modes if m.name == name), None)

    @property
    def unstable(self) -> list[Mode]:
        return [m for m in self.modes if not m.stable and abs(m.real) > 1e-8]

    def summary(self) -> str:
        """One line. A spiral divergence is called out separately because it is the one instability
        that is entirely normal — most aeroplanes have one and it is flown without noticing."""
        bad = [m for m in self.unstable if m.name != "spiral"]
        if bad:
            return "UNSTABLE: " + ", ".join(m.name for m in bad)
        spiral = self.named("spiral")
        if spiral is not None and not spiral.stable:
            doubling = spiral.time_to_double
            return (
                f"stable except a divergent spiral, doubling in {doubling:.0f} s"
                if doubling else "stable except a divergent spiral"
            )
        return "all modes stable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trim": {
                "velocity_mps": self.trim.velocity_mps,
                "cl_required": self.trim.cl_required,
                "alpha_deg": self.trim.alpha_deg,
                "alpha_measured_deg": self.trim.alpha_measured_deg,
                "mass_kg": self.trim.mass_kg,
            },
            "longitudinal": [m.to_dict() for m in self.longitudinal],
            "lateral": [m.to_dict() for m in self.lateral],
            "a_longitudinal": (
                self.a_longitudinal.tolist() if self.a_longitudinal is not None else None
            ),
            "a_lateral": self.a_lateral.tolist() if self.a_lateral is not None else None,
            "notes": list(self.notes),
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DynamicsReport":
        trim_payload = payload.get("trim") or {}
        trim = TrimPoint(
            velocity_mps=trim_payload.get("velocity_mps", 0.0),
            dynamic_pressure=0.0,
            cl_required=trim_payload.get("cl_required", 0.0),
            alpha_deg=trim_payload.get("alpha_deg", 0.0),
            alpha_measured_deg=trim_payload.get("alpha_measured_deg", 0.0),
            mass_kg=trim_payload.get("mass_kg", 0.0),
        )
        return cls(
            trim=trim,
            longitudinal=[Mode.from_dict(m) for m in payload.get("longitudinal") or []],
            lateral=[Mode.from_dict(m) for m in payload.get("lateral") or []],
            a_longitudinal=(
                np.array(payload["a_longitudinal"]) if payload.get("a_longitudinal") else None
            ),
            a_lateral=np.array(payload["a_lateral"]) if payload.get("a_lateral") else None,
            notes=list(payload.get("notes") or []),
        )


def analyse_aircraft(
    derivatives: StabilityResult,
    mass: MassProperties,
    condition: OperatingCondition,
) -> DynamicsReport:
    """Derivatives plus mass into named modes, with everything that qualifies them attached."""
    if not mass.complete:
        raise ValueError(f"mass properties incomplete: {mass.missing()}")

    trim = statespace.trim_level_flight(derivatives, mass, condition)
    a_long = longitudinal_matrix(derivatives, mass, trim)
    a_lat = lateral_matrix(derivatives, mass, trim)

    report = DynamicsReport(
        trim=trim,
        longitudinal=analyse(a_long, LONGITUDINAL_STATES, lateral=False),
        lateral=analyse(a_lat, LATERAL_STATES, lateral=True),
        a_longitudinal=a_long,
        a_lateral=a_lat,
    )
    report.notes.extend(_qualifications(derivatives, mass, trim, report))
    return report


def _qualifications(
    derivatives: StabilityResult,
    mass: MassProperties,
    trim: TrimPoint,
    report: DynamicsReport,
) -> list[str]:
    """Everything that makes these numbers less certain than they look."""
    notes: list[str] = []

    complaint = trim.complaint()
    if complaint:
        notes.append(complaint)

    if mass.estimated:
        notes.append(
            "Inertias were estimated from radii of gyration, not weighed - every frequency here "
            "scales with the square root of an inertia, so treat them as indicative"
        )

    if not derivatives.rates_validated:
        notes.append(
            "The rate derivatives did not pass their damping check, so every damping ratio below "
            "is unreliable"
        )

    notes.append(
        "C_m_alphadot is not measured by either solver and is taken as zero. It normally supplies "
        "roughly a third of short-period damping, so that damping ratio is an under-estimate."
    )

    # The independent check: eigenvalues against the closed-form reductions.
    for name, (omega, zeta) in (
        ("short period", short_period_approximation(derivatives, mass, trim)),
        ("phugoid", phugoid_approximation(trim, derivatives.CD)),
    ):
        mode = report.named(name)
        if mode is None or omega <= 0 or mode.natural_frequency <= 0:
            continue
        error = abs(mode.natural_frequency - omega) / omega
        if error > APPROXIMATION_TOLERANCE:
            notes.append(
                f"The {name} eigenvalue ({mode.natural_frequency:.3f} rad/s) is {error * 100:.0f}% "
                f"from its closed-form approximation ({omega:.3f} rad/s) - the matrix may be wrong"
            )
    return notes


def time_response(
    matrix: np.ndarray,
    initial: np.ndarray,
    duration_s: float = 60.0,
    points: int = 600,
) -> tuple[np.ndarray, np.ndarray]:
    """Free response to an initial disturbance: (time, states).

    Solved by matrix exponential rather than by stepping an integrator. The system is linear and
    time-invariant, so the exponential is exact at every sample and there is no step size to get
    wrong — a stiff pair like a short period sitting on a phugoid is precisely where an explicit
    integrator quietly loses the fast mode.
    """
    from scipy.linalg import expm

    matrix = np.asarray(matrix, dtype=float)
    initial = np.asarray(initial, dtype=float)
    times = np.linspace(0.0, duration_s, points)
    step = times[1] - times[0] if points > 1 else duration_s

    advance = expm(matrix * step)
    states = np.zeros((points, matrix.shape[0]))
    states[0] = initial
    for i in range(1, points):
        states[i] = advance @ states[i - 1]
    return times, states


def excite(mode: Mode, states: tuple[str, ...], magnitude: float = 1.0) -> np.ndarray:
    """An initial condition that excites one mode and as little else as possible.

    Built from the mode's own participation, so asking to see the phugoid gives a disturbance in
    the states the phugoid actually moves rather than a generic nudge that rings every mode at once.
    """
    vector = np.array([mode.participation.get(name, 0.0) for name in states], dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector * (magnitude / norm) if norm > 0 else vector
