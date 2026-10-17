"""What we are asking the solver, and how hard to ask.

Three things live here, and they are separated on purpose.

`Atmosphere` is the ISA. It is here rather than in a solver because Reynolds number is a property
of the *condition*, not of whoever is solving it — the strip-theory viscous correction, the
reference Reynolds VSPAERO wants, and the stall check all need the same numbers and must not each
carry their own copy. (In the previous generation this function lived in `kernels/sizing/energy.py`
and was the single import that tied the whole aero path to the sizing tree.)

`Fidelity` is the coarse-to-fine ladder as an enum rather than a scatter of keyword arguments.
Every knob that trades accuracy for time is resolved from it in one place, so a screening run and a
finalist run differ by one value at the call site. It is also part of the cache key: a QUICK result
must never be served to a caller that asked for HIGH.

`OperatingCondition` is the question. Notably it does NOT carry design variants or candidate lists
— varying the shape is a design-space concern that belongs above the solver, and folding it in here
is what made the previous generation's settings object accumulate a wing-library name and a list of
geometry variants it had no business knowing about.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

#: ISA sea-level constants.
SEA_LEVEL_DENSITY = 1.225        # kg/m^3
SEA_LEVEL_TEMPERATURE = 288.15   # K
SEA_LEVEL_PRESSURE = 101325.0    # Pa
LAPSE_RATE = 0.0065              # K/m, troposphere
GAS_CONSTANT = 287.053           # J/(kg K)
GAMMA = 1.4

#: Sutherland's law constants for air.
_SUTHERLAND_MU0 = 1.716e-5
_SUTHERLAND_T0 = 273.15
_SUTHERLAND_S = 110.4


def temperature(altitude_m: float) -> float:
    """ISA temperature, troposphere only. Above 11 km it holds at the tropopause value, which is
    well outside anything this tool is for but keeps the function total."""
    return max(SEA_LEVEL_TEMPERATURE - LAPSE_RATE * max(altitude_m, 0.0), 216.65)


def air_density(altitude_m: float) -> float:
    """ISA density. Closed form rather than a table lookup because this runs inside a UI loop on
    every keystroke."""
    if altitude_m <= 0:
        return SEA_LEVEL_DENSITY
    return SEA_LEVEL_DENSITY * (temperature(altitude_m) / SEA_LEVEL_TEMPERATURE) ** 4.256


def dynamic_viscosity(altitude_m: float) -> float:
    """Sutherland's law.

    The previous generation hard-coded 1.789e-5 at every Reynolds call site. At 2 km that is about
    1% optimistic, which is nothing for a reference Reynolds number and starts to matter once local
    strip Reynolds is driving an airfoil polar lookup near the low-Re knee.
    """
    t = temperature(altitude_m)
    return (
        _SUTHERLAND_MU0
        * (t / _SUTHERLAND_T0) ** 1.5
        * (_SUTHERLAND_T0 + _SUTHERLAND_S)
        / (t + _SUTHERLAND_S)
    )


def speed_of_sound(altitude_m: float) -> float:
    return math.sqrt(GAMMA * GAS_CONSTANT * temperature(altitude_m))


@dataclass(frozen=True)
class Atmosphere:
    """The air, resolved once at a given altitude."""

    altitude_m: float = 0.0

    @property
    def density(self) -> float:
        return air_density(self.altitude_m)

    @property
    def viscosity(self) -> float:
        return dynamic_viscosity(self.altitude_m)

    @property
    def temperature(self) -> float:
        return temperature(self.altitude_m)

    @property
    def speed_of_sound(self) -> float:
        return speed_of_sound(self.altitude_m)

    def reynolds(self, velocity_mps: float, length_m: float) -> float:
        """Re = rho V L / mu. Returns 0.0 for a non-positive length rather than raising, because a
        degenerate strip is a thing the integrator skips, not an error."""
        if length_m <= 0:
            return 0.0
        return self.density * velocity_mps * length_m / self.viscosity

    def mach(self, velocity_mps: float) -> float:
        return velocity_mps / self.speed_of_sound


@dataclass(frozen=True)
class FidelitySettings:
    """The knobs one rung of the ladder resolves to."""

    alpha_points: int
    wake_iterations: int
    #: OpenVSP surface tessellation. The STANDARD values are the ones that were found by
    #: measurement to keep a thin symmetric tail section healthy at model scale; going below them is
    #: how a mesh silently degenerates into a solve that returns zeros instead of failing.
    tess_w: int
    sect_tess_u: int


class Fidelity(Enum):
    """Coarse to fine. One value at the call site changes mesh, wake and sweep density together.

    QUICK exists for optimiser screening: thousands of candidates where the only question is "is
    this shape still in the running", and an answer that is consistently coarse ranks them
    correctly even when its absolute numbers are off. STANDARD is normal design work. HIGH is for
    the handful of finalists and for anything that will be compared against CFD or a tunnel.

    QUICK's mesh is deliberately still above the degeneracy threshold rather than as coarse as
    VSPAERO will accept. A screening rung that occasionally returns zeros is worse than useless in
    an optimiser: it does not fail, it just reports a shape as spectacularly good.
    """

    QUICK = "quick"
    STANDARD = "standard"
    HIGH = "high"

    @property
    def settings(self) -> FidelitySettings:
        return _FIDELITY[self]


_FIDELITY: dict[Fidelity, FidelitySettings] = {
    Fidelity.QUICK: FidelitySettings(alpha_points=3, wake_iterations=3, tess_w=17, sect_tess_u=6),
    Fidelity.STANDARD: FidelitySettings(alpha_points=9, wake_iterations=5, tess_w=33, sect_tess_u=12),
    Fidelity.HIGH: FidelitySettings(alpha_points=17, wake_iterations=9, tess_w=41, sect_tess_u=16),
}


#: The 2D section solvers Drag Model C can be built on.
SECTION_SOURCES = ("neuralfoil", "xfoil")

#: Which components go into the solve. Reference values always come from the wing regardless, so
#: polars from different scopes stay comparable — and that comparison (what does adding the tail
#: cost in drag?) is the whole reason scopes exist.
SCOPES = ("wing", "wing_tail", "aircraft")


@dataclass
class OperatingCondition:
    """One question for the solver: this shape, at this speed and altitude, over these angles."""

    velocity_mps: float = 20.0
    altitude_m: float = 0.0
    alpha_start_deg: float = -4.0
    alpha_end_deg: float = 12.0
    #: None means "take the count from the fidelity rung", which is the normal case. Set it only to
    #: override the ladder for one run.
    alpha_points: int | None = None
    #: Sideslip. A single value is a symmetric solve; a range drives a beta sweep for its own sake.
    #:
    #: It is NOT how the lateral-directional static derivatives are obtained, though this comment
    #: said so until 2026-09-02. The Phase 0 probe looked for a `StabilityType` input, did not find
    #: one, and concluded OpenVSP had no stability mode — but the input is called **`UnsteadyType`**
    #: and `STABILITY_DEFAULT` runs the full analysis, which is what `solvers/stability.py` has
    #: actually done since Phase 3. It perturbs beta internally, so the derivatives need no sweep
    #: here. An input that is not where you looked is not an input that does not exist.
    beta_start_deg: float = 0.0
    beta_end_deg: float = 0.0
    beta_points: int = 1
    fidelity: Fidelity = Fidelity.STANDARD
    scope: str = "aircraft"
    #: False: every surface in the THIN set (vortex-lattice). True: everything THICK (panels).
    #: Thick is more accurate and much slower, and it is the only treatment that represents a
    #: fuselage at all.
    thick: bool = False
    #: Moment reference point. Static margin and Cm_alpha are meaningless without one, and VSPAERO
    #: takes it directly as Xcg/Ycg/Zcg.
    xyz_ref: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Reference values. None means "take them from the wing", which is almost always right.
    sref_m2: float | None = None
    bref_m: float | None = None
    cref_m: float | None = None
    #: Which 2D section solver supplies the profile drag for Drag Model C.
    #:
    #: "neuralfoil" is the production path — milliseconds, and it cannot fail to converge, which
    #: is what makes it safe underneath an optimiser. "xfoil" is the ground truth, worth spending
    #: on a handful of finalists. It is on the condition rather than passed separately because it
    #: changes the answer, and everything that changes the answer has to reach the cache key.
    section_source: str = "neuralfoil"
    #: Control deflections in degrees, keyed by `ControlSurface.name`.
    control_deflections: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {self.scope!r}")
        if self.section_source not in SECTION_SOURCES:
            raise ValueError(
                f"section_source must be one of {SECTION_SOURCES}, got {self.section_source!r}"
            )

    @property
    def atmosphere(self) -> Atmosphere:
        return Atmosphere(self.altitude_m)

    def alphas(self) -> np.ndarray:
        n = self.alpha_points if self.alpha_points is not None else self.fidelity.settings.alpha_points
        return np.linspace(self.alpha_start_deg, self.alpha_end_deg, max(2, n))

    def betas(self) -> np.ndarray:
        if self.beta_points <= 1:
            return np.array([self.beta_start_deg])
        return np.linspace(self.beta_start_deg, self.beta_end_deg, self.beta_points)

    @property
    def wake_iterations(self) -> int:
        return self.fidelity.settings.wake_iterations

    def mach(self) -> float:
        return self.atmosphere.mach(self.velocity_mps)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fidelity"] = self.fidelity.value
        payload["xyz_ref"] = list(self.xyz_ref)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperatingCondition":
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in payload.items() if k in known}
        if "fidelity" in kwargs:
            kwargs["fidelity"] = Fidelity(kwargs["fidelity"])
        if "xyz_ref" in kwargs:
            kwargs["xyz_ref"] = tuple(kwargs["xyz_ref"])
        return cls(**kwargs)
