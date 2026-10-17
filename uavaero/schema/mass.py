"""Mass and inertia — stated, never sized.

Dynamic modes cannot be computed without these. Short-period frequency goes as the square root of
Cm_α·q·S·c over Iyy; Dutch roll needs Izz; every dimensional derivative divides by mass. There is
no version of Phase 4 that avoids them.

**This is not the sizing loop coming back.** What is out of scope in this tool is *deriving* mass
from requirements — the constraint diagram, the energy budget, the mass fixed point that made the
previous generation's aircraft a by-product of its brief. Stating what your aircraft weighs is a
different act entirely, the same kind as stating how fast it is flying or where its CG is. You
already author `xyz_ref`; this is its two companions.

**Inertia is the hard part to state, so there are two ways to.** Almost nobody knows their aircraft's
Izz in kg·m², but most people can estimate a radius of gyration as a fraction of span or length, and
those fractions are well-bounded for conventional configurations. `from_radii_of_gyration` is that
route, and it is explicitly an estimate: a result computed through it says so, because a mode
frequency is only as good as the inertia under it and an estimated Iyy should not produce a
confidently-quoted short period.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

#: Radius of gyration as a fraction of the reference length, for a conventional light aircraft.
#: Roll about span, pitch about length, yaw about the mean of the two. These are the usual bounds
#: quoted for the class; they are a starting point to be replaced by a real weights breakdown or a
#: swing test, not an answer.
TYPICAL_RADII = {"roll": 0.25, "pitch": 0.30, "yaw": 0.35}


@dataclass
class MassProperties:
    """What the aircraft weighs and how that mass is distributed.

    Inertias are about the CG in body axes, kg·m². `ixz` is the product of inertia that couples
    roll and yaw; it is small for most conventional aircraft and zero is a fair starting value, but
    it is carried because on a swept or high-wing configuration it is not negligible and leaving it
    out quietly changes the Dutch roll.
    """

    mass_kg: float = 0.0
    ixx: float = 0.0  # roll
    iyy: float = 0.0  # pitch
    izz: float = 0.0  # yaw
    ixz: float = 0.0  # roll-yaw coupling
    #: True when the inertias came from radii of gyration rather than a real breakdown. Travels
    #: with every result computed from them.
    estimated: bool = False

    def __post_init__(self) -> None:
        if self.mass_kg < 0:
            raise ValueError(f"mass must not be negative, got {self.mass_kg}")

    @property
    def complete(self) -> bool:
        """Whether a dynamics run is possible at all. A zero inertia divides by zero somewhere
        downstream and produces an infinite frequency, which plots as an empty chart rather than
        as an error."""
        return self.mass_kg > 0 and self.ixx > 0 and self.iyy > 0 and self.izz > 0

    def missing(self) -> str:
        """What still has to be entered, phrased for the person entering it."""
        gaps = [
            name for name, value in
            (("mass", self.mass_kg), ("Ixx", self.ixx), ("Iyy", self.iyy), ("Izz", self.izz))
            if value <= 0
        ]
        return "" if not gaps else f"{', '.join(gaps)} not entered"

    @property
    def tensor(self) -> np.ndarray:
        """The 3x3 inertia tensor in body axes. Note the sign: the off-diagonal terms carry a
        minus, which is the definition of the product of inertia and the single easiest place in
        rigid-body dynamics to lose a sign."""
        return np.array([
            [self.ixx, 0.0, -self.ixz],
            [0.0, self.iyy, 0.0],
            [-self.ixz, 0.0, self.izz],
        ])

    @classmethod
    def from_radii_of_gyration(
        cls,
        mass_kg: float,
        span_m: float,
        length_m: float,
        roll: float = TYPICAL_RADII["roll"],
        pitch: float = TYPICAL_RADII["pitch"],
        yaw: float = TYPICAL_RADII["yaw"],
    ) -> "MassProperties":
        """Inertias from radii of gyration expressed as fractions of span and length.

        I = m·(k·L)². Roll is referenced to span, pitch to length, and yaw to the mean of the two —
        which is the conventional treatment and is where the usual fraction bounds come from.

        The result is marked `estimated`, and that mark is meant to survive all the way to the
        screen. A short-period frequency computed from a guessed Iyy is a guessed frequency.
        """
        yaw_reference = 0.5 * (span_m + length_m)
        return cls(
            mass_kg=mass_kg,
            ixx=mass_kg * (roll * span_m) ** 2,
            iyy=mass_kg * (pitch * length_m) ** 2,
            izz=mass_kg * (yaw * yaw_reference) ** 2,
            ixz=0.0,
            estimated=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MassProperties":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})
