"""Stability derivatives, and the three-way check that says whether to believe them.

VSPAERO's stability run reports *perturbed values*, not derivatives: a base solve, then one solve
per perturbed variable, each with the size of its own perturbation. The derivative is
`(perturbed − base) / delta`, per degree for the angles.

**The check that makes this trustworthy.** Static margin can be arrived at three independent ways
from one stability run, and all three must agree:

1. VSPAERO reports `SM` directly.
2. VSPAERO reports the neutral point `X_np`, and `SM = (X_np − x_cg) / cref`.
3. The derivatives give `SM = −Cm_α / CL_α`.

Measured on the reference aircraft at 4°, CG 0.085 m: 0.18057, 0.18060, 0.1803. Route 3 is the one
that exercises the perturbation arithmetic, so when it agrees with the other two the whole
derivative set is being read correctly — and when it does not, every derivative below is suspect
rather than just the static margin. `consistency()` is that check, and it runs on every result.

A sign convention worth stating because it is the one everybody gets wrong: `Cm_α` negative is
statically stable, and positive static margin is stable. `Cn_β` positive is directionally stable.
`Cl_β` negative is laterally stable (dihedral effect). Three different sign conventions in one
object, and each one is the standard for its own axis.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

#: How far the three routes to static margin may differ before the result is not to be trusted.
#: They are computed from the same solve by different arithmetic, so agreement should be tight;
#: this tolerance is for floating point and VSPAERO's own rounding, not for genuine disagreement.
CONSISTENCY_TOLERANCE = 0.01

#: Conventional bands, for the "is this sane" readout. Not constraints — a canard or a tailless
#: aircraft lives outside them deliberately. They exist so an order-of-magnitude error is visible.
STATIC_MARGIN_TYPICAL = (0.05, 0.20)


def per_radian(per_degree: float) -> float:
    return per_degree * 180.0 / math.pi


@dataclass
class StabilityResult:
    """The derivative set from one stability solve, in per-radian units throughout.

    Per radian because that is what a state-space model wants, and converting once here is better
    than converting at each of the dozen places Phase 4 will read these.
    """

    # -- where it was measured -------------------------------------------------------------
    alpha_deg: float = 0.0
    beta_deg: float = 0.0
    velocity_mps: float = 0.0
    x_cg_m: float = 0.0
    sref_m2: float = 0.0
    bref_m: float = 0.0
    cref_m: float = 0.0

    # -- the base solve --------------------------------------------------------------------
    CL: float = 0.0
    CD: float = 0.0
    Cm: float = 0.0

    # -- longitudinal, per radian ----------------------------------------------------------
    CL_alpha: float = 0.0
    CD_alpha: float = 0.0
    Cm_alpha: float = 0.0

    # -- lateral-directional, per radian ---------------------------------------------------
    CY_beta: float = 0.0
    Cl_beta: float = 0.0
    Cn_beta: float = 0.0

    # -- rate derivatives, per radian of non-dimensional rate ------------------------------
    #: **These come from AVL, not VSPAERO.** VSPAERO's rate output was cross-checked against AVL
    #: on the reference aircraft with the moment reference matched, and it does not survive:
    #: Cn_r comes back positive (an aircraft that drives its own yaw), and the ratios to AVL are
    #: 28x, 58x, 25x, 17x, 157x and -49x — not one consistent factor, so it is not a
    #: non-dimensionalisation that could be corrected for. AVL's are all physically sensible with
    #: every damping sign correct. See `rate_source`.
    CL_q: float = 0.0
    Cm_q: float = 0.0
    #: Side force with roll and yaw rate. Small, and only AVL reports them, but they are what
    #: couples the Dutch roll to sideslip and leaving them at zero shifts its damping.
    CY_p: float = 0.0
    CY_r: float = 0.0
    Cl_p: float = 0.0
    Cn_p: float = 0.0
    Cl_r: float = 0.0
    Cn_r: float = 0.0
    rates_validated: bool = False

    # -- what VSPAERO reported directly ----------------------------------------------------
    neutral_point_x_m: float | None = None
    static_margin_reported: float | None = None

    #: Which code each half came from. Recorded because they are different codes: the statics
    #: are VSPAERO's and the rates are AVL's, and anything reading these needs to know that rather
    #: than infer it.
    static_source: str = "vspaero"
    rate_source: str = ""
    #: Which code supplied `Cl_beta`. Empty means VSPAERO's own, "avl" means its value was
    #: rejected as geometrically impossible and AVL's substituted - see
    #: `solvers/stability.implausible_dihedral_effect`.
    dihedral_effect_source: str = ""
    #: Derivative-by-derivative comparison against the independent code, when one was run.
    cross_check: dict[str, Any] = field(default_factory=dict)

    solver: str = ""
    notes: list[str] = field(default_factory=list)

    # ── static margin, three ways ─────────────────────────────────────────────────────────

    @property
    def side_force_height_m(self) -> float | None:
        """The height above the CG at which this derivative set's own side force must act.

        A rolling moment from sideslip is a side force times a moment arm, so
        `Cl_beta = CY_beta · z / b` and `z = Cl_beta / CY_beta · b`. That makes the two lateral
        derivatives checkable against the *geometry*: whatever produces the side force has to be
        somewhere the aircraft actually is.

        It is a side-force-weighted average height over every contributor, not the fin's own
        aerodynamic centre - a fuselage carries side force near z = 0 and pulls the average down.
        So a value below the fin root is not by itself impossible. A value far below it, on an
        aircraft whose fin sits well above the CG, is.
        """
        if not self.CY_beta or self.bref_m <= 0:
            return None
        return self.Cl_beta / self.CY_beta * self.bref_m

    @property
    def static_margin_from_derivatives(self) -> float | None:
        """SM = −Cm_α / CL_α. The route that exercises the perturbation arithmetic."""
        if self.CL_alpha == 0:
            return None
        return -self.Cm_alpha / self.CL_alpha

    @property
    def static_margin_from_neutral_point(self) -> float | None:
        """SM = (X_np − x_cg) / cref."""
        if self.neutral_point_x_m is None or self.cref_m <= 0:
            return None
        return (self.neutral_point_x_m - self.x_cg_m) / self.cref_m

    @property
    def static_margin(self) -> float | None:
        """The one to show. VSPAERO's own value where it gave one, since it is what the other two
        are checked against."""
        if self.static_margin_reported is not None:
            return self.static_margin_reported
        return self.static_margin_from_neutral_point or self.static_margin_from_derivatives

    def consistency(self) -> str:
        """Why this derivative set should not be believed, or "" if the three routes agree.

        A disagreement here is not a static-margin problem. All three routes come from one solve;
        if they part company, the perturbation values are being read wrong and *every* derivative
        in this object is suspect.
        """
        routes = {
            "reported": self.static_margin_reported,
            "neutral point": self.static_margin_from_neutral_point,
            "derivatives": self.static_margin_from_derivatives,
        }
        available = {name: value for name, value in routes.items() if value is not None}
        if len(available) < 2:
            return ""
        spread = max(available.values()) - min(available.values())
        if spread > CONSISTENCY_TOLERANCE:
            detail = ", ".join(f"{name} {value:+.4f}" for name, value in available.items())
            return (
                f"static margin disagrees between the three routes by {spread:.4f} ({detail}) - "
                f"the perturbation values are being read wrong, so every derivative here is suspect"
            )
        return ""

    # ── stability verdicts ────────────────────────────────────────────────────────────────

    @property
    def longitudinally_stable(self) -> bool:
        """Cm_α negative. The aircraft pitches nose-down when disturbed nose-up."""
        return self.Cm_alpha < 0

    @property
    def directionally_stable(self) -> bool:
        """Cn_β positive. The aircraft yaws to face the relative wind — weathercock stability."""
        return self.Cn_beta > 0

    @property
    def laterally_stable(self) -> bool:
        """Cl_β negative. Sideslip produces a roll away from it — the dihedral effect."""
        return self.Cl_beta < 0

    def rate_complaint(self) -> str:
        """Why the rate derivatives should not be believed, or "" if they look physical.

        Three signs are near-universal for a conventional aircraft, and each one is a damping term
        — the aircraft resisting its own motion. A positive value is the aircraft *driving* that
        motion, which no conventional configuration does:

            Cm_q < 0   pitch damping, from the tail's angle of attack changing as it swings
            Cl_p < 0   roll damping, from the down-going wing seeing more incidence
            Cn_r < 0   yaw damping, from the fin

        Measured on the reference aircraft, VSPAERO's raw rate output gives Cn_r = +0.0009 and
        Cl_p = -0.023 — one sign wrong and one an order of magnitude below the -0.4 a wing of this
        aspect ratio should produce. Both are consistent with the rate perturbations being
        dimensional (per rad/s) where the standard derivatives are non-dimensional (per pb/2V), but
        that has not been confirmed against an independent solver, so the values are carried,
        flagged, and not used.
        """
        wrong = [
            name for name, value in
            (("Cm_q", self.Cm_q), ("Cl_p", self.Cl_p), ("Cn_r", self.Cn_r))
            if value >= 0
        ]
        if wrong:
            return (
                f"{', '.join(wrong)} came back non-negative, which is an aircraft that drives its "
                f"own rotation rather than damping it - the rate derivatives are not usable"
            )
        return ""

    def disagreements(self, tolerance: float = 0.25) -> list[str]:
        """Which derivatives the two codes do not agree on.

        Reported rather than resolved. Two independent codes disagreeing about Cm_alpha by a
        factor of two is a real fact about how well this aircraft is known at this stage, and
        picking a winner without a third source would be inventing a confidence nobody has — the
        same reasoning that keeps three drag models side by side rather than averaging them.
        """
        return [
            name for name, row in self.cross_check.items()
            if isinstance(row, dict) and not row.get("agrees", True)
        ]

    def verdicts(self) -> dict[str, bool]:
        return {
            "longitudinal": self.longitudinally_stable,
            "directional": self.directionally_stable,
            "lateral": self.laterally_stable,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["static_margin"] = self.static_margin
        payload["verdicts"] = self.verdicts()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StabilityResult":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})
