"""Non-dimensional derivatives plus mass, into the two linear models the modes come from.

The classical decoupled treatment: a 4-state longitudinal model in [u, w, q, θ] and a 4-state
lateral-directional model in [v, p, r, φ], both about a level-flight trim point, both in stability
axes. Every flight-dynamics text writes them this way, which is the point — it is what makes the
result checkable against something other than itself.

*(The plan called for numerically linearising a nonlinear 13-state model instead. That is the more
general machine and it is the right one for simulating a manoeuvre, but the five modes come from
the linear model either way, and building it analytically means the matrices can be compared
term-by-term with a textbook and the eigenvalues against closed-form approximations. A numerical
linearisation would have added a trim solve and a finite-difference step between the derivatives
and the answer, and every one of those is somewhere for an error to hide during validation.)*

**What is not measured, and therefore is zero.** Neither VSPAERO nor AVL produces the speed
derivatives (C_D_u, C_L_u, C_m_u) or the lag-of-downwash term C_m_αdot. The speed derivatives are
genuinely near zero at low Mach and their absence costs almost nothing. C_m_αdot is not negligible
— it typically contributes a third of the short-period damping — so a damping ratio from this model
is an under-estimate, and `Modes` says so rather than letting it pass as the answer.

Sign conventions follow Etkin: x forward, y starboard, z down; positive elevator trailing-edge
down. The one that catches people is the inertia tensor's off-diagonal, which carries a minus by
definition — see `MassProperties.tensor`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavaero.schema.condition import OperatingCondition
from uavaero.schema.mass import MassProperties
from uavaero.schema.stability import StabilityResult

GRAVITY = 9.80665

LONGITUDINAL_STATES = ("u", "w", "q", "theta")
LATERAL_STATES = ("v", "p", "r", "phi")


@dataclass
class TrimPoint:
    """Level flight at one speed: what lift coefficient it takes, and at what incidence."""

    velocity_mps: float
    dynamic_pressure: float
    cl_required: float
    alpha_deg: float
    #: The angle the derivatives were actually measured at. If it is far from `alpha_deg`, the
    #: model is being built from derivatives that describe a different flight condition.
    alpha_measured_deg: float
    mass_kg: float

    @property
    def mismatch_deg(self) -> float:
        return abs(self.alpha_deg - self.alpha_measured_deg)

    def complaint(self, tolerance_deg: float = 2.0) -> str:
        """Why this trim point should not be used with these derivatives, or ""."""
        if self.mismatch_deg > tolerance_deg:
            return (
                f"level flight at {self.velocity_mps:g} m/s needs {self.alpha_deg:+.1f} deg, but "
                f"the derivatives were measured at {self.alpha_measured_deg:+.1f} deg - re-run the "
                f"stability solve at the trim incidence or the modes describe a different aircraft"
            )
        return ""


def trim_level_flight(
    derivatives: StabilityResult, mass: MassProperties, condition: OperatingCondition
) -> TrimPoint:
    """The lift coefficient and incidence that hold level flight at this speed and weight.

    Lift equals weight, so CL is fixed by the wing loading and the dynamic pressure — there is
    nothing to iterate. The incidence follows from the lift-curve slope and the zero-alpha lift,
    which is recovered from the measured point rather than assumed.
    """
    air = condition.atmosphere
    q = 0.5 * air.density * condition.velocity_mps**2
    weight = mass.mass_kg * GRAVITY
    cl_required = weight / (q * derivatives.sref_m2) if q * derivatives.sref_m2 > 0 else 0.0

    alpha_measured = np.radians(derivatives.alpha_deg)
    cl_at_zero = derivatives.CL - derivatives.CL_alpha * alpha_measured
    alpha_trim = (
        (cl_required - cl_at_zero) / derivatives.CL_alpha if derivatives.CL_alpha else 0.0
    )
    return TrimPoint(
        velocity_mps=condition.velocity_mps,
        dynamic_pressure=q,
        cl_required=cl_required,
        alpha_deg=float(np.degrees(alpha_trim)),
        alpha_measured_deg=derivatives.alpha_deg,
        mass_kg=mass.mass_kg,
    )


def longitudinal_matrix(
    derivatives: StabilityResult, mass: MassProperties, trim: TrimPoint
) -> np.ndarray:
    """A for x = [u, w, q, theta], level flight.

        u_dot     = X_u·u + X_w·w                       - g·theta
        w_dot     = Z_u·u + Z_w·w + (u0 + Z_q)·q
        q_dot     = M_u·u + M_w·w + M_q·q
        theta_dot =                      q

    The `-g·theta` in the first row is the whole phugoid: a pitch attitude change trades speed for
    height with almost no damping, which is why that mode is slow and nearly undamped in every
    aeroplane ever built.
    """
    u0 = trim.velocity_mps
    q_bar, s, c = trim.dynamic_pressure, derivatives.sref_m2, derivatives.cref_m
    m, iyy = mass.mass_kg, mass.iyy
    cl0, cd0 = trim.cl_required, derivatives.CD

    # Force derivatives, per second. Speed derivatives (C_D_u, C_L_u) are taken as zero: neither
    # solver reports them and both are genuinely near zero below Mach 0.3.
    x_u = -(2.0 * cd0) * q_bar * s / (m * u0)
    x_w = -(derivatives.CD_alpha - cl0) * q_bar * s / (m * u0)
    z_u = -(2.0 * cl0) * q_bar * s / (m * u0)
    z_w = -(derivatives.CL_alpha + cd0) * q_bar * s / (m * u0)
    z_q = -derivatives.CL_q * (c / (2.0 * u0)) * q_bar * s / m

    # Moment derivatives, per second. C_m_u is zero for the same reason as above; C_m_alphadot is
    # zero because nothing measures it, and that one does cost real short-period damping.
    m_w = derivatives.Cm_alpha * q_bar * s * c / (u0 * iyy)
    m_q = derivatives.Cm_q * (c / (2.0 * u0)) * q_bar * s * c / iyy

    return np.array([
        [x_u, x_w, 0.0, -GRAVITY],
        [z_u, z_w, u0 + z_q, 0.0],
        [0.0, m_w, m_q, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])


def lateral_matrix(
    derivatives: StabilityResult, mass: MassProperties, trim: TrimPoint
) -> np.ndarray:
    """A for x = [v, p, r, phi], level flight.

    Roll and yaw are coupled through the product of inertia, so the rolling and yawing moment
    derivatives are replaced by their primed forms — the standard treatment. With `ixz` at zero the
    primes collapse back to the plain derivatives, which is why leaving it out looks harmless right
    up until the configuration is one where it is not.
    """
    u0 = trim.velocity_mps
    q_bar, s, b = trim.dynamic_pressure, derivatives.sref_m2, derivatives.bref_m
    m, ixx, izz, ixz = mass.mass_kg, mass.ixx, mass.izz, mass.ixz

    y_v = derivatives.CY_beta * q_bar * s / (m * u0)
    y_p = derivatives.CY_p * (b / (2.0 * u0)) * q_bar * s / m
    y_r = derivatives.CY_r * (b / (2.0 * u0)) * q_bar * s / m

    l_v = derivatives.Cl_beta * q_bar * s * b / (u0 * ixx)
    l_p = derivatives.Cl_p * (b / (2.0 * u0)) * q_bar * s * b / ixx
    l_r = derivatives.Cl_r * (b / (2.0 * u0)) * q_bar * s * b / ixx

    n_v = derivatives.Cn_beta * q_bar * s * b / (u0 * izz)
    n_p = derivatives.Cn_p * (b / (2.0 * u0)) * q_bar * s * b / izz
    n_r = derivatives.Cn_r * (b / (2.0 * u0)) * q_bar * s * b / izz

    # Inertia coupling. `denominator` is 1 when ixz is zero, so this is a no-op on a symmetric,
    # uncoupled configuration rather than a special case to remember.
    ratio = (ixz * ixz) / (ixx * izz)
    denominator = 1.0 - ratio
    a, e = ixz / ixx, ixz / izz

    def primed(rolling: float, yawing: float) -> tuple[float, float]:
        return (
            (rolling + a * yawing) / denominator,
            (yawing + e * rolling) / denominator,
        )

    l_v_p, n_v_p = primed(l_v, n_v)
    l_p_p, n_p_p = primed(l_p, n_p)
    l_r_p, n_r_p = primed(l_r, n_r)

    return np.array([
        [y_v, y_p, y_r - u0, GRAVITY],
        [l_v_p, l_p_p, l_r_p, 0.0],
        [n_v_p, n_p_p, n_r_p, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])


# ════════ closed-form approximations, for checking the matrices ═════════════════════════════
#
# These are the textbook two-degree-of-freedom reductions. They are not used to compute anything —
# the eigenvalues of the matrices above are the answer — but they are what the eigenvalues get
# checked against, because they are derivable independently and a matrix with a sign error in it
# will not match them.


def short_period_approximation(
    derivatives: StabilityResult, mass: MassProperties, trim: TrimPoint
) -> tuple[float, float]:
    """(natural frequency rad/s, damping ratio) from the classic short-period reduction.

    Holds speed constant and keeps only w and q, which is a good approximation because the mode is
    fast enough that speed has no time to change.
    """
    u0 = trim.velocity_mps
    q_bar, s, c = trim.dynamic_pressure, derivatives.sref_m2, derivatives.cref_m
    m, iyy = mass.mass_kg, mass.iyy

    z_w = -(derivatives.CL_alpha + derivatives.CD) * q_bar * s / (m * u0)
    m_w = derivatives.Cm_alpha * q_bar * s * c / (u0 * iyy)
    m_q = derivatives.Cm_q * (c / (2.0 * u0)) * q_bar * s * c / iyy

    omega_squared = z_w * m_q - u0 * m_w
    if omega_squared <= 0:
        return 0.0, 0.0
    omega = float(np.sqrt(omega_squared))
    zeta = float(-(z_w + m_q) / (2.0 * omega))
    return omega, zeta


def phugoid_approximation(trim: TrimPoint, cd: float) -> tuple[float, float]:
    """(natural frequency rad/s, damping ratio) from Lanchester's reduction.

    omega = g·sqrt(2)/u0 and zeta = CD/(CL·sqrt(2)). Remarkable for depending on almost nothing:
    the frequency is set by speed and gravity alone, and the damping only by lift-to-drag. It is
    the strongest independent check available on the longitudinal matrix, because it shares no
    terms with the short-period reduction.
    """
    u0 = trim.velocity_mps
    if u0 <= 0 or trim.cl_required <= 0:
        return 0.0, 0.0
    omega = GRAVITY * np.sqrt(2.0) / u0
    zeta = cd / (trim.cl_required * np.sqrt(2.0))
    return float(omega), float(zeta)
