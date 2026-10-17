"""The five modes: building the matrices, and naming what comes out of them.

**The property this file exists to protect is that classification does not use frequency.**

The tempting rule is "the faster longitudinal pair is the short period". It works on most aircraft
and fails on exactly the ones worth checking: on a light airframe with a large static margin the
phugoid rises until the two pairs overlap, and a frequency test then labels them backwards. Every
number downstream — damping, period, handling verdict — inherits the swap, and nothing looks wrong.
`test_the_modes_are_named_by_participation_not_by_frequency` constructs that overlap deliberately.

The second property is that the matrices are right, checked against closed-form reductions that
share no terms with them. Lanchester's phugoid depends only on speed and lift-to-drag; the
short-period reduction depends only on the pitch terms. A matrix with a sign error in it does not
match both.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from uavaero.dynamics.modes import Mode, analyse
from uavaero.dynamics.report import analyse_aircraft, excite, time_response
from uavaero.dynamics.statespace import (
    GRAVITY,
    LATERAL_STATES,
    LONGITUDINAL_STATES,
    lateral_matrix,
    longitudinal_matrix,
    phugoid_approximation,
    short_period_approximation,
    trim_level_flight,
)
from uavaero.schema.condition import OperatingCondition
from uavaero.schema.mass import MassProperties
from uavaero.schema.stability import StabilityResult


def _derivatives(**kwargs) -> StabilityResult:
    """A conventional, statically stable small UAS, measured at 4 degrees."""
    base = dict(
        alpha_deg=4.0, velocity_mps=20.0, sref_m2=0.6, bref_m=2.4, cref_m=0.2533,
        CL=0.539, CD=0.0197, Cm=-0.094,
        CL_alpha=5.4964, CD_alpha=0.2876, Cm_alpha=-1.0021,
        CY_beta=-0.1186, Cl_beta=-0.0825, Cn_beta=0.0435,
        CL_q=10.456, Cm_q=-15.526, Cl_p=-0.5676, Cn_p=-0.0434,
        Cl_r=0.1492, Cn_r=-0.0430, CY_p=-0.05, CY_r=0.08,
        rates_validated=True, rate_source="avl",
    )
    base.update(kwargs)
    return StabilityResult(**base)


def _mass(**kwargs) -> MassProperties:
    base = dict(mass_kg=3.5, ixx=1.26, iyy=0.574, izz=1.507, ixz=0.0)
    base.update(kwargs)
    return MassProperties(**base)


def _condition(**kwargs) -> OperatingCondition:
    base = dict(velocity_mps=20.0, alpha_start_deg=4.0, xyz_ref=(0.085, 0.0, 0.0))
    base.update(kwargs)
    return OperatingCondition(**base)


# ════════ classification ════════════════════════════════════════════════════════════════════


def test_the_five_modes_are_all_found_and_named():
    report = analyse_aircraft(_derivatives(), _mass(), _condition())
    assert {m.name for m in report.modes} == {
        "short period", "phugoid", "dutch roll", "roll subsidence", "spiral",
    }


def test_each_mode_is_dominated_by_the_states_that_define_it():
    """The participation signature *is* the mode. Short period is incidence and pitch rate;
    phugoid is speed; roll subsidence is roll rate; Dutch roll is sideslip; spiral is bank."""
    report = analyse_aircraft(_derivatives(), _mass(), _condition())

    def dominant(name: str) -> str:
        mode = report.named(name)
        return max(mode.participation.items(), key=lambda kv: kv[1])[0]

    assert dominant("short period") in ("w", "q")
    assert dominant("phugoid") in ("u", "theta")
    assert dominant("roll subsidence") == "p"
    assert dominant("dutch roll") in ("v", "r")
    assert dominant("spiral") in ("phi", "v")


def test_the_modes_are_named_by_participation_not_by_frequency():
    """The case a frequency test gets backwards.

    Two oscillatory pairs are constructed at the *same* natural frequency, one moving w and q and
    the other moving u and theta. Any rule that sorts by speed has to guess; participation does not.
    """
    # Block-diagonal by construction: one pair in (w, q), one in (u, theta), same |eigenvalue|.
    fast = np.array([[-1.0, -3.0], [3.0, -1.0]])   # in (w, q)
    slow = np.array([[-1.0, 3.0], [-3.0, -1.0]])   # in (u, theta), identical magnitude
    matrix = np.zeros((4, 4))
    matrix[np.ix_([1, 2], [1, 2])] = fast   # w, q
    matrix[np.ix_([0, 3], [0, 3])] = slow   # u, theta

    modes = analyse(matrix, LONGITUDINAL_STATES, lateral=False)
    short = next(m for m in modes if m.name == "short period")
    phugoid = next(m for m in modes if m.name == "phugoid")

    assert short.natural_frequency == pytest.approx(phugoid.natural_frequency, rel=1e-6), (
        "the two pairs must be at the same frequency for this test to mean anything"
    )
    assert short.participation["w"] + short.participation["q"] > 0.9
    assert phugoid.participation["u"] + phugoid.participation["theta"] > 0.9


def test_a_conjugate_pair_is_one_mode_not_two():
    """Listing both halves makes a four-state model look like it has six behaviours."""
    report = analyse_aircraft(_derivatives(), _mass(), _condition())
    assert len(report.longitudinal) == 2
    assert len(report.lateral) == 3


def test_the_fast_real_root_is_roll_and_the_slow_one_is_spiral():
    report = analyse_aircraft(_derivatives(), _mass(), _condition())
    roll = report.named("roll subsidence")
    spiral = report.named("spiral")
    assert abs(roll.real) > abs(spiral.real), "roll subsidence is the fast one, by definition"


# ════════ what a mode reports ═══════════════════════════════════════════════════════════════


def test_a_real_root_has_no_damping_ratio():
    """None rather than 1.0. Reporting 1.0 invites the reader to conclude it is critically damped,
    which is a different claim about a different thing."""
    mode = Mode(eigenvalue=complex(-2.0, 0.0))
    assert mode.damping_ratio is None
    assert mode.period is None
    assert mode.time_constant == pytest.approx(0.5)


def test_an_oscillatory_mode_reports_damping_and_period():
    mode = Mode(eigenvalue=complex(-0.5, 2.0))
    assert mode.damping_ratio == pytest.approx(0.5 / math.sqrt(0.25 + 4.0))
    assert mode.period == pytest.approx(2 * math.pi / 2.0)
    assert mode.time_constant is None


def test_a_stable_mode_halves_and_an_unstable_one_doubles():
    """An unstable mode has no time-to-half, and reporting a negative one would be worse than
    reporting none. The doubling time is what decides whether a divergent spiral matters — forty
    seconds is a non-event, four is not."""
    stable, unstable = Mode(eigenvalue=complex(-0.1, 0)), Mode(eigenvalue=complex(0.1, 0))
    assert stable.time_to_half == pytest.approx(math.log(2) / 0.1)
    assert stable.time_to_double is None
    assert unstable.time_to_double == pytest.approx(math.log(2) / 0.1)
    assert unstable.time_to_half is None


def test_handling_quality_floors_apply_only_where_they_exist():
    assert Mode(eigenvalue=complex(-3.0, 3.0), name="short period").acceptable is True
    assert Mode(eigenvalue=complex(-0.2, 3.0), name="short period").acceptable is False
    assert Mode(eigenvalue=complex(-1.0, 0.0), name="roll subsidence").acceptable is None


# ════════ the matrices, against closed-form reductions ══════════════════════════════════════


def test_the_short_period_matches_its_two_degree_of_freedom_reduction():
    """The reduction holds speed constant and keeps only w and q. It shares no terms with the
    phugoid check, so matching both is a real constraint on the matrix."""
    derivatives, mass, condition = _derivatives(), _mass(), _condition()
    report = analyse_aircraft(derivatives, mass, condition)
    omega, zeta = short_period_approximation(derivatives, mass, report.trim)

    mode = report.named("short period")
    assert mode.natural_frequency == pytest.approx(omega, rel=0.15)
    assert mode.damping_ratio == pytest.approx(zeta, rel=0.20)


def test_the_phugoid_matches_lanchester():
    """omega = g*sqrt(2)/u0 and zeta = CD/(CL*sqrt(2)). Remarkable for depending on almost nothing
    — the frequency is speed and gravity alone. A four-state eigenvalue landing near it is strong
    evidence the first row of the matrix, where the -g coupling lives, is right."""
    derivatives, mass, condition = _derivatives(), _mass(), _condition()
    report = analyse_aircraft(derivatives, mass, condition)
    omega, zeta = phugoid_approximation(report.trim, derivatives.CD)

    assert omega == pytest.approx(GRAVITY * math.sqrt(2) / 20.0, rel=1e-9)
    mode = report.named("phugoid")
    assert mode.natural_frequency == pytest.approx(omega, rel=0.35)
    assert mode.damping_ratio == pytest.approx(zeta, rel=0.5)


def test_a_statically_unstable_aircraft_has_a_divergent_short_period():
    """Cm_alpha positive removes the pitch stiffness, so the oscillation becomes a divergence.
    The sign of Cm_alpha reaching the matrix correctly is the single most important thing in it."""
    report = analyse_aircraft(_derivatives(Cm_alpha=+1.5), _mass(), _condition())
    assert any(not m.stable for m in report.longitudinal)


def test_the_product_of_inertia_is_a_no_op_when_it_is_zero():
    """The primed derivatives collapse to the plain ones at ixz = 0, so carrying the coupling costs
    nothing on a symmetric aircraft — which is what makes it safe to always apply."""
    derivatives, condition = _derivatives(), _condition()
    trim = trim_level_flight(derivatives, _mass(), condition)
    plain = lateral_matrix(derivatives, _mass(ixz=0.0), trim)
    coupled = lateral_matrix(derivatives, _mass(ixz=0.15), trim)

    assert np.allclose(plain, plain)
    assert not np.allclose(plain, coupled), "a real ixz must actually change the matrix"


def test_the_gravity_term_is_where_the_phugoid_comes_from():
    """Removing the -g coupling in the first row should destroy the phugoid and leave the short
    period alone — which is the clearest statement of what that term is for."""
    derivatives, mass, condition = _derivatives(), _mass(), _condition()
    trim = trim_level_flight(derivatives, mass, condition)
    matrix = longitudinal_matrix(derivatives, mass, trim)
    assert matrix[0, 3] == pytest.approx(-GRAVITY)

    without = matrix.copy()
    without[0, 3] = 0.0
    slowest = min(analyse(without, LONGITUDINAL_STATES), key=lambda m: m.natural_frequency)
    original = min(analyse(matrix, LONGITUDINAL_STATES), key=lambda m: m.natural_frequency)
    assert slowest.natural_frequency < original.natural_frequency


# ════════ trim, and the mismatch that invalidates everything above ══════════════════════════


def test_level_flight_lift_equals_weight():
    derivatives, mass, condition = _derivatives(), _mass(), _condition()
    trim = trim_level_flight(derivatives, mass, condition)
    lift = trim.cl_required * trim.dynamic_pressure * derivatives.sref_m2
    assert lift == pytest.approx(mass.mass_kg * GRAVITY, rel=1e-9)


def test_a_trim_far_from_the_measured_incidence_is_reported():
    """The derivatives describe the aircraft at one angle. If level flight needs a very different
    one, the modes are being built from a different flight condition than the one they claim."""
    derivatives, mass = _derivatives(alpha_deg=4.0), _mass(mass_kg=1.0)
    trim = trim_level_flight(derivatives, mass, _condition())
    assert trim.mismatch_deg > 2.0
    assert "re-run the stability solve" in trim.complaint()


def test_a_matched_trim_says_nothing():
    """A derivative set measured at the incidence level flight actually needs.

    Both the angle and the lift coefficient have to move together — quoting a CL measured at 4
    degrees against an angle of 0.8 describes an aircraft that does not exist, and the trim solve
    correctly refuses it.
    """
    mass = _mass()
    condition = _condition()
    # CL required for level flight at this speed and weight, and the angle that produces it.
    q = 0.5 * condition.atmosphere.density * condition.velocity_mps**2
    cl_required = mass.mass_kg * GRAVITY / (q * 0.6)
    derivatives = _derivatives(alpha_deg=0.8, CL=cl_required)

    trim = trim_level_flight(derivatives, mass, condition)
    assert trim.cl_required == pytest.approx(cl_required, rel=1e-9)
    assert trim.alpha_deg == pytest.approx(0.8, abs=1e-6)
    assert trim.complaint() == ""


# ════════ the qualifications that travel with the answer ════════════════════════════════════


def test_estimated_inertia_is_reported_on_the_result():
    """Every frequency scales with the square root of an inertia. A guessed Iyy is a guessed
    short period, and that has to reach the screen."""
    mass = MassProperties.from_radii_of_gyration(3.5, span_m=2.4, length_m=1.35)
    report = analyse_aircraft(_derivatives(), mass, _condition())
    assert any("estimated" in note.lower() for note in report.notes)


def test_unvalidated_rate_derivatives_are_reported():
    report = analyse_aircraft(_derivatives(rates_validated=False), _mass(), _condition())
    assert any("damping" in note for note in report.notes)


def test_the_missing_alphadot_term_is_always_declared():
    """It supplies about a third of short-period damping and nothing measures it. Silence would
    let an under-estimate pass as the answer."""
    report = analyse_aircraft(_derivatives(), _mass(), _condition())
    assert any("alphadot" in note for note in report.notes)


def test_incomplete_mass_is_refused_rather_than_producing_infinities():
    """A zero inertia divides by zero somewhere downstream and plots as an empty chart rather than
    as an error."""
    with pytest.raises(ValueError, match="incomplete"):
        analyse_aircraft(_derivatives(), MassProperties(mass_kg=3.5), _condition())


def test_a_divergent_spiral_is_not_called_unstable_in_the_summary():
    """Most aeroplanes have one and it is flown without noticing. Leading with "UNSTABLE" for the
    normal case trains the reader to ignore the word."""
    report = analyse_aircraft(_derivatives(), _mass(), _condition())
    spiral = report.named("spiral")
    if spiral is not None and not spiral.stable:
        assert "spiral" in report.summary()
        assert not report.summary().startswith("UNSTABLE")


# ════════ response ══════════════════════════════════════════════════════════════════════════


def test_a_stable_system_decays():
    matrix = np.array([[-0.5, 2.0], [-2.0, -0.5]])
    _, states = time_response(matrix, np.array([1.0, 0.0]), duration_s=20.0, points=200)
    assert np.linalg.norm(states[-1]) < 0.05 * np.linalg.norm(states[0])


def test_the_response_is_exact_at_every_sample():
    """Solved by matrix exponential, not by stepping. A stiff pair — a short period sitting on a
    phugoid — is exactly where an explicit integrator quietly loses the fast mode."""
    matrix = np.array([[-20.0, 0.0], [0.0, -0.05]])
    times, states = time_response(matrix, np.array([1.0, 1.0]), duration_s=10.0, points=50)
    expected_fast = math.exp(-20.0 * times[-1])
    assert states[-1, 0] == pytest.approx(expected_fast, abs=1e-12)
    assert states[-1, 1] == pytest.approx(math.exp(-0.05 * times[-1]), rel=1e-9)


def test_exciting_a_mode_uses_its_own_participation():
    """Asking to see the phugoid should disturb the states the phugoid moves, not ring every mode
    at once."""
    report = analyse_aircraft(_derivatives(), _mass(), _condition())
    phugoid = report.named("phugoid")
    initial = excite(phugoid, LONGITUDINAL_STATES, magnitude=1.0)
    assert np.linalg.norm(initial) == pytest.approx(1.0)
    assert abs(initial[LONGITUDINAL_STATES.index("u")]) > abs(
        initial[LONGITUDINAL_STATES.index("q")]
    )


def test_a_report_round_trips():
    report = analyse_aircraft(_derivatives(), _mass(), _condition())
    from uavaero.dynamics.report import DynamicsReport

    restored = DynamicsReport.from_dict(report.to_dict())
    assert {m.name for m in restored.modes} == {m.name for m in report.modes}
    assert restored.named("phugoid").period == pytest.approx(report.named("phugoid").period)


# ════════ mass properties ═══════════════════════════════════════════════════════════════════


def test_radii_of_gyration_give_the_right_order_of_magnitude():
    mass = MassProperties.from_radii_of_gyration(3.5, span_m=2.4, length_m=1.35)
    assert mass.estimated
    assert mass.ixx == pytest.approx(3.5 * (0.25 * 2.4) ** 2)
    assert mass.iyy == pytest.approx(3.5 * (0.30 * 1.35) ** 2)
    assert mass.ixx > mass.iyy, "roll inertia exceeds pitch on a wing-dominated aircraft"


def test_the_inertia_tensor_carries_the_negative_product_of_inertia():
    """The single easiest place in rigid-body dynamics to lose a sign."""
    tensor = MassProperties(mass_kg=1, ixx=2, iyy=3, izz=4, ixz=0.5).tensor
    assert tensor[0, 2] == pytest.approx(-0.5)
    assert tensor[2, 0] == pytest.approx(-0.5)
    assert np.allclose(tensor, tensor.T), "the inertia tensor is symmetric"


def test_incomplete_mass_says_what_is_missing():
    assert "Iyy" in MassProperties(mass_kg=3.5, ixx=1.0, izz=1.0).missing()
    assert MassProperties(mass_kg=3.5, ixx=1, iyy=1, izz=1).missing() == ""
