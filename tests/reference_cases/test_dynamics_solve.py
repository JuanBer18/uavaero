"""The whole chain: geometry through VSPAERO and AVL into five named modes.

Phase 4's criterion is that short period, phugoid, Dutch roll, roll subsidence and spiral are named
and reported for a real airframe, and that the result is validated. The validation here is against
closed-form reductions rather than against a published aircraft's tabulated modes, and that is a
deliberate limitation worth stating: reproducing a specific published aeroplane would mean quoting
its derivatives and inertias from memory, and a validation built on half-remembered reference data
is worse than none. The Lanchester phugoid and the two-degree-of-freedom short period are
independently derivable, share no terms with each other, and are checked here against the full
four-state eigenvalues.

A published-aircraft comparison remains outstanding and would be a stronger check.
"""
from __future__ import annotations

import pytest

from tests.markers import requires_openvsp
from uavaero.dynamics.report import analyse_aircraft
from uavaero.dynamics.statespace import phugoid_approximation, short_period_approximation
from uavaero.schema.airframe import default_airframe
from uavaero.schema.condition import Fidelity, OperatingCondition
from uavaero.schema.mass import MassProperties
from uavaero.solvers import stability

pytestmark = [requires_openvsp, pytest.mark.slow]


@pytest.fixture(scope="module")
def analysed():
    condition = OperatingCondition(
        fidelity=Fidelity.QUICK, scope="wing_tail", alpha_start_deg=4.0,
        velocity_mps=20.0, xyz_ref=(0.085, 0.0, 0.0),
    )
    derivatives = stability.solve(default_airframe(), condition)
    mass = MassProperties.from_radii_of_gyration(mass_kg=3.5, span_m=2.4, length_m=1.35)
    return derivatives, mass, condition, analyse_aircraft(derivatives, mass, condition)


def test_all_five_modes_are_named(analysed):
    """The Phase 4 criterion, on measured aerodynamics rather than a fixture."""
    *_, report = analysed
    assert {m.name for m in report.modes} == {
        "short period", "phugoid", "dutch roll", "roll subsidence", "spiral",
    }


def test_the_modes_are_physically_ordered(analysed):
    """Short period fast and well-damped, phugoid slow and barely damped, roll subsidence the
    fastest real root. Any classification failure shows up here as a mode in the wrong place."""
    *_, report = analysed
    short = report.named("short period")
    phugoid = report.named("phugoid")
    roll = report.named("roll subsidence")
    dutch = report.named("dutch roll")

    assert short.natural_frequency > phugoid.natural_frequency * 5
    assert short.damping_ratio > phugoid.damping_ratio
    assert phugoid.period > dutch.period
    assert roll.time_constant is not None and roll.time_constant < 1.0


def test_the_eigenvalues_match_their_closed_form_reductions(analysed):
    """The validation. Two reductions that share no terms with each other, both derivable by hand,
    both matched by the four-state eigenvalues — a matrix with a sign error does not do that."""
    derivatives, mass, _, report = analysed

    omega_sp, zeta_sp = short_period_approximation(derivatives, mass, report.trim)
    short = report.named("short period")
    assert short.natural_frequency == pytest.approx(omega_sp, rel=0.20)
    assert short.damping_ratio == pytest.approx(zeta_sp, rel=0.25)

    omega_ph, _ = phugoid_approximation(report.trim, derivatives.CD)
    phugoid = report.named("phugoid")
    assert phugoid.natural_frequency == pytest.approx(omega_ph, rel=0.40)


def test_the_reference_aircraft_is_conventionally_behaved(analysed):
    """Stable short period, stable Dutch roll, fast roll subsidence. The spiral is allowed to
    diverge — most aeroplanes do, and it is flown without noticing."""
    *_, report = analysed
    assert report.named("short period").stable
    assert report.named("dutch roll").stable
    assert report.named("roll subsidence").stable
    assert not [m for m in report.unstable if m.name != "spiral"]


def test_the_qualifications_travel_with_the_result(analysed):
    """Estimated inertia and the missing alphadot term both make these numbers less certain than
    they look, and both have to reach the reader."""
    *_, report = analysed
    joined = " ".join(report.notes).lower()
    assert "estimated" in joined
    assert "alphadot" in joined


def test_a_response_can_be_computed_for_every_mode(analysed):
    """Each mode's own participation drives its disturbance, so every one is excitable."""
    from uavaero.dynamics.report import excite, time_response
    from uavaero.dynamics.statespace import LATERAL_STATES, LONGITUDINAL_STATES

    *_, report = analysed
    for mode in report.modes:
        longitudinal = mode in report.longitudinal
        matrix = report.a_longitudinal if longitudinal else report.a_lateral
        states = LONGITUDINAL_STATES if longitudinal else LATERAL_STATES
        times, response = time_response(matrix, excite(mode, states), duration_s=20.0, points=100)
        assert response.shape == (100, 4)
        assert not (response != response).any(), f"{mode.name} produced NaN"
