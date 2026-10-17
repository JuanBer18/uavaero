"""The stability run against the real solver, and against a second code.

Phase 3's criterion is that *measured* derivatives — not hardcoded defaults — populate the
stability model, validated against a known case or an independent solver. Both halves are here:
the three-way static margin check validates the perturbation arithmetic internally, and AVL
validates the physics externally.

The measured disagreement is part of the record. VSPAERO and AVL agree on CL_alpha to 7% and on
Cn_beta to 23%, and differ on Cm_alpha by a factor of two. That is reported, not resolved — two
independent codes disagreeing is a real fact about how well this aircraft is known at this stage,
and picking a winner without a third source would be inventing a confidence nobody has.
"""
from __future__ import annotations

import pytest

from tests.markers import requires_openvsp
from uavaero.schema.airframe import default_airframe
from uavaero.schema.condition import Fidelity, OperatingCondition
from uavaero.solvers import avl, stability

pytestmark = [requires_openvsp, pytest.mark.slow]


@pytest.fixture(scope="module")
def solved():
    condition = OperatingCondition(
        fidelity=Fidelity.QUICK, scope="wing_tail", alpha_start_deg=4.0,
        velocity_mps=20.0, xyz_ref=(0.085, 0.0, 0.0),
    )
    return condition, stability.solve(default_airframe(), condition)


def test_the_stability_run_produces_a_derivative_set(solved):
    _, result = solved
    assert result.CL_alpha > 0
    assert result.neutral_point_x_m is not None
    assert result.static_margin is not None


def test_the_three_routes_to_static_margin_agree(solved):
    """The internal validation. All three come from one solve, so agreement means the perturbation
    arithmetic is right and every derivative is being read correctly."""
    _, result = solved
    assert result.consistency() == "", result.consistency()
    assert result.static_margin_from_derivatives == pytest.approx(
        result.static_margin_reported, abs=0.01
    )


def test_the_reference_aircraft_is_stable_on_all_three_axes(solved):
    """Cm_alpha negative, Cn_beta positive, Cl_beta negative — three conventions, one aircraft."""
    _, result = solved
    assert result.verdicts() == {
        "longitudinal": True, "directional": True, "lateral": True,
    }
    assert 0.02 < result.static_margin < 0.40, "a plausible static margin for a tailed aircraft"


def test_the_lift_curve_slope_is_near_lifting_line(solved):
    """A known-answer check that does not depend on the other solver."""
    _, result = solved
    assert 4.0 < result.CL_alpha < 7.5


@pytest.mark.skipif(not avl.available(), reason="avl binary not found")
def test_the_rate_derivatives_come_from_avl_and_damp(solved):
    """VSPAERO's own rates fail the damping sign check — Cn_r comes back positive — and do not
    correspond to AVL's by any single scale factor (the ratios are 28x, 58x, 25x, 17x, 157x and
    -49x), so they are not a non-dimensionalisation that could be corrected for. The finished
    result carries AVL's instead, and this is the check that they arrived."""
    _, result = solved
    assert result.rate_source == "avl"
    assert result.rates_validated
    assert result.Cm_q < 0, "pitch damping"
    assert result.Cl_p < 0, "roll damping"
    assert result.Cn_r < 0, "yaw damping"


@pytest.mark.skipif(not avl.available(), reason="avl binary not found")
def test_the_two_codes_agree_on_the_lift_curve_slope(solved):
    """The external validation. Two entirely separate codes on the same geometry, with the moment
    reference matched — they should be close on CL_alpha, and they are."""
    _, result = solved
    row = result.cross_check.get("CL_alpha")
    assert row is not None
    assert abs(row["relative_error"]) < 0.15


@pytest.mark.skipif(not avl.available(), reason="avl binary not found")
def test_a_disagreement_is_recorded_rather_than_smoothed_over(solved):
    """The cross-check exists to surface this, not to hide it. Whether or not Cm_alpha happens to
    disagree on a given run, any disagreement must appear in the notes rather than be averaged
    away."""
    _, result = solved
    for name in result.disagreements():
        assert name in result.cross_check
        row = result.cross_check[name]
        assert row["vspaero"] != row["avl"]
    if result.disagreements():
        assert any("disagree" in note for note in result.notes)
