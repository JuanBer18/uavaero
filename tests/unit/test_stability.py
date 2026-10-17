"""Stability derivatives: the perturbation arithmetic, and the checks that police it.

VSPAERO's stability run reports *perturbed values*, not derivatives — a base solve plus one solve
per nudged variable, each carrying its own delta. Reading those fields as derivatives directly
gives numbers about a hundred times too small, and, which is the trap, they still describe a
perfectly plausible aircraft. So the arithmetic is tested against values measured from a real run.

Two guards matter more than the rest.

**The three-way static margin check.** SM is reachable three ways from one solve: VSPAERO reports
it, VSPAERO reports the neutral point, and the derivatives give −Cm_α/CL_α. The third exercises the
perturbation arithmetic, so when it agrees the whole derivative set is being read correctly — and
when it does not, every derivative is suspect rather than just that one.

**The damping sign check.** Cm_q, Cl_p and Cn_r are all damping terms; a positive value is an
aircraft that drives its own rotation. VSPAERO's raw rate output fails this (Cn_r comes back
positive), which is why the rates in a finished result come from AVL instead.
"""
from __future__ import annotations

import math

import pytest

from uavaero.schema.condition import OperatingCondition
from uavaero.schema.stability import StabilityResult, per_radian
from uavaero.solvers.stability import ALPHA, BASE, BETA, PITCH_RATE, derivative, parse

#: Field values measured from a real VSPAERO stability run on the reference aircraft at 4 degrees,
#: wing+tail scope, CG 0.085 m. Kept verbatim so the arithmetic is pinned against the solver's own
#: output rather than against numbers invented to make it pass.
MEASURED = {
    "Base_Aero_CL": [0.54085], "Base_Aero_CD": [0.01981], "Base_Aero_CMm": [-0.08419],
    "Alpha_CL": [0.54185], "Alpha_CD": [0.01986], "Alpha_CMm": [-0.08437], "Alpha_Delta": [0.01],
    "Beta_CS": [0.0007], "Beta_CMl": [0.00197], "Beta_CMn": [0.00018], "Beta_Delta": [0.01],
    "Base_Aero_CS": [0.00072], "Base_Aero_CMl": [0.00198], "Base_Aero_CMn": [0.00017],
    "Pitch_Rate_CMm": [-0.08682], "Pitch_Rate_CL": [0.54095], "Pitch_Rate_Delta": [0.01],
    "X_np": [0.13074], "SM": [0.18057],
}


def _condition() -> OperatingCondition:
    return OperatingCondition(
        scope="wing_tail", alpha_start_deg=4.0, velocity_mps=20.0, xyz_ref=(0.085, 0.0, 0.0),
    )


# ════════ the perturbation arithmetic ═══════════════════════════════════════════════════════


def test_a_derivative_is_the_difference_over_the_delta():
    """Not the perturbed value itself. Reading `Alpha_CL` as CL_alpha gives 0.54 per radian — a
    wing with almost no lift slope, which is wrong by two orders of magnitude and still looks like
    a number rather than an error."""
    assert derivative(MEASURED, ALPHA, "CL") == pytest.approx((0.54185 - 0.54085) / 0.01)


def test_a_missing_field_is_zero_rather_than_an_exception():
    """A configuration with no surface to produce a given derivative is ordinary, and zero is the
    honest value for it."""
    assert derivative(MEASURED, "Yaw___Rate", "CMn") == 0.0
    assert derivative({}, ALPHA, "CL") == 0.0


def test_a_zero_delta_does_not_divide_by_zero():
    assert derivative({**MEASURED, "Alpha_Delta": [0.0]}, ALPHA, "CL") == 0.0


def test_angles_are_converted_to_per_radian():
    """The state-space model in Phase 4 wants per radian, and converting once here beats converting
    at each of the dozen places that will read these."""
    assert per_radian(0.1) == pytest.approx(5.7296, rel=1e-4)
    result = parse(MEASURED, _condition(), (0.6, 2.4, 0.2533))
    assert result.CL_alpha == pytest.approx(per_radian(0.1), rel=1e-3)
    assert 4.0 < result.CL_alpha < 8.0, "a lift-curve slope should be near 2*pi"


def test_the_parsed_result_reproduces_the_solvers_own_static_margin():
    """The end-to-end check on the arithmetic, against the number VSPAERO computed itself."""
    result = parse(MEASURED, _condition(), (0.6, 2.4, 0.2533))
    assert result.static_margin_from_derivatives == pytest.approx(0.1806, abs=0.005)
    assert result.static_margin_reported == pytest.approx(0.18057)
    assert result.consistency() == ""


def test_the_padded_field_names_are_not_typos():
    """`Roll__Rate_` and `Yaw___Rate_` carry padding underscores so every prefix is the same width.
    Stripping them to one underscore silently returns zero for every rate derivative."""
    from uavaero.solvers.stability import ROLL_RATE, YAW_RATE

    assert ROLL_RATE == "Roll__Rate"
    assert YAW_RATE == "Yaw___Rate"
    assert len(ROLL_RATE) == len(YAW_RATE) == len(PITCH_RATE) == len("Pitch_Rate")


# ════════ the three-way check ═══════════════════════════════════════════════════════════════


def _result(**kwargs) -> StabilityResult:
    base = dict(
        CL_alpha=5.4964, Cm_alpha=-1.0021, cref_m=0.2533, x_cg_m=0.085,
        neutral_point_x_m=0.1312, static_margin_reported=0.1822,
    )
    base.update(kwargs)
    return StabilityResult(**base)


def test_the_three_routes_to_static_margin_agree_on_a_real_result():
    result = _result()
    assert result.static_margin_from_derivatives == pytest.approx(0.1823, abs=0.001)
    assert result.static_margin_from_neutral_point == pytest.approx(0.1822, abs=0.001)
    assert result.consistency() == ""


def test_a_disagreement_condemns_every_derivative_not_just_the_margin():
    """All three routes come from one solve. If they part company the perturbation values are being
    read wrong, and the message has to say that rather than let someone conclude the aircraft is
    merely oddly balanced."""
    complaint = _result(Cm_alpha=-3.0).consistency()
    assert complaint
    assert "every derivative here is suspect" in complaint


def test_one_route_alone_cannot_be_checked():
    """Not an error — a result with only the derivatives is still usable, it just has nothing to be
    checked against, and claiming a passed check would be worse than admitting there was none."""
    lonely = StabilityResult(CL_alpha=5.5, Cm_alpha=-1.0)
    assert lonely.consistency() == ""
    assert lonely.static_margin_from_neutral_point is None


# ════════ the damping signs ═════════════════════════════════════════════════════════════════


def test_vspaeros_raw_rates_are_rejected():
    """The measured values from a real run. Cn_r positive is an aircraft that drives its own yaw."""
    raw = _result(Cm_q=-0.2695, Cl_p=-0.0231, Cn_r=+0.0009)
    complaint = raw.rate_complaint()
    assert "Cn_r" in complaint
    assert "damping" in complaint


def test_avls_rates_pass():
    """AVL's values from the same aircraft: every damping term negative and of a sensible size."""
    good = _result(Cm_q=-15.5258, Cl_p=-0.5676, Cn_r=-0.0430)
    assert good.rate_complaint() == ""


@pytest.mark.parametrize("field", ["Cm_q", "Cl_p", "Cn_r"])
def test_each_damping_term_is_checked_individually(field):
    assert field in _result(**{field: 0.5}).rate_complaint()


# ════════ stability verdicts and their three different sign conventions ═════════════════════


def test_each_axis_uses_its_own_standard_sign_convention():
    """Cm_alpha negative, Cn_beta positive, Cl_beta negative are all "stable". Three conventions in
    one object, and each is the standard for its own axis — which is exactly why this is asserted
    rather than left to whoever reads the field next."""
    stable = _result(Cm_alpha=-1.0, Cn_beta=+0.04, Cl_beta=-0.08)
    assert stable.verdicts() == {"longitudinal": True, "directional": True, "lateral": True}

    unstable = _result(Cm_alpha=+1.0, Cn_beta=-0.04, Cl_beta=+0.08)
    assert unstable.verdicts() == {"longitudinal": False, "directional": False, "lateral": False}


def test_disagreements_are_listed_rather_than_resolved():
    """Two codes differing on Cm_alpha by a factor of two is a real fact about how well this
    aircraft is known. Picking a winner without a third source would invent a confidence nobody
    has — the same reasoning that keeps three drag models side by side."""
    result = _result(cross_check={
        "CL_alpha": {"vspaero": 5.50, "avl": 5.90, "agrees": True},
        "Cm_alpha": {"vspaero": -1.00, "avl": -0.53, "agrees": False},
    })
    assert result.disagreements() == ["Cm_alpha"]


def test_a_result_round_trips():
    """It is stored in the airframe file beside the aero solves."""
    original = _result(Cm_q=-15.5, Cl_p=-0.57, Cn_r=-0.043, rate_source="avl",
                       notes=["codes disagree on Cm_alpha"])
    restored = StabilityResult.from_dict(original.to_dict())
    assert restored.Cm_q == pytest.approx(-15.5)
    assert restored.rate_source == "avl"
    assert restored.notes == ["codes disagree on Cm_alpha"]
    assert restored.static_margin == pytest.approx(original.static_margin)


def test_the_serialised_form_is_json_plain():
    import json

    payload = _result().to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert not math.isnan(payload["static_margin"])
