"""The optimiser, and the ways it would cheat if it could.

**An optimiser is an adversary.** It does not find good aircraft; it finds high scores, and it will
take any route to one. Every guard built in the previous four phases becomes an attack surface the
moment a violation scores well:

  - a degenerate mesh returns zeros, and zero drag is an infinite L/D
  - the thick-panel solve returns negative induced drag, which beats zero
  - a strip with no section data is dropped, so a wing outside the polar range sheds viscous drag
  - a missing parasite model leaves the comparison with only its two optimistic members
  - a stalled section is still reported by a panel method as making lift

Every one of those is already *detected* by earlier phases. The tests below are what make the
detection *decide the score*, and each names the specific exploit it closes. If any of them starts
passing a broken result, an overnight run will find that hole and the answer will be an aircraft
that only exists inside the bug.

The second theme is that **variables must be independent**. Span, area and aspect ratio are three
names for two numbers, and handing all three to a search lets it ask for the impossible — which it
does immediately.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from uavaero.geometry import authoring
from uavaero.optimisation.objectives import (
    REJECTED_SCORE,
    StabilityConstraints,
    max_endurance_factor,
    max_lift_to_drag,
    min_drag,
    trust,
)
from uavaero.optimisation.screening import Limits, RejectionLedger, screen
from uavaero.optimisation.space import DesignSpace, standard_variables
from uavaero.schema.airframe import default_airframe
from uavaero.schema.condition import OperatingCondition
from uavaero.schema.results import (
    AeroResult,
    DragBreakdown,
    DragModel,
    MeshDiagnostics,
    Polar,
    ReferenceValues,
)
from uavaero.schema.stability import StabilityResult


def _result(**kwargs) -> AeroResult:
    """A healthy three-model result, which each test then breaks in one specific way."""
    alpha = np.array([-2.0, 2.0, 6.0, 10.0])
    cl = np.array([0.10, 0.45, 0.80, 1.10])
    cd = np.array([0.020, 0.024, 0.032, 0.048])
    drag = DragBreakdown(cd_induced=[0.001, 0.007, 0.021, 0.040])
    for key, offset in (("vspaero_native", 0.0), ("parasite_buildup", 0.004),
                        ("strip_neuralfoil", -0.003)):
        drag.add(DragModel(key, key, list(cd + offset), basis="test"))

    base = dict(
        polar=Polar(alpha, cl, cd, np.zeros(4), source="vspaero"),
        drag=drag,
        reference=ReferenceValues(0.6, 2.4, 0.2533),
        mesh=MeshDiagnostics(nodes=800, loops=1500, kutta_nodes=27, surfaces=3),
        solver="vspaero", primary_model="strip_neuralfoil",
    )
    base.update(kwargs)
    return AeroResult(**base)


# ════════ the exploits the gate closes ══════════════════════════════════════════════════════


def test_a_healthy_result_is_trusted():
    assert trust(_result()).ok


def test_a_degenerate_mesh_cannot_be_scored():
    """It returns zeros rather than failing, and zero drag is an infinite lift-to-drag ratio — the
    single most attractive number an optimiser can be shown."""
    verdict = trust(_result(mesh=MeshDiagnostics(nodes=800, loops=1500, kutta_nodes=0)))
    assert not verdict
    assert "degenerate" in verdict.reason


def test_negative_induced_drag_cannot_be_scored():
    """Measured from the real thick-panel solve: CDi of -0.14 with every mesh diagnostic green.
    Better than zero, so an optimiser maximising L/D drives straight at it."""
    broken = _result()
    broken.drag.cd_induced = [-0.14, -0.13, 0.002, 0.03]
    verdict = trust(broken)
    assert not verdict
    assert "negative" in verdict.reason


def test_non_positive_drag_cannot_be_scored():
    broken = _result()
    broken.drag.models["strip_neuralfoil"].cd = [0.0, 0.02, 0.03, 0.04]
    assert not trust(broken)


def test_a_result_missing_a_drag_model_cannot_be_scored():
    """The comparison's whole value is that three methods disagree. A candidate whose expensive,
    pessimistic model failed is scored on its two optimistic ones — and across a population that is
    a systematic bias towards shapes that break the strip correction."""
    broken = _result()
    del broken.drag.models["parasite_buildup"]
    verdict = trust(broken)
    assert not verdict
    assert "drag model" in verdict.reason


def test_an_under_estimated_drag_cannot_be_scored():
    """Phase 2 counts strips it could not evaluate and says so in the notes. This is what makes
    that note bite: a wing pushed outside the section data loses viscous drag and looks efficient."""
    verdict = trust(_result(notes=[
        "3 strip(s) had no usable section data - the strip-corrected drag is an under-estimate"
    ]))
    assert not verdict
    assert "under-estimate" in verdict.reason


def test_a_stalled_section_cannot_be_scored():
    """A panel method reports a stalled section as still making lift, so the polar stays smooth and
    the candidate looks like a high-CL wing rather than a stalled one."""
    verdict = trust(_result(native={"stall": {"ok": False, "summary": "tip is past Cl_max"}}))
    assert not verdict
    assert "stall" in verdict.reason


def test_a_non_finite_drag_cannot_be_scored():
    broken = _result()
    broken.drag.models["strip_neuralfoil"].cd = [float("nan"), 0.02, 0.03, 0.04]
    assert not trust(broken)


def test_an_empty_polar_cannot_be_scored():
    empty = _result(polar=Polar(np.array([]), np.array([]), np.array([]), np.array([])))
    assert not trust(empty)


def test_the_gate_is_blunt_rather_than_a_penalty():
    """A softer rule — penalise a bit, score the rest — leaves a gradient pointing into the broken
    region, and a differential-evolution population will follow it. Refusal has no gradient."""
    verdict = trust(_result(mesh=MeshDiagnostics(kutta_nodes=0)))
    assert verdict.ok is False
    assert verdict.reason, "a refusal must say why, or a run cannot be diagnosed"


# ════════ objectives ════════════════════════════════════════════════════════════════════════


def test_candidates_are_compared_at_a_lift_coefficient_not_an_angle():
    """Comparing at a fixed angle compares two aircraft at different lift coefficients, and the one
    that happens to make more lift at 4 degrees wins on incidence alone."""
    objective = max_lift_to_drag(target_cl=0.80)
    # CL 0.80 is the third sample, where CD (strip-corrected) is 0.032 - 0.003 = 0.029.
    assert objective(_result()) == pytest.approx(-(0.80 / 0.029), rel=1e-6)


def test_maximising_is_negated_at_source():
    """Every objective returns a value to minimise, so no flag has to be threaded through the
    driver — and no caller can get the sign wrong."""
    assert max_lift_to_drag()(_result()) < 0
    assert max_endurance_factor()(_result()) < 0
    assert min_drag()(_result()) > 0


def test_endurance_is_not_the_same_optimum_as_glide():
    """CL^1.5/CD peaks at a higher lift coefficient than CL/CD. An optimiser told to maximise L/D
    hands back an aircraft that stays up for less time."""
    result = _result()
    glide = [0.80 / (result.cd[i]) for i in range(4)]
    endurance = [float(result.polar.CL[i]) ** 1.5 / result.cd[i] for i in range(4)]
    assert int(np.argmax(endurance)) >= int(np.argmax(glide))


def test_a_zero_drag_objective_returns_the_rejection_score_not_infinity():
    """SciPy can turn an infinity into a NaN internally, which converts "this candidate is bad"
    into "the whole run is broken"."""
    broken = _result()
    for model in broken.drag.models.values():
        model.cd = [0.0] * 4
    assert max_lift_to_drag()(broken) == REJECTED_SCORE


# ════════ screening, before anything expensive ══════════════════════════════════════════════


def _airframe(**kwargs):
    from uavaero.geometry.planform import Planform

    defaults = dict(span=2.4, aspect_ratio=9.6, taper=0.667, sweep_mode="straight_qc")
    defaults.update(kwargs)
    return authoring.build(
        "candidate", Planform(**defaults),
        htail=authoring.TailSizing(0.098, 2.5, 0.88),
        vtail=authoring.TailSizing(0.034, 1.85, 0.88),
    )


def test_a_sane_aircraft_passes():
    assert screen(_airframe(), OperatingCondition(scope="wing_tail"))


def test_a_vanishing_tip_chord_is_refused_because_it_would_flatter_itself():
    """Not merely small: below the section data the strip correction drops those strips, the
    profile drag comes out low, and the candidate scores well. That is the exploit, not the size.

    The extreme corner of the standard bounds — 3.2 m span, aspect ratio 16, taper 0.35 — produces
    a 104 mm tip, which is comfortably above the 40 mm default. That is worth knowing: the default
    limit is permissive and it is the *Reynolds* check below that does the real work on a slow
    aircraft. This test pins the chord mechanism against a limit that bites.
    """
    aircraft = _airframe(span=3.2, aspect_ratio=16.0, taper=0.35)
    tip = min(s.chord for s in aircraft.wing.sections)
    assert tip == pytest.approx(0.104, abs=0.005), "the corner of the bounds, for the record"
    assert screen(aircraft, OperatingCondition(scope="wing_tail")), "40 mm default does not bite"

    verdict = screen(aircraft, OperatingCondition(scope="wing_tail"), Limits(min_chord_m=0.15))
    assert not verdict
    assert "chord" in verdict.reason


def test_a_low_reynolds_tip_is_refused():
    """Where the polars have no data, the correction silently drops strips."""
    verdict = screen(_airframe(), OperatingCondition(scope="wing_tail", velocity_mps=2.0),
                     Limits())
    assert not verdict
    assert "Reynolds" in verdict.reason


def test_an_oversized_span_is_refused():
    verdict = screen(_airframe(span=3.2), OperatingCondition(scope="wing_tail"),
                     Limits(max_span_m=2.5))
    assert not verdict
    assert "span" in verdict.reason


def test_a_tail_on_top_of_the_wing_is_refused():
    """An arm range reaching zero puts the tail on the wing, where it produces a delightful drag
    and no pitch stiffness at all."""
    aircraft = _airframe()
    authoring.apply_tail(aircraft, authoring.TailSizing(0.098, 2.5, 0.02), "htail")
    verdict = screen(aircraft, OperatingCondition(scope="wing_tail"))
    assert not verdict
    assert "behind the wing" in verdict.reason


def test_screening_is_cheap_enough_to_be_worth_it():
    """It has to cost microseconds against a solve's tens of seconds, or there is no point."""
    import time

    aircraft, condition = _airframe(), OperatingCondition(scope="wing_tail")
    started = time.perf_counter()
    for _ in range(200):
        screen(aircraft, condition)
    assert (time.perf_counter() - started) / 200 < 0.005


# ════════ the ledger ════════════════════════════════════════════════════════════════════════


def test_the_ledger_names_the_commonest_rejection():
    """The most useful diagnostic in a run. A population 90% rejected for one reason means the
    bounds are wrong, and that shows up here long before it shows up as slow convergence."""
    ledger = RejectionLedger()
    for _ in range(9):
        ledger.record("tip chord 12 mm is below 40 mm, where the section data runs out")
    ledger.record("span 4.10 m exceeds the 4.00 m limit")

    assert ledger.total == 10
    assert ledger.worst[1] == 9
    assert "tip chord" in ledger.worst[0]
    assert "9x" in ledger.summary().replace("    9x", "9x")


# ════════ the design space ══════════════════════════════════════════════════════════════════


def test_the_space_is_within_the_workable_band():
    space = DesignSpace(default_airframe(), standard_variables())
    assert 8 <= len(space.variables) <= 20


def test_area_is_derived_rather_than_driven():
    """Span, area and aspect ratio are three names for two numbers. Exposing all three lets the
    search ask for combinations that cannot exist, which it does immediately."""
    space = DesignSpace(default_airframe(), standard_variables())
    assert "area" not in space.names
    assert {"span", "aspect_ratio"} <= set(space.names)

    vector = space.centre()
    aircraft = space.build(vector)
    facts = authoring.summary(aircraft)
    span, aspect = facts["wing_span_m"], facts["wing_aspect_ratio"]
    assert facts["wing_area_m2"] == pytest.approx(span**2 / aspect, rel=1e-6)


def test_building_a_candidate_does_not_disturb_the_base():
    """An optimiser sharing one airframe between candidates accumulates every change it ever
    tried, and the hundredth candidate is the sum of the first ninety-nine."""
    base = default_airframe()
    before = base.to_dict()
    space = DesignSpace(base, standard_variables())
    space.build(space.centre())
    space.build(np.array([v.high for v in space.variables]))
    assert base.to_dict() == before


def test_changing_wing_sweep_holds_the_tail_volumes():
    """The arm is measured from the wing's aerodynamic centre, which a sweep change moves. Without
    re-placing the tails, an optimiser varying sweep would be silently varying tail volume and
    would find improvements that are nothing of the kind."""
    space = DesignSpace(default_airframe(), standard_variables())
    vector = space.centre()
    sweep_index = space.names.index("sweep_deg")

    straight = vector.copy()
    straight[sweep_index] = 0.0
    swept = vector.copy()
    swept[sweep_index] = 20.0

    before = authoring.summary(space.build(straight))
    after = authoring.summary(space.build(swept))
    assert after["htail_arm_m"] == pytest.approx(before["htail_arm_m"], abs=1e-9)
    assert after["horizontal_tail_volume"] == pytest.approx(
        before["horizontal_tail_volume"], rel=1e-6
    )


def test_the_tail_volume_asked_for_is_the_tail_volume_built():
    """The tails are driven by volume coefficient, and the area is derived from the wing the vector
    just asked for. This is the round trip: `Vh` in, the same `Vh` back out of the geometry.

    It has to hold across the *whole* space, not at the centre, because the wing area swings by 23x
    between the bounds and that is precisely what the derivation has to absorb.
    """
    space = DesignSpace(default_airframe(), standard_variables())
    names = space.names
    h, v = names.index("htail_volume"), names.index("vtail_volume")
    low = np.array([b[0] for b in space.bounds])
    high = np.array([b[1] for b in space.bounds])
    rng = np.random.default_rng(0)

    for _ in range(50):
        vector = low + rng.random(len(low)) * (high - low)
        facts = authoring.summary(space.build(vector))
        assert facts["horizontal_tail_volume"] == pytest.approx(vector[h], rel=1e-9)
        assert facts["vertical_tail_volume"] == pytest.approx(vector[v], rel=1e-9)


def test_no_candidate_in_the_space_is_rejected_for_its_tail_volume():
    """The point of the change. Sampling tail *area* against a wing the optimiser is simultaneously
    resizing left `Vh` uncontrolled by any variable: it goes as `S_h·l·b / S_w²`, the wing area
    swings 23x across the bounds, and roughly 30% of every population was rejected for a tail
    volume nothing had chosen. Measured before: 71.3% passed. After: all of them.
    """
    space = DesignSpace(default_airframe(), standard_variables())
    condition = OperatingCondition(velocity_mps=15.0, altitude_m=100.0, scope="aircraft")
    low = np.array([b[0] for b in space.bounds])
    high = np.array([b[1] for b in space.bounds])
    rng = np.random.default_rng(1)

    for _ in range(200):
        vector = low + rng.random(len(low)) * (high - low)
        verdict = screen(space.build(vector), condition)
        assert verdict.ok, verdict.reason


def test_the_arm_is_applied_before_the_volume():
    """Order dependence with teeth. The area follows from the volume and the arm together, so a
    vector that set the volume and then moved the tail would silently change the volume it had just
    asked for. The list order is the only thing preventing that, so it is asserted."""
    names = [v.name for v in standard_variables()]
    for tail in ("htail", "vtail"):
        assert names.index(f"{tail}_arm") < names.index(f"{tail}_volume")


def test_values_are_clipped_to_their_bounds():
    space = DesignSpace(default_airframe(), standard_variables())
    beyond = np.array([v.high * 10 for v in space.variables])
    aircraft = space.build(beyond)
    assert authoring.summary(aircraft)["wing_span_m"] <= space.variables[0].high + 1e-9


def test_a_variable_that_cannot_move_the_aircraft_is_dropped():
    """A tail knob on a tailless airframe applies to nothing — `_tail_setter` returns early, so the
    dimension is inert. Differential evolution still samples and mutates it every generation, and
    it lands in the reported best as a number describing a surface that is not there."""
    tailless = copy.deepcopy(default_airframe())
    for kind in ("htail", "vtail"):
        authoring.remove_component(tailless, kind)

    space = DesignSpace(tailless, standard_variables())
    assert space.names == ["span", "aspect_ratio", "taper", "sweep_deg", "dihedral_deg",
                           "tip_twist_deg"]
    assert set(space.omitted) == {
        "htail_arm", "htail_volume", "htail_aspect",
        "vtail_arm", "vtail_volume", "vtail_aspect",
    }


def test_only_the_missing_surfaces_knobs_are_dropped():
    """Losing the fin must not cost the horizontal tail its variables."""
    finless = copy.deepcopy(default_airframe())
    authoring.remove_component(finless, "vtail")

    space = DesignSpace(finless, standard_variables())
    assert space.omitted == ["vtail_arm", "vtail_volume", "vtail_aspect"]
    assert "htail_volume" in space.names


def test_a_complete_aircraft_keeps_every_variable():
    space = DesignSpace(default_airframe(), standard_variables())
    assert space.omitted == []
    assert len(space.variables) == 12


def test_a_vector_is_read_against_the_variables_that_survived():
    """The vector is positional, so dropping a variable has to shorten the space's own bounds and
    names together or every value lands on the wrong knob."""
    tailless = copy.deepcopy(default_airframe())
    for kind in ("htail", "vtail"):
        authoring.remove_component(tailless, kind)

    space = DesignSpace(tailless, standard_variables())
    assert len(space.bounds) == len(space.names) == len(space.centre())

    vector = space.centre()
    facts = authoring.summary(space.build(vector))
    span_index = space.names.index("span")
    assert facts["wing_span_m"] == pytest.approx(vector[span_index], rel=1e-9)


def test_an_unworkably_large_space_is_refused():
    variables = standard_variables() * 3
    with pytest.raises(ValueError, match="workable range"):
        DesignSpace(default_airframe(), variables)


# ════════ post-solve constraints ════════════════════════════════════════════════════════════


def _derivatives(**kwargs) -> StabilityResult:
    base = dict(CL_alpha=5.5, Cm_alpha=-1.0, Cn_beta=0.04, Cl_beta=-0.08,
                cref_m=0.2533, x_cg_m=0.085, neutral_point_x_m=0.1312,
                static_margin_reported=0.1822)
    base.update(kwargs)
    return StabilityResult(**base)


def test_a_conventional_aircraft_satisfies_the_constraints():
    assert StabilityConstraints().check(_derivatives())


def test_a_static_margin_outside_the_band_is_refused():
    verdict = StabilityConstraints(static_margin=(0.05, 0.10)).check(_derivatives())
    assert not verdict
    assert "static margin" in verdict.reason


def test_directional_instability_is_refused():
    verdict = StabilityConstraints().check(_derivatives(Cn_beta=-0.02))
    assert not verdict
    assert "directionally" in verdict.reason


def test_an_internally_inconsistent_derivative_set_is_refused():
    """If the three routes to static margin disagree, the perturbation values are being read wrong
    and every derivative is suspect — including the ones the constraint just checked."""
    verdict = StabilityConstraints().check(_derivatives(Cm_alpha=-4.0))
    assert not verdict
    assert "inconsistent" in verdict.reason
