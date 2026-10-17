"""The end-to-end solve: a real airframe, a real OpenVSP build, a real VSPAERO run.

This is the acceptance case for the solver layer, and it is deliberately a *known-answer* test
rather than a smoke test. "It ran and produced numbers" is exactly what every failure mode in this
pipeline also does — a degenerate mesh returns zeros, a half-span reference value returns a
plausible polar for the wrong aircraft, and neither raises.

The load-bearing assertion is `test_the_lift_curve_slope_matches_lifting_line_theory`. Lifting-line
gives CL_alpha = a0 / (1 + a0/(pi AR e)) for a finite wing, and that is a strong enough constraint
to catch the reference-value bug this codebase has actually had: feeding VSPAERO the half span as
`bref` quartered the aspect ratio, and at AR 2.4 rather than 9.6 the predicted slope moves from
about 5.0 to about 3.0 — far outside any tolerance. The original bug was caught by noticing an
Oswald efficiency above 1, which is luck. This is not luck.

Marked slow: each solve is tens of seconds of real solver time.
"""
from __future__ import annotations

import math

import pytest

from tests.markers import requires_neuralfoil, requires_openvsp
from uavaero.schema.airframe import ParametricAirframe, default_airframe
from uavaero.schema.condition import Fidelity, OperatingCondition
from uavaero.solvers.base import SolveFailed
from uavaero.solvers.vspaero import VSPAeroSolver
from uavaero.store.cache import CachedSolver, ResultCache

pytestmark = [requires_openvsp, pytest.mark.slow]


@pytest.fixture(scope="module")
def wing_solve():
    """One QUICK wing-scope solve of the conventional starter, shared across this module.

    Module-scoped because it is tens of seconds and every assertion below interrogates the same
    result from a different angle.
    """
    airframe = default_airframe()
    condition = OperatingCondition(fidelity=Fidelity.QUICK, scope="wing", velocity_mps=20.0)
    return airframe, condition, VSPAeroSolver().evaluate(airframe, condition, workdir=None)


# ════════ the answer is the right shape ═════════════════════════════════════════════════════


def test_a_solve_produces_a_polar_over_the_requested_angles(wing_solve):
    _, condition, result = wing_solve
    assert len(result.polar.alpha_deg) == len(condition.alphas())
    assert result.polar.CL[0] < result.polar.CL[-1], "lift must increase with incidence"


def test_the_mesh_is_healthy(wing_solve):
    """A mesh with no Kutta nodes solves to zeros rather than failing, and zeros read as a
    spectacularly efficient aircraft."""
    _, _, result = wing_solve
    assert result.mesh.ok
    assert result.mesh.kutta_nodes and result.mesh.kutta_nodes > 0


def test_the_reference_values_come_from_the_wing_with_the_span_doubled(wing_solve):
    """`LiftingSurface.span()` is one side; `area()` is already doubled. This is the asymmetry that
    produced a real bug, so the reference values are asserted against the geometry directly rather
    than against a remembered number."""
    airframe, _, result = wing_solve
    wing = airframe.wing
    assert result.reference.sref_m2 == pytest.approx(wing.area())
    assert result.reference.bref_m == pytest.approx(2.0 * wing.span())
    assert result.reference.cref_m == pytest.approx(wing.mean_aerodynamic_chord())
    assert result.reference.aspect_ratio == pytest.approx(9.6, rel=1e-3)


def test_the_lift_curve_slope_matches_lifting_line_theory(wing_solve):
    """The known-answer check, and the one that would have caught the half-span reference bug.

    CL_alpha = a0 / (1 + a0 / (pi AR e)) with a0 = 2 pi. At AR 9.6 that is about 5.0 per radian;
    at the quartered AR a half-span `bref` would produce, it is about 3.0. A 12% tolerance is wide
    enough for the difference between a panel solve and thin-airfoil theory and nowhere near wide
    enough to admit the bug.
    """
    _, _, result = wing_solve
    ar = result.reference.aspect_ratio
    _, e = result.fit_cd0_e()
    expected = 2 * math.pi / (1 + 2 * math.pi / (math.pi * ar * e))
    assert result.cl_alpha_per_rad == pytest.approx(expected, rel=0.12)


def test_the_result_records_what_produced_it(wing_solve):
    _, _, result = wing_solve
    assert result.solver == "vspaero"
    assert result.fidelity == "quick"
    assert result.scope == "wing"
    assert result.elapsed_s > 0
    assert result.reynolds > 0 and 0 < result.mach < 0.3


# ════════ the drag comparison ═══════════════════════════════════════════════════════════════


def test_the_drag_models_disagree(wing_solve):
    """The disagreement is the product, not a defect. VSPAERO's own total drag uses a wetted-area
    parasite buildup that knows nothing about the boundary layer; the strip correction solves each
    station's section at its own Reynolds number. At UAS Reynolds numbers these differ by tens of
    percent, and collapsing them to one number would be inventing a confidence nobody has.

    (Phase 0 asserted two models, Phase 2 three. On 2026-09-02 VSPAERO's own total was dropped and
    the remaining two were re-cut: the wing by section polars, the body by the wetted-area buildup.
    What stays here is the property that outlives any particular list of models.)
    """
    _, _, result = wing_solve
    assert len(result.drag.predictions()) >= 2
    assert result.drag.relative_spread(1) > 0.01


def test_every_drag_model_says_how_it_was_arrived_at(wing_solve):
    """When two models disagree by 30% the first question is what each one actually did. An answer
    that has to be reconstructed from source is an answer nobody checks."""
    _, _, result = wing_solve
    for model in result.drag.models.values():
        assert len(model.basis) > 40, model.key


def test_the_primary_model_is_the_one_with_section_data_and_is_not_double_counted(wing_solve):
    """The primary is whichever model carries real section data, and it must sit above the induced
    drag alone (it adds profile drag) while staying below the buildup applied to everything (which
    would be counting the wing twice)."""
    _, _, result = wing_solve
    assert result.primary_model in result.drag.models
    assert "neuralfoil" in result.primary_model
    primary = result.drag.cd(result.primary_model)
    induced = result.drag.cd_induced
    assert all(c > i for c, i in zip(primary, induced)), "the primary must exceed induced alone"
    buildup = result.drag.models.get("parasite_buildup")
    if buildup is not None:
        assert not all(c > n for c, n in zip(primary, buildup.cd)), "looks like double counting"


def test_the_span_load_is_captured_at_every_angle(wing_solve):
    """One span load per alpha, not just the last. The strip correction integrates each one, and
    the stall check needs the highest-incidence one specifically."""
    _, condition, result = wing_solve
    assert len(result.span_loads) == len(condition.alphas())
    tip_load = result.span_loads[-1]
    assert len(tip_load.y_m) == len(tip_load.chord_m) == len(tip_load.cl_local)
    assert all(c > 0 for c in tip_load.chord_m)


# ════════ refusing, rather than answering the wrong question ════════════════════════════════


def test_a_scope_that_needs_a_tail_is_refused_before_anything_expensive_runs():
    """Not a solver error — a question that cannot be asked of this airframe yet. The answer is a
    sentence naming what is missing, checked before OpenVSP is even started."""
    bare = ParametricAirframe(name="bare-wing", wing=default_airframe().wing)
    solver = VSPAeroSolver()

    assert "no tail" in solver.unavailable_for(bare, OperatingCondition(scope="aircraft"))
    with pytest.raises(SolveFailed, match="no tail"):
        solver.evaluate(bare, OperatingCondition(scope="aircraft"))


def test_a_wing_with_one_section_cannot_be_meshed():
    from uavaero.schema.airframe import LiftingSurface, WingSection

    stub = ParametricAirframe(
        name="stub",
        wing=LiftingSurface(name="wing", sections=[WingSection(chord=0.3)]),
    )
    assert "two sections" in VSPAeroSolver().unavailable_for(
        stub, OperatingCondition(scope="wing")
    )


# ════════ the fidelity ladder, against the real solver ══════════════════════════════════════


@requires_neuralfoil
def test_a_finer_rung_actually_produces_a_finer_mesh(tmp_path):
    """The ladder's claim, checked against the solver rather than against the settings table. A
    rung that resolved to a different number but the same mesh would pass every unit test and cost
    exactly as much as the rung above it."""
    airframe = default_airframe()
    solver = VSPAeroSolver()
    coarse = solver.evaluate(
        airframe, OperatingCondition(fidelity=Fidelity.QUICK, scope="wing"),
        workdir=tmp_path / "quick",
    )
    fine = solver.evaluate(
        airframe, OperatingCondition(fidelity=Fidelity.STANDARD, scope="wing", alpha_points=3),
        workdir=tmp_path / "standard",
    )
    assert fine.mesh.nodes > coarse.mesh.nodes
    assert len(fine.span_loads[-1].y_m) > len(coarse.span_loads[-1].y_m)


def test_the_second_identical_solve_is_served_from_cache(tmp_path):
    """The Phase 0 criterion. Also the property the whole optimiser rests on: a converged
    population is a population full of near-duplicates, and re-solving them is the difference
    between an overnight run and a week."""
    cache = ResultCache(tmp_path / "results.db")
    solver = CachedSolver(VSPAeroSolver(), cache)
    airframe = default_airframe()
    condition = OperatingCondition(fidelity=Fidelity.QUICK, scope="wing")

    first = solver.evaluate(airframe, condition, workdir=tmp_path / "run")
    second = solver.evaluate(airframe, condition, workdir=tmp_path / "run")

    assert cache.stats["hits"] == 1 and cache.stats["misses"] == 1
    assert second.to_dict() == first.to_dict()
    assert second.elapsed_s == first.elapsed_s, "a cache hit must not restate the solve time"


# ════════ Phase 2: three models, and the airfoil actually mattering ═════════════════════════


@pytest.fixture(scope="module")
def three_model_solve():
    """A wing-scope solve carrying all three drag models."""
    airframe = default_airframe()
    condition = OperatingCondition(fidelity=Fidelity.QUICK, scope="wing", velocity_mps=20.0)
    return airframe, condition, VSPAeroSolver().evaluate(airframe, condition)


def test_a_solve_produces_independent_drag_estimates(three_model_solve):
    """Each surface by the method that suits it, plus one wholly independent cross-check. VSPAERO's
    own total was dropped on 2026-09-02: it is the same wetted-area method as the buildup, applied
    less transparently, so it added a third number without adding a third opinion."""
    _, _, result = three_model_solve
    assert "vspaero_native" not in result.drag.models
    assert len(result.drag.models) >= 2
    for model in result.drag.models.values():
        assert len(model.basis) > 60, model.key
        assert len(model.cd) == len(result.polar.CL)


def test_they_disagree_enough_to_be_worth_keeping_apart(three_model_solve):
    """If they agreed to within a percent there would be no reason to carry three. At UAS Reynolds
    numbers they do not: measured spread on this case is around 40%."""
    _, _, result = three_model_solve
    middle = len(result.polar.CL) // 2
    assert result.drag.relative_spread(middle) > 0.05
    assert result.drag.spread(middle) > 0


def test_the_parasite_buildup_ran_in_metres(three_model_solve):
    """The unit trap, checked on a real run rather than a fixture. `LengthUnit` must be `LEN_M`;
    the default is imperial and `PD_UNITS_METRIC` is centimetres."""
    _, condition, result = three_model_solve
    parasite = result.native.get("parasite")
    assert parasite is not None, "the buildup should have run"
    wing = parasite["components"][0]
    expected = condition.atmosphere.reynolds(condition.velocity_mps, wing["reference_length_m"])
    assert wing["reynolds"] == pytest.approx(expected, rel=0.02)


def test_the_stall_check_reports_a_margin(three_model_solve):
    _, _, result = three_model_solve
    report = result.native.get("stall")
    assert report is not None
    assert report["stations"]
    assert report["summary"]


@requires_neuralfoil
def test_a_thicker_airfoil_moves_the_strip_model_more_than_the_native_one(tmp_path):
    """The other half of the Phase 2 criterion, and the reason the strip correction exists.

    Same planform, same camber line, different thickness — naca2412 against naca2418. VSPAERO's own
    drag barely notices, because a panel method's parasite term is a wetted-area buildup that is
    nearly blind to section shape. The strip correction notices, because it is reading a real 2D
    polar for each section at its own Reynolds number.

    If this ever fails, airfoil choice has stopped affecting the answer, which is the failure mode
    the whole model exists to prevent — and it is silent, because every number still looks fine.
    """
    import dataclasses

    def solve(airfoil: str):
        airframe = default_airframe()
        airframe.wing.sections = [
            dataclasses.replace(s, airfoil=airfoil) for s in airframe.wing.sections
        ]
        condition = OperatingCondition(fidelity=Fidelity.QUICK, scope="wing", velocity_mps=20.0)
        return VSPAeroSolver().evaluate(airframe, condition, workdir=tmp_path / airfoil)

    thin, thick = solve("naca2412"), solve("naca2418")
    middle = len(thin.polar.CL) // 2

    def shift(key: str) -> float:
        a, b = thin.drag.cd(key)[middle], thick.drag.cd(key)[middle]
        return abs(b - a) / a

    strip_shift = shift(thin.primary_model)
    buildup_shift = shift("parasite_buildup") if "parasite_buildup" in thin.drag.models else 0.0
    assert strip_shift > 0.02, "an 18% section must cost measurably more profile drag than a 12%"
    assert strip_shift > buildup_shift, (
        f"the section-based model must be the one that responds to section shape "
        f"(section {strip_shift:.1%} vs buildup {buildup_shift:.1%})"
    )


@requires_openvsp
def test_the_solve_emits_no_parasite_drag_warning(tmp_path):
    """The regression for the component set.

    `VSPAEROComputeGeometry` leaves an `NGonMesh` in the model, and handing ParasiteDrag `SET_ALL`
    hands it that mesh — which it reports as `Warning: Geom ID ... not included in Parasite Drag
    calculation`. Harmless to the numbers, but it is the analysis being asked the wrong question,
    and it sat unnoticed for two phases because nothing in the app displayed solver output.

    Asserted on the *aircraft* scope, because that is the one where the fuselage has to survive
    into the buildup — a set fix that silently dropped it would still produce no warning.

    It also pins the component split the composite model rests on. Both buildups run over named
    user sets now, so a mistake in either one is a wrong set rather than a warning.
    """
    from uavaero.solvers.vspaero import run_isolated

    captured: list[str] = []
    condition = OperatingCondition(fidelity=Fidelity.QUICK, scope="aircraft", thick=False)
    result, error = run_isolated(
        default_airframe(), condition, tmp_path / "parasite", timeout_s=600,
        on_output=captured.append,
    )

    assert result is not None, error
    assert not [line for line in captured if "not included in Parasite" in line]

    # Two buildups over deliberately different component sets, and the split is the whole basis
    # of the composite drag model: the body half must NOT contain the wing (the strip integrator
    # covers it, and counting it twice is the failure this guards), while the cross-check must
    # contain everything, or it is not an independent estimate of the same total.
    assert result.parasite is not None
    body = {component.name for component in result.parasite.components}
    assert body == {"htail", "vtail", "fuselage"}, (
        "the composite's buildup covers every component except the wing"
    )

    assert result.parasite_excludes_wing
    assert result.parasite_all is not None
    everything = {component.name for component in result.parasite_all.components}
    assert everything == {"wing", "htail", "vtail", "fuselage"}, (
        "the cross-check covers the whole aircraft, wing included"
    )
