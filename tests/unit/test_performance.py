"""The speed sweep: level-flight drag and power against airspeed.

Two properties are worth more than the rest.

**Best endurance is slower than best range.** For a parabolic polar the minimum-power speed is
`3^-1/4` — about 0.76 — times the minimum-drag speed, and that is a known answer this can be
checked against rather than a number to eyeball. It also matters practically: confusing the two is
how a loitering aircraft ends up cruising too fast for the job it was built for.

**A speed outside the solved polar is dropped, never extrapolated.** At the slow end the required
lift coefficient is simply more than the wing reached in the sweep, and a fitted `k·CL²` will
happily return a number there — smooth, plausible, and invented. That is the same refusal
`sections/base.cd_at_cl` makes across a gap in a section polar, one level up.

Built on a synthetic `AeroResult` rather than a solve, so these run in milliseconds and test the
arithmetic rather than VSPAERO. The real solver path is exercised in the reference cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from uavaero.aero.performance import GRAVITY, speed_sweep
from uavaero.schema.airframe import default_airframe
from uavaero.schema.results import AeroResult, DragBreakdown, Polar, ReferenceValues, SpanLoad


class _FlatPolarSource:
    """A section source with one constant profile drag, so the viscous half cannot move the
    answer. That isolates the induced-drag and trim arithmetic, which is what these test."""

    name = "flat"

    def cd(self, airfoil: str, reynolds: float, cl: float) -> float:
        return 0.010

    def cl_max(self, airfoil: str, reynolds: float) -> float:
        return 1.4


def _result(airframe, cl_max: float = 1.3) -> AeroResult:
    """A clean parabolic polar: CL linear in alpha, CDi = k·CL²."""
    alpha = np.linspace(-4.0, 12.0, 17)
    cl = np.linspace(-0.25, cl_max, 17)
    k = 0.045
    induced = k * cl**2
    span = airframe.wing.span()
    ys = list(np.linspace(0.05 * span, 0.95 * span, 8))
    chords = [airframe.wing.sections[0].chord] * 8
    return AeroResult(
        polar=Polar(alpha, cl, induced + 0.008, np.zeros_like(cl), source="synthetic"),
        drag=DragBreakdown(cd_induced=[float(v) for v in induced]),
        span_loads=[
            SpanLoad(alpha_deg=float(a), y_m=ys, chord_m=chords,
                     cl_local=[float(c)] * 8)
            for a, c in zip(alpha, cl)
        ],
        reference=ReferenceValues(sref_m2=airframe.wing.area()),
    )


@pytest.fixture
def airframe():
    return default_airframe()


@pytest.fixture
def sweep(airframe):
    return speed_sweep(
        _result(airframe), airframe, mass_kg=2.5,
        speeds_mps=list(np.arange(9.0, 30.5, 0.5)),
        lookup=_FlatPolarSource(),
    )


# ════════ the arithmetic of level flight ════════════════════════════════════════════════════


def test_the_required_lift_coefficient_falls_as_the_square_of_speed(sweep):
    """Lift equals weight, so CL is fixed by speed - it is not a free choice at any point on this
    curve, which is the whole difference between a speed sweep and an alpha sweep."""
    fast = [p for p in sweep.points if p.velocity_mps in (10.0, 20.0)]
    assert len(fast) == 2
    slow_point, fast_point = fast
    assert fast_point.cl_required == pytest.approx(slow_point.cl_required / 4.0, rel=1e-6)


def test_lift_equals_weight_at_every_point(sweep, airframe):
    """The defining constraint, asserted directly rather than trusted."""
    density = sweep.density
    for point in sweep.points:
        lift = 0.5 * density * point.velocity_mps**2 * sweep.sref_m2 * point.cl_required
        assert lift == pytest.approx(2.5 * GRAVITY, rel=1e-9)


# ════════ the known answer ══════════════════════════════════════════════════════════════════


def test_best_endurance_is_slower_than_best_range(sweep):
    """The property that makes this curve worth computing at all."""
    assert sweep.best_endurance().velocity_mps < sweep.best_range().velocity_mps


def test_the_two_speeds_sit_where_a_parabolic_polar_says_they_should(sweep):
    """`V_minpower / V_mindrag = 3^(-1/4)` for CD = CD0 + k·CL². A known answer, so the check is
    against theory rather than against a remembered number from a previous run."""
    ratio = sweep.best_endurance().velocity_mps / sweep.best_range().velocity_mps
    assert ratio == pytest.approx(3.0 ** -0.25, rel=0.06)


def test_maximum_lift_to_drag_is_where_induced_and_profile_drag_are_equal(sweep):
    """The other classic identity: at minimum drag the two halves are the same size. It exercises
    the split rather than the total, so a bug that moved drag between them would show here and
    nowhere else."""
    best = sweep.best_range()
    assert best.cd_induced == pytest.approx(best.cd_profile, rel=0.10)


# ════════ refusing, rather than inventing ═══════════════════════════════════════════════════


def test_a_speed_below_the_solved_polar_is_dropped_not_extrapolated(airframe):
    """The fit would happily answer past the last solved point, and the number would look
    entirely reasonable."""
    result = _result(airframe, cl_max=0.9)
    sweep = speed_sweep(
        result, airframe, mass_kg=2.5, speeds_mps=[6.0, 8.0, 20.0], lookup=_FlatPolarSource(),
    )
    flown = [p.velocity_mps for p in sweep.points]
    assert 6.0 not in flown, "6 m/s needs more CL than the polar reached"
    assert 20.0 in flown
    assert any("outside the solved polar" in note for note in sweep.notes)


def test_a_sweep_with_no_stated_mass_refuses_rather_than_returning_zeros(airframe):
    """Every point divides by weight. Mass is stated, never sized - and an unstated mass has to
    stop the calculation, not silently produce an aircraft that weighs nothing and glides
    forever."""
    sweep = speed_sweep(
        _result(airframe), airframe, mass_kg=0.0, speeds_mps=[10.0, 20.0],
        lookup=_FlatPolarSource(),
    )
    assert sweep.points == []
    assert any("stated mass" in note for note in sweep.notes)


def test_missing_strips_are_reported_because_the_emptiest_point_looks_the_fastest(airframe):
    """A speed whose profile drag lost stations is the speed that will win a comparison, so the
    count travels with the result."""

    class _Blind(_FlatPolarSource):
        name = "blind"

        def cd(self, airfoil, reynolds, cl):
            return None

    sweep = speed_sweep(
        _result(airframe), airframe, mass_kg=2.5, speeds_mps=[12.0, 20.0], lookup=_Blind(),
    )
    assert all(p.missing_strips > 0 for p in sweep.points)
    assert any("floor rather than an estimate" in note for note in sweep.notes)


def test_power_required_is_drag_times_speed(sweep):
    """No propeller and no motor efficiency is assumed anywhere, because none is known - this is
    the power the air demands, not the power a battery has to supply."""
    point = sweep.points[len(sweep.points) // 2]
    density, sref = sweep.density, sweep.sref_m2
    assert point.power_watts(density, sref) == pytest.approx(
        point.drag_newtons(density, sref) * point.velocity_mps
    )
