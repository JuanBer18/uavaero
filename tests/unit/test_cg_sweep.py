"""Static margin against CG position, and the modes that follow from it.

The property worth protecting is that **this costs no solve**. The neutral point is a property of
the shape, not of where the mass sits, so one stability run answers the whole curve. If that ever
stopped being true the sweep would silently become twenty solves, and the reason to have it at all
would be gone.

The known answers here are the textbook ones, so the tests are against theory rather than against
numbers remembered from a previous run:

- static margin falls linearly as the CG moves aft, and crosses zero exactly at the neutral point
- short-period frequency falls with it, because pitch stiffness *is* the static margin
- damping ratio rises as the frequency falls, since the damping term does not change with CG

That last one is the reason the sweep is worth looking at rather than computing `(X_np − x)/cref`
in your head: the aircraft gets softer *and* better damped as the CG moves aft, and where those
two cross is a judgement no single number makes for you.
"""
from __future__ import annotations

import pytest

from uavaero.dynamics.cg import CONVENTIONAL_MARGIN, cg_sweep
from uavaero.schema.condition import OperatingCondition
from uavaero.schema.mass import MassProperties
from uavaero.schema.stability import StabilityResult

NEUTRAL_POINT = 0.35
CREF = 0.25


@pytest.fixture
def derivatives():
    """A conventional set, solved with the CG at 0.30 m and a neutral point at 0.35 m."""
    return StabilityResult(
        x_cg_m=0.30, cref_m=CREF, sref_m2=0.6, bref_m=2.4,
        CL=0.45, CL_alpha=5.0, CD=0.03, Cm_alpha=-1.0, Cm_q=-12.0,
        CY_beta=-0.4, Cl_beta=-0.12, Cn_beta=0.08,
        Cl_p=-0.5, Cn_r=-0.1, Cl_r=0.1, Cn_p=-0.03,
        neutral_point_x_m=NEUTRAL_POINT, alpha_deg=3.0,
    )


@pytest.fixture
def sweep(derivatives):
    return cg_sweep(
        derivatives,
        MassProperties(mass_kg=2.5, ixx=0.3, iyy=0.4, izz=0.6),
        OperatingCondition(velocity_mps=20.0),
        [0.20 + 0.01 * i for i in range(19)],
        mac_leading_edge_x_m=0.22,
    )


# ════════ the arithmetic, against its definition ════════════════════════════════════════════


def test_static_margin_is_zero_exactly_at_the_neutral_point(sweep):
    """The definition, asserted rather than assumed. Everything else in this module hangs off it."""
    at_np = min(sweep.points, key=lambda p: abs(p.x_m - NEUTRAL_POINT))
    assert at_np.x_m == pytest.approx(NEUTRAL_POINT, abs=1e-9)
    assert at_np.static_margin == pytest.approx(0.0, abs=1e-9)


def test_the_margin_falls_linearly_as_the_cg_moves_aft(sweep):
    """`SM = (X_np − x)/cref` is a straight line in x with slope −1/cref, so a sweep that curves
    means something other than the definition is being computed."""
    first, last = sweep.points[0], sweep.points[-1]
    slope = (last.static_margin - first.static_margin) / (last.x_m - first.x_m)
    assert slope == pytest.approx(-1.0 / CREF, rel=1e-9)


def test_cm_alpha_and_the_margin_stay_consistent(sweep, derivatives):
    """`SM = −Cm_alpha / CL_alpha` at every point, because they are one statement rearranged."""
    for point in sweep.points:
        assert point.static_margin == pytest.approx(
            -point.Cm_alpha / derivatives.CL_alpha, rel=1e-9
        )


def test_the_sweep_never_runs_a_solver(sweep, derivatives):
    """The whole reason this exists. The neutral point is a property of the shape, so the curve is
    arithmetic on one solve - it must not have consumed the derivative set as a starting guess for
    something more expensive."""
    assert derivatives.x_cg_m == 0.30, "the input must not be mutated"
    assert derivatives.Cm_alpha == -1.0
    assert len(sweep.points) == 19


# ════════ what the modes do, against theory ═════════════════════════════════════════════════


def test_the_short_period_softens_as_the_cg_moves_aft(sweep):
    """Pitch stiffness *is* the static margin, so frequency has to fall with it."""
    flying = [p for p in sweep.points if p.short_period_wn]
    assert len(flying) > 4
    assert flying[0].short_period_wn > flying[-1].short_period_wn


def test_short_period_damping_rises_as_its_frequency_falls(sweep):
    """`zeta = damping / (2 · wn)` and the damping term does not change with CG, so a softer
    aircraft is a better damped one. This is the trade the sweep exists to show."""
    flying = [p for p in sweep.points if p.short_period_wn and p.short_period_damping]
    assert flying[0].short_period_damping < flying[-1].short_period_damping


def test_a_cg_behind_the_neutral_point_is_reported_unstable(sweep):
    """Statically unstable in pitch is not an error, it is a design decision - but it must not be
    reported as if it were merely a small margin."""
    aft = [p for p in sweep.points if p.x_m > NEUTRAL_POINT + 1e-9]
    assert aft, "the sweep must reach past the neutral point to be useful"
    assert all(not p.stable for p in aft)
    assert all(p.static_margin < 0 for p in aft)


# ════════ what it tells the user ════════════════════════════════════════════════════════════


def test_the_conventional_band_is_a_contiguous_range_forward_of_the_neutral_point(sweep):
    low, high = sweep.conventional_span()
    assert low < high < NEUTRAL_POINT
    for point in sweep.points:
        if low <= point.x_m <= high:
            assert CONVENTIONAL_MARGIN[0] <= point.static_margin <= CONVENTIONAL_MARGIN[1]


def test_the_recommendation_lands_in_the_middle_of_that_band(sweep):
    target = 0.5 * (CONVENTIONAL_MARGIN[0] + CONVENTIONAL_MARGIN[1])
    best = sweep.recommended()
    assert best is not None
    assert best.conventional
    assert abs(best.static_margin - target) <= 0.5 * (CONVENTIONAL_MARGIN[1] - target)


def test_holding_cm_q_far_from_the_solved_cg_is_declared(sweep):
    """`Cm_q` and `Iyy` are held at the solved values, so the damping degrades away from the CG the
    solve ran at. The margin stays exact; the damping does not, and the difference has to be said
    rather than discovered."""
    assert any("Cm_q and Iyy are held" in note for note in sweep.notes)


def test_a_result_with_no_neutral_point_recovers_one_rather_than_refusing(derivatives):
    """VSPAERO does not always report `X_np`, but `SM = −Cm_alpha/CL_alpha` gives it about the CG
    the solve ran at - which is the same number by a different route."""
    derivatives.neutral_point_x_m = None
    derivatives.static_margin_reported = None
    sweep = cg_sweep(
        derivatives, MassProperties(mass_kg=2.5, ixx=0.3, iyy=0.4, izz=0.6),
        OperatingCondition(velocity_mps=20.0), [0.28, 0.32],
    )
    assert sweep.neutral_point_x_m == pytest.approx(NEUTRAL_POINT, rel=1e-9)
    assert any("recovered from Cm_alpha" in note for note in sweep.notes)


def test_a_sweep_with_no_reference_chord_says_so_rather_than_dividing_by_zero():
    sweep = cg_sweep(
        StabilityResult(cref_m=0.0), MassProperties(mass_kg=1.0),
        OperatingCondition(), [0.1, 0.2],
    )
    assert sweep.points == []
    assert any("reference chord" in note for note in sweep.notes)
