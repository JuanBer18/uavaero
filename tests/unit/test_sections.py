"""The 2D section layer: one interface, two solvers, and the hole neither may paper over.

The property this file exists to protect is **never interpolate across a hole**.

XFOIL does not converge everywhere. When it fails between CL 0.6 and CL 1.1, `numpy.interp` will
cheerfully draw a straight line across the gap and return a number that is smooth, plausible, and
entirely invented. That is the most dangerous thing a polar database can do, because the fiction is
indistinguishable from a measurement at the call site and it errs in the direction that makes a bad
airfoil look acceptable. `cd_at_cl` returns None instead, and the strip integrator counts the strip
as missing rather than pretending it integrated it.

The second property: **a section that could not be evaluated is dropped and counted, never
zeroed**. A wing quietly missing a third of its viscous drag reads as a remarkably efficient wing.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.markers import requires_neuralfoil
from uavaero.schema.results import Polar
from uavaero.solvers.sections.base import (
    MAX_CL_GAP,
    SectionLookup,
    SectionSource,
    cd_at_cl,
    cl_max_of,
)
from uavaero.solvers.sections.polar_db import PolarDatabase


def _polar(cl: list[float], cd: list[float]) -> Polar:
    n = len(cl)
    return Polar(
        alpha_deg=np.linspace(-4, 12, n), CL=np.array(cl), CD=np.array(cd),
        CM=np.zeros(n), source="test",
    )


class _StubSource:
    """A source with a hole in it, standing in for XFOIL's convergence behaviour."""

    name = "stub"

    def __init__(self, cl: list[float], cd: list[float], fails: bool = False) -> None:
        self._polar = _polar(cl, cd)
        self.fails = fails
        self.calls = 0

    def polar(self, airfoil, alpha_deg, reynolds, mach=0.0):
        self.calls += 1
        if self.fails:
            raise RuntimeError("XFOIL converged on none of the requested alphas")
        return self._polar


# ════════ the hole ══════════════════════════════════════════════════════════════════════════


def test_a_normal_polar_interpolates():
    polar = _polar([0.0, 0.2, 0.4, 0.6], [0.010, 0.011, 0.013, 0.016])
    assert cd_at_cl(polar, 0.3) == pytest.approx(0.012)


def test_a_gap_wider_than_the_limit_is_refused_rather_than_bridged():
    """The regression. Two converged points either side of a non-converged band are not describing
    the same flow any more — that is what the gap means — and joining them invents a drag value
    across a stall or a separation bubble."""
    holed = _polar([0.0, 0.2, 1.0, 1.2], [0.010, 0.011, 0.030, 0.045])
    assert cd_at_cl(holed, 0.6) is None, "0.2 to 1.0 is an 0.8 gap and must not be crossed"
    assert cd_at_cl(holed, 0.1) == pytest.approx(0.0105), "the intact part still answers"


def test_the_gap_limit_is_the_one_that_is_documented():
    just_under = _polar([0.0, MAX_CL_GAP - 0.01], [0.010, 0.020])
    just_over = _polar([0.0, MAX_CL_GAP + 0.01], [0.010, 0.020])
    assert cd_at_cl(just_under, MAX_CL_GAP / 2) is not None
    assert cd_at_cl(just_over, MAX_CL_GAP / 2) is None


def test_asking_beyond_the_data_is_refused_rather_than_extrapolated():
    """Extrapolating profile drag past the last converged point is how a stalled section comes back
    reporting cruise drag."""
    polar = _polar([0.0, 0.4, 0.8], [0.010, 0.013, 0.020])
    assert cd_at_cl(polar, 1.6) is None
    assert cd_at_cl(polar, -0.9) is None


def test_a_polar_with_one_point_cannot_answer():
    assert cd_at_cl(_polar([0.5], [0.01]), 0.5) is None


def test_cl_max_is_the_highest_value_reached():
    """Taken as the maximum rather than by hunting for a slope reversal: a polar cut short by
    non-convergence has its real Cl_max above the last converged point, and "at least this" is the
    honest reading for a stall margin."""
    assert cl_max_of(_polar([0.0, 0.9, 1.3, 1.1], [0.01, 0.02, 0.04, 0.06])) == pytest.approx(1.3)
    assert cl_max_of(_polar([], [])) is None


# ════════ the lookup ════════════════════════════════════════════════════════════════════════


def test_polars_are_cached_per_airfoil_and_reynolds_bucket():
    """Local chord varies continuously along a tapered panel, so unbucketed Reynolds would miss on
    every single strip and turn one section solve into fifty."""
    source = _StubSource([0.0, 0.5, 1.0], [0.010, 0.012, 0.020])
    lookup = SectionLookup(source)
    for reynolds in (300_000, 301_000, 305_000, 310_000):
        lookup.cd("naca2412", reynolds, 0.5)
    assert source.calls == 1, "all four Reynolds numbers fall in one bucket"


def test_a_source_failure_is_cached_so_it_is_not_retried_per_strip():
    """An airfoil XFOIL cannot handle would otherwise be retried once per strip, turning one
    failure into fifty slow ones."""
    source = _StubSource([], [], fails=True)
    lookup = SectionLookup(source)
    for _ in range(5):
        assert lookup.cd("exotic", 300_000, 0.5) is None
    assert source.calls == 1
    assert any("exotic" in m for m in lookup.misses)


def test_a_refusal_is_recorded_so_the_caller_can_report_it():
    lookup = SectionLookup(_StubSource([0.0, 0.2, 1.0], [0.01, 0.011, 0.03]))
    assert lookup.cd("naca2412", 300_000, 0.6) is None
    assert lookup.misses, "a silent None is how a missing model becomes an agreeing one"


@requires_neuralfoil
def test_neuralfoil_answers_a_reasonable_request():
    from uavaero.solvers.sections.neuralfoil import NeuralFoilSource

    lookup = SectionLookup(NeuralFoilSource())
    cd = lookup.cd("naca2412", 300_000, 0.5)
    assert cd is not None
    assert 0.004 < cd < 0.030, f"a NACA 2412 at Re 3e5 and CL 0.5 should be here, got {cd}"
    assert lookup.cl_max("naca2412", 300_000) > 1.0


@requires_neuralfoil
def test_both_shipped_sources_satisfy_the_interface():
    from uavaero.solvers.sections.neuralfoil import NeuralFoilSource
    from uavaero.solvers.sections.xfoil import XFoilSource

    assert isinstance(NeuralFoilSource(), SectionSource)
    assert isinstance(XFoilSource(), SectionSource)


# ════════ the polar store ═══════════════════════════════════════════════════════════════════


def test_a_stored_polar_comes_back(tmp_path):
    database = PolarDatabase(tmp_path / "polars.db")
    polar = _polar([0.0, 0.5, 1.0], [0.010, 0.012, 0.020])
    database.put("sd7062", 150_000, polar)
    restored = database.get("sd7062", 150_000)
    assert restored is not None
    assert restored.CL == pytest.approx(polar.CL)


def test_a_different_reynolds_is_a_different_row(tmp_path):
    database = PolarDatabase(tmp_path / "polars.db")
    database.put("sd7062", 150_000, _polar([0.0, 1.0], [0.01, 0.02]))
    assert database.get("sd7062", 300_000) is None


def test_a_recorded_failure_is_distinguishable_from_never_having_tried(tmp_path):
    """Rediscovering that XFOIL cannot solve a section costs the same timeout every time, so the
    failure is a fact worth storing — but "we tried and it failed" and "we never asked" have to
    stay tellable apart, or a retry can never be requested deliberately."""
    database = PolarDatabase(tmp_path / "polars.db")
    assert database.known_failure("weird", 50_000) is None

    database.put_failure("weird", 50_000, "did not converge at any alpha")
    assert database.get("weird", 50_000) is None
    assert "converge" in database.known_failure("weird", 50_000)


def test_convergence_state_is_stored_not_inferred(tmp_path):
    """A polar that only half-solved must be identifiable later as a partial answer rather than
    mistaken for a well-behaved section with a narrow envelope."""
    database = PolarDatabase(tmp_path / "polars.db")
    partial = _polar([0.0, 0.5], [0.010, 0.012])
    partial.meta = {"n_requested": 49, "n_converged": 2}
    database.put("fussy", 75_000, partial)

    entry = database.entries("fussy")[0]
    assert entry["n_requested"] == 49
    assert entry["n_converged"] == 2
    assert entry["cl_max"] == pytest.approx(0.5)


def test_the_reynolds_grid_is_tighter_where_low_re_behaviour_changes_fastest():
    """The difference between Re 50k and 75k is a different airfoil; the difference between 750k
    and 1M is a rounding error. The grid has to reflect that or it wastes solves at the top and
    misses the knee at the bottom."""
    from uavaero.solvers.sections.polar_db import STANDARD_RE_GRID

    steps = np.diff(STANDARD_RE_GRID)
    assert steps[0] < steps[-1]
    assert STANDARD_RE_GRID[0] == 50_000
