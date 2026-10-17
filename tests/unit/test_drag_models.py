"""The three drag models, the stall check, and the guard on results that cannot be right.

Two properties here are worth more than the rest.

**A healthy mesh is not a healthy solve.** VSPAERO's thick/panel treatment on the conventional
reference aircraft returns an induced drag of −0.14 while reporting 1116 nodes and 28 Kutta nodes —
every mesh diagnostic green, and an answer that says the wing produces thrust by making lift. A
human reading a table notices. An optimiser maximising L/D notices too, and promotes it. So
`VspAeroResult.implausible` is checked in the worker beside the Kutta test.

**Model B is not a total until induced drag is added to it.** OpenVSP's parasite buildup carries no
lift dependence at all. Reported raw it would sit below the other two at every angle and look like
a different quantity, because it is one.
"""
from __future__ import annotations

import numpy as np
import pytest

from uavaero.aero import stall
from uavaero.aero.strips import StripStation
from uavaero.schema.condition import OperatingCondition
from uavaero.schema.results import MeshDiagnostics, Polar
from uavaero.solvers.parasite import ComponentDrag, ParasiteResult, sanity_check
from uavaero.solvers.vspaero import VspAeroResult


def _result(cd_induced: list[float], cd_total: list[float]) -> VspAeroResult:
    n = len(cd_total)
    return VspAeroResult(
        polar=Polar(np.linspace(-4, 12, n), np.linspace(-0.2, 1.2, n), np.array(cd_total),
                    np.zeros(n), source="vspaero"),
        cd_induced=cd_induced,
        mesh=MeshDiagnostics(nodes=1116, loops=2000, kutta_nodes=28, surfaces=4),
        sref_m2=0.6, bref_m=2.4, cref_m=0.2533,
    )


# ════════ the solve that cannot be right ════════════════════════════════════════════════════


def test_negative_induced_drag_is_refused_however_healthy_the_mesh_is():
    """The measured failure, with the measured numbers. The mesh here is the real one from the
    thick solve that produced −0.1396: every diagnostic passes."""
    result = _result([-0.1396, -0.1318, 0.0027], [-0.1302, -0.1206, 0.022])
    assert result.mesh.ok, "the mesh genuinely looks fine, which is the whole problem"
    complaint = result.implausible()
    assert complaint
    assert "negative" in complaint


def test_a_healthy_solve_is_not_refused():
    """The thin solve of the same aircraft, with its real numbers."""
    result = _result([-0.0009, 0.0063, 0.0276], [0.0093, 0.0179, 0.047])
    assert result.implausible() == ""


def test_numerical_noise_near_zero_lift_is_tolerated():
    """A panel solve wobbles either side of zero at the bottom of a sweep. Rejecting −0.001 would
    fail every healthy result; the case this guard exists for is two orders of magnitude larger."""
    assert _result([-0.001, 0.006, 0.028], [0.009, 0.018, 0.047]).implausible() == ""


def test_non_positive_total_drag_is_refused():
    assert "non-positive" in _result([0.001, 0.006], [0.0, 0.018]).implausible()


# ════════ the parasite buildup ══════════════════════════════════════════════════════════════


def _parasite(wing_reynolds: float) -> ParasiteResult:
    return ParasiteResult(
        cd_parasite=0.0141,
        components=[ComponentDrag("wing", 1.2278, 0.2533, wing_reynolds, 1.25, 0.0136, 1.0)],
        sref_m2=0.6, velocity_mps=20.0, altitude_m=0.0,
    )


def test_the_unit_trap_is_caught_by_checking_reynolds():
    """The trap is invisible in the drag coefficient — every unit setting produces a plausible CD.
    It is glaring in Reynolds, which can be computed independently, so that is what gets checked.

    Measured on the reference aircraft at 20 m/s: imperial default gives Re 32,220 and CD 0.0325;
    PD_UNITS_METRIC (centimetres, despite the name) gives Re 3,468 and CD 0.0605; LEN_M gives
    Re 346,817 and CD 0.0191. A factor of three in the answer, from a setting nobody would check.
    """
    condition = OperatingCondition(velocity_mps=20.0)
    assert sanity_check(_parasite(346_817), condition, 0.2533) == ""

    for wrong in (32_220, 3_468):
        complaint = sanity_check(_parasite(wrong), condition, 0.2533)
        assert "length unit" in complaint, wrong


def test_a_buildup_with_no_components_is_reported_rather_than_trusted():
    empty = ParasiteResult(cd_parasite=0.0, sref_m2=0.6)
    assert sanity_check(empty, OperatingCondition(), 0.25)


def test_wetted_area_ratio_is_reported_for_sanity():
    """Total wetted over reference. A bare wing sits near 2; a conventional light aircraft near 4.
    A number far from those usually means a component was double-counted or missed."""
    assert _parasite(346_817).wetted_area_ratio == pytest.approx(1.2278 / 0.6, rel=1e-6)


# ════════ the stall check ═══════════════════════════════════════════════════════════════════


class _Sections:
    """A section source with a known Cl_max, so the margin arithmetic is checkable."""

    name = "stub"

    def __init__(self, cl_max: float) -> None:
        self._cl_max = cl_max

    def cl_max(self, airfoil: str, reynolds: float) -> float:
        return self._cl_max


def _stations(cl_locals: list[float]) -> list[StripStation]:
    return [
        StripStation(y_m=0.2 * (i + 1), chord_m=0.25, width_m=0.2, reynolds=300_000,
                     cl_local=cl, airfoil="naca2412", cd_profile=0.01)
        for i, cl in enumerate(cl_locals)
    ]


def test_a_station_over_its_sections_limit_is_flagged():
    report = stall.check(_stations([0.6, 0.9, 1.4]), _Sections(1.2), alpha_deg=12.0)
    assert not report.ok
    assert len(report.exceeded) == 1
    assert report.worst.cl_local == pytest.approx(1.4)
    assert report.worst.margin == pytest.approx(-0.2)


def test_the_report_says_where_because_root_and_tip_stall_are_different_aircraft():
    """A root stall is docile and gives warning; a tip stall takes the ailerons with it and rolls.
    "This wing stalls" is not an actionable sentence."""
    summary = stall.check(_stations([0.6, 0.8, 1.5]), _Sections(1.2), alpha_deg=12.0).summary()
    assert "100% semispan" in summary
    assert "1.50" in summary and "1.20" in summary


def test_a_wing_with_margin_everywhere_is_ok():
    report = stall.check(_stations([0.4, 0.5, 0.6]), _Sections(1.3))
    assert report.ok
    assert report.exceeded == []
    assert report.worst.margin == pytest.approx(1.3 - 0.6)


def test_a_wing_close_to_the_limit_is_mentioned_before_it_goes_over():
    """Stall margin collapses quickly with incidence, so a wing at 97% of Cl_max at cruise has no
    reserve — and it is not "exceeded", so nothing else would say so."""
    summary = stall.check(_stations([0.5, 0.8, 1.28]), _Sections(1.32)).summary()
    assert "Little stall margin" in summary


def test_a_section_with_no_data_leaves_the_margin_unknown_rather_than_passing():
    """An unrated station must not read as a station with margin. Silence and safety look the same
    in a boolean."""

    class _Blind:
        name = "blind"

        def cl_max(self, airfoil, reynolds):
            return None

    report = stall.check(_stations([0.6, 1.9]), _Blind())
    assert report.worst is None
    assert "unknown" in report.summary()
    assert report.ok, "nothing was measured, so nothing can be claimed to be exceeded"


# ════════ which components the buildup actually runs over ═══════════════════════════════════


class _FakeVsp:
    """Just enough OpenVSP to exercise the set arithmetic without a solver."""

    SET_FIRST_USER = 3

    def __init__(self, geoms: list[str]) -> None:
        self._geoms = geoms
        self.flags: dict[tuple[str, int], bool] = {}
        self.names: dict[int, str] = {}

    def FindGeoms(self):  # noqa: N802 - matching OpenVSP's own casing
        return list(self._geoms)

    def SetSetName(self, index, name):  # noqa: N802
        self.names[index] = name

    def SetSetFlag(self, geom_id, index, flag):  # noqa: N802
        self.flags[(geom_id, index)] = flag


def test_the_buildup_runs_over_the_components_not_over_everything_in_the_model():
    """`VSPAEROComputeGeometry` leaves an `NGonMesh` behind and it lands in `SET_ALL`. Handing
    `SET_ALL` to ParasiteDrag hands it a mesh artefact it cannot build a wetted area for, which it
    reports as `Warning: Geom ID ... not included in Parasite Drag calculation`.

    Every real component was still included, so the numbers were right — but the analysis was being
    asked the wrong question, and the warning went unseen for two phases because nothing displayed
    the solver's output. Naming the components also stops the buildup silently acquiring whatever
    a future analysis leaves lying in the model.
    """
    from uavaero.solvers.parasite import component_set

    ours = ["WING000000", "HTAIL00000", "VTAIL00000"]
    mesh_left_behind = "MESHNGON00"
    vsp = _FakeVsp(ours + [mesh_left_behind])

    index = component_set(vsp, ours)

    assert index == _FakeVsp.SET_FIRST_USER
    assert all(vsp.flags[(gid, index)] for gid in ours)
    assert vsp.flags[(mesh_left_behind, index)] is False


def test_the_component_set_is_named_so_it_is_identifiable_in_the_model():
    from uavaero.solvers.parasite import COMPONENT_SET_NAME, component_set

    vsp = _FakeVsp(["A"])
    component_set(vsp, ["A"])
    assert vsp.names[_FakeVsp.SET_FIRST_USER] == COMPONENT_SET_NAME


def test_the_set_is_rebuilt_rather_than_added_to():
    """A stale membership from a previous solve would quietly change the next buildup — and a
    parasite drag that drifts between runs of the same aircraft is very hard to attribute."""
    from uavaero.solvers.parasite import component_set

    vsp = _FakeVsp(["A", "B"])
    component_set(vsp, ["A", "B"])
    component_set(vsp, ["A"])
    assert vsp.flags[("B", _FakeVsp.SET_FIRST_USER)] is False
