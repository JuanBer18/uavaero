"""The saved airframe: what survives a save, and what a save must not quietly change.

The regression that matters most here is `test_saving_an_airframe_does_not_change_its_cache_key`.

Floats are rounded to ten decimal places when written, so the file a person opens does not contain
`0.30000000000000004`. That rounding is harmless to the geometry — a tenth of a nanometre — but the
cache key is a hash over the airframe, so for a while the key computed in memory and the key
computed after a reload were different. The consequence was not a slow cache: every recorded solve
in the file compared unequal to its own recomputed key, so reopening an airframe reported all of its
results as stale and the drag comparison came back empty. The fix was to hash the canonical
persisted form, and this test is what stops it regressing.

The second property is that **a stale solve is kept and labelled, never deleted**. Results measured
on a shape you have since changed are still yours; they simply must not be presented as describing
the current aircraft, because a stale polar looks exactly like a fresh one.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from uavaero.geometry.authoring import TailSizing, build
from uavaero.geometry.planform import Planform
from uavaero.schema.condition import Fidelity, OperatingCondition
from uavaero.schema.results import AeroResult, DragBreakdown, DragModel, Polar, ReferenceValues
from uavaero.store import airframe_file as af
from uavaero.store.cache import ResultCache


@pytest.fixture
def aircraft():
    return build(
        "test-aircraft",
        Planform(span=2.40, aspect_ratio=9.60, taper=0.667, sweep_mode="straight_qc",
                 dihedral_deg=3.0, tip_twist_deg=-2.0),
        htail=TailSizing(0.098, 2.50, 0.880, taper=0.75, incidence_deg=-1.0),
        vtail=TailSizing(0.034, 1.85, 0.880, taper=0.70),
    )


def _result(solver="vspaero", fidelity="quick", scope="wing") -> AeroResult:
    drag = DragBreakdown(cd_induced=[0.002, 0.009])
    drag.add(DragModel("vspaero_native", "VSPAERO native", [0.030, 0.038], basis="test"))
    drag.add(DragModel("strip_neuralfoil", "Strip-corrected", [0.026, 0.033], basis="test"))
    return AeroResult(
        polar=Polar(np.array([0.0, 6.0]), np.array([0.20, 0.72]), np.array([0.030, 0.038]),
                    np.zeros(2), source=solver),
        drag=drag, reference=ReferenceValues(0.6, 2.4, 0.2533),
        solver=solver, fidelity=fidelity, scope=scope, elapsed_s=9.5,
        primary_model="strip_neuralfoil",
    )


# ════════ the key must survive a save ═══════════════════════════════════════════════════════


def test_saving_an_airframe_does_not_change_its_cache_key(aircraft, tmp_path):
    """The regression. Floats are rounded on write for readability; the hash has to see the same
    rounded form, or every result in the file is orphaned the moment it is reopened."""
    file = af.create("test", airframe=aircraft)
    before = ResultCache.key(file.airframe, file.condition, "vspaero")
    reloaded = af.load(af.save(file, tmp_path / "a.yaml"))
    after = ResultCache.key(reloaded.airframe, reloaded.condition, "vspaero")
    assert before == after


def test_a_recorded_solve_is_still_current_after_a_reload(aircraft, tmp_path):
    """The user-visible half of the same bug: results must not vanish when you reopen a file."""
    file = af.create("test", airframe=aircraft)
    file.record(_result())
    reloaded = af.load(af.save(file, tmp_path / "a.yaml"))

    assert len(reloaded.solves) == 1
    assert len(reloaded.current_solves()) == 1
    assert reloaded.stale_solves() == []
    assert reloaded.latest("vspaero") is not None


def test_the_geometry_survives_a_save_to_the_precision_the_file_states(aircraft, tmp_path):
    """Ten decimal places on a dimension in metres is a tenth of a nanometre. The test asserts the
    contract rather than exact equality, because exact equality is not what the file promises."""
    file = af.create("test", airframe=aircraft)
    reloaded = af.load(af.save(file, tmp_path / "a.yaml"))
    for before, after in zip(aircraft.wing.sections, reloaded.airframe.wing.sections):
        assert after.chord == pytest.approx(before.chord, abs=1e-10)
        assert after.span == pytest.approx(before.span, abs=1e-10)
        assert after.airfoil == before.airfoil
    assert reloaded.airframe.wing.area() == pytest.approx(aircraft.wing.area(), abs=1e-9)


# ════════ recording solves ══════════════════════════════════════════════════════════════════


def test_re_solving_the_same_question_replaces_rather_than_accumulates(aircraft):
    file = af.create("test", airframe=aircraft)
    file.record(_result())
    file.record(_result())
    assert len(file.solves) == 1


def test_a_different_question_keeps_its_own_entry(aircraft):
    """Several fidelities and scopes have to be readable side by side — that is the comparison."""
    file = af.create("test", airframe=aircraft)
    file.record(_result(fidelity="quick"), OperatingCondition(fidelity=Fidelity.QUICK, scope="wing"))
    file.record(_result(fidelity="high"), OperatingCondition(fidelity=Fidelity.HIGH, scope="wing"))
    file.record(_result(scope="aircraft"), OperatingCondition(fidelity=Fidelity.QUICK, scope="aircraft"))
    assert len(file.solves) == 3


def test_changing_the_aircraft_makes_an_old_solve_stale_without_deleting_it(aircraft):
    """Kept, because deleting someone's results because they nudged a chord would be indefensible.
    Not current, because a stale polar is indistinguishable from a fresh one."""
    file = af.create("test", airframe=aircraft)
    file.record(_result())
    assert len(file.current_solves()) == 1

    file.airframe.wing.sections[1] = dataclasses.replace(
        file.airframe.wing.sections[1], span=1.35
    )

    assert len(file.solves) == 1, "the solve must not be deleted"
    assert file.current_solves() == [], "and must not be reported as current"
    assert len(file.stale_solves()) == 1
    assert file.latest() is None


def test_latest_can_be_filtered_by_solver(aircraft):
    file = af.create("test", airframe=aircraft)
    file.record(_result(solver="vspaero"))
    file.record(_result(solver="avl"))
    assert file.latest("avl").result.solver == "avl"
    assert file.latest("vspaero").result.solver == "vspaero"
    assert file.latest("nonexistent") is None


# ════════ the file itself ═══════════════════════════════════════════════════════════════════


def test_a_file_round_trips_its_results_and_condition(aircraft, tmp_path):
    file = af.create("test", airframe=aircraft,
                     condition=OperatingCondition(velocity_mps=24.0, fidelity=Fidelity.HIGH,
                                                  scope="aircraft", thick=True))
    file.record(_result())
    reloaded = af.load(af.save(file, tmp_path / "a.yaml"))

    assert reloaded.condition == file.condition
    solve = reloaded.solves[0]
    assert solve.result.primary_model == "strip_neuralfoil"
    assert list(solve.result.drag.models) == ["vspaero_native", "strip_neuralfoil"]
    assert solve.result.drag.cd("strip_neuralfoil") == [0.026, 0.033]


def test_numpy_values_do_not_break_a_save(aircraft, tmp_path):
    """A result arrives full of `np.float64`, which PyYAML's safe dumper cannot represent — it
    raises part-way through and loses the whole file rather than one field."""
    file = af.create("test", airframe=aircraft)
    result = _result()
    result.elapsed_s = np.float64(12.5)
    result.reynolds = np.float64(346877.0)
    file.record(result)

    reloaded = af.load(af.save(file, tmp_path / "a.yaml"))
    assert reloaded.solves[0].result.elapsed_s == pytest.approx(12.5)
    assert type(reloaded.solves[0].result.reynolds) is float


def test_the_saved_file_is_readable_by_a_human(aircraft, tmp_path):
    """It is hand-editable and version-controlled, which is the reason floats are rounded at all."""
    text = af.save(af.create("test", airframe=aircraft), tmp_path / "a.yaml").read_text("utf-8")
    assert "schema_version: 1" in text
    assert "0.30000000000000004" not in text
    assert "!!python" not in text, "nothing may serialise as an arbitrary python object"


def test_a_file_from_a_newer_build_is_refused_rather_than_half_read(aircraft, tmp_path):
    """Opening it would work and saving would silently discard whatever the newer version added."""
    file = af.create("test", airframe=aircraft)
    payload = file.to_dict()
    payload["schema_version"] = af.SCHEMA_VERSION + 1
    with pytest.raises(af.AirframeFileError, match="newer version"):
        af.AirframeFile.from_dict(payload)


def test_a_missing_file_says_where_it_looked(tmp_path):
    with pytest.raises(af.AirframeFileError, match="No airframe at"):
        af.load(tmp_path / "nothing.yaml")


def test_an_empty_file_is_reported_rather_than_loaded_as_a_blank_aircraft(tmp_path):
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    with pytest.raises(af.AirframeFileError, match="empty"):
        af.load(tmp_path / "empty.yaml")


def test_the_path_is_not_stored_in_the_file(aircraft, tmp_path):
    """A file that carried its own path would claim the wrong one as soon as it was copied."""
    file = af.create("test", airframe=aircraft)
    af.save(file, tmp_path / "a.yaml")
    assert "path" not in file.to_dict()
    assert file.path == tmp_path / "a.yaml"


@pytest.mark.parametrize(
    "name,expected",
    [("Skywalker X-8", "skywalker-x-8"), ("  spaced  out  ", "spaced-out"),
     ("!!!", "airframe"), ("Zephyr/Mk2", "zephyr-mk2")],
)
def test_names_become_filenames_without_rejecting_anything(name, expected):
    """Naming an aircraft should not be a validation exercise."""
    assert af.slug(name) == expected


def test_a_new_airframe_is_a_complete_aircraft(tmp_path):
    """Not a bare wing. An aircraft you cannot solve every scope of is a worse starting point than
    one you can edit."""
    file = af.create("fresh")
    assert file.airframe.htail is not None
    assert file.airframe.vtail is not None
    assert file.airframe.fuselages


def test_the_library_lists_what_was_saved(aircraft, tmp_path):
    af.save(af.create("Alpha", airframe=aircraft), af.path_for("Alpha", tmp_path))
    af.save(af.create("Beta", airframe=aircraft), af.path_for("Beta", tmp_path))
    assert af.names(tmp_path) == ["alpha", "beta"]
    af.delete("Alpha", tmp_path)
    assert af.names(tmp_path) == ["beta"]


def test_listing_an_absent_library_is_empty_rather_than_an_error(tmp_path):
    assert af.names(tmp_path / "not-created-yet") == []
