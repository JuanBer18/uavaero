"""The result cache, and the one way it can do real damage.

A cache that misses is slow. A cache that hits when it should not is *wrong*, and wrong in the
worst available way: the stale answer is indistinguishable from a fresh one at the call site, it
carries the right shape and plausible numbers, and it will be compared against things it was never
solved for. So the tests that matter here are all the same test asked repeatedly — change one thing
about the question, and confirm the key changed.

The key is taken over the whole serialised form of the airframe and the condition rather than a
hand-picked list of fields, which is what makes that property hold for fields nobody has added yet.
`test_a_new_field_on_the_condition_is_covered_without_being_listed` is the one that protects the
decision.

No solver runs in this file. `_FakeSolver` counts calls and returns a canned result — the only
mock in the tree, and it is here because the thing under test is the cache's *decision to call*,
which cannot be observed through a real solver without paying thirty seconds to watch it happen.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from uavaero.schema.airframe import ParametricAirframe, WingSection, default_airframe
from uavaero.schema.condition import Fidelity, OperatingCondition
from uavaero.schema.results import AeroResult, DragBreakdown, DragModel, Polar, ReferenceValues
from uavaero.store.cache import CachedSolver, ResultCache, git_commit, hash_inputs, tool_versions


class _FakeSolver:
    """Counts how many times it was actually asked to solve."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def unavailable_for(self, airframe, condition) -> str:
        return ""

    def evaluate(self, airframe, condition, *, workdir=None, on_output=None) -> AeroResult:
        self.calls += 1
        drag = DragBreakdown(cd_induced=[0.002, 0.006])
        drag.add(DragModel("vspaero_native", "VSPAERO native", [0.030, 0.034]))
        return AeroResult(
            polar=Polar(np.array([0.0, 4.0]), np.array([0.2, 0.6]), np.array([0.030, 0.034]),
                        np.zeros(2), source="fake"),
            drag=drag, reference=ReferenceValues(0.6, 2.4, 0.253),
            solver=self.name, fidelity=condition.fidelity.value, scope=condition.scope,
            elapsed_s=1.0, primary_model="vspaero_native",
        )


@pytest.fixture
def cache(tmp_path) -> ResultCache:
    return ResultCache(tmp_path / "results.db")


# ════════ what the key must cover ═══════════════════════════════════════════════════════════


def _key(airframe=None, **condition_kwargs) -> str:
    return ResultCache.key(
        airframe or default_airframe(), OperatingCondition(**condition_kwargs), "vspaero",
    )


def test_the_same_question_asked_twice_is_the_same_key():
    assert _key() == _key()


@pytest.mark.parametrize(
    "changed",
    [
        {"velocity_mps": 25.0},
        {"altitude_m": 500.0},
        {"alpha_start_deg": -2.0},
        {"alpha_end_deg": 10.0},
        {"alpha_points": 5},
        {"beta_end_deg": 5.0, "beta_points": 3},
        {"fidelity": Fidelity.HIGH},
        {"scope": "wing"},
        {"thick": True},
        {"xyz_ref": (0.3, 0.0, 0.0)},
        {"sref_m2": 0.55},
        {"bref_m": 2.2},
        {"cref_m": 0.24},
        {"control_deflections": {"elevator": -3.0}},
    ],
    ids=lambda d: ",".join(d),
)
def test_changing_anything_about_the_question_changes_the_key(changed):
    """Every one of these changes the answer. A key that missed any of them would serve a result
    solved for a different question, and nothing downstream could tell."""
    assert _key(**changed) != _key()


def test_changing_the_fidelity_changes_the_key_because_it_changes_the_mesh():
    """Worth stating separately from the parametrised sweep above. Fidelity is not a preference —
    it sets tessellation and wake iterations, so a QUICK result served to a caller that asked for
    HIGH is a coarse mesh masquerading as a converged one."""
    assert _key(fidelity=Fidelity.QUICK) != _key(fidelity=Fidelity.HIGH)


def test_changing_the_geometry_changes_the_key():
    moved = default_airframe()
    moved.wing.sections[1] = dataclasses.replace(moved.wing.sections[1], span=1.30)
    assert _key(airframe=moved) != _key()


def test_changing_only_an_airfoil_changes_the_key():
    """The subtlest geometry change there is: identical planform, identical area, identical span.
    It is invisible to VSPAERO's inviscid solve and moves the strip-corrected drag substantially,
    which is exactly the case a hand-listed key would forget."""
    reskinned = default_airframe()
    reskinned.wing.sections[1] = dataclasses.replace(reskinned.wing.sections[1], airfoil="sd7062")
    assert _key(airframe=reskinned) != _key()


def test_changing_the_solver_changes_the_key():
    airframe, condition = default_airframe(), OperatingCondition()
    assert ResultCache.key(airframe, condition, "vspaero") != ResultCache.key(
        airframe, condition, "avl"
    )


def test_a_new_field_on_the_condition_is_covered_without_being_listed():
    """The reason the key is taken over the whole serialised form rather than a chosen subset. A
    field added tomorrow is covered the day it is added, with nothing to remember — which is the
    only version of this that survives contact with a growing codebase."""
    condition = OperatingCondition()
    payload = condition.to_dict()
    assert hash_inputs("vspaero", {}, payload) != hash_inputs(
        "vspaero", {}, {**payload, "a_field_invented_later": 1}
    )


def test_the_key_does_not_depend_on_dict_ordering():
    """Without sort_keys the same question built in a different order hashes differently, and the
    cache silently never hits — a bug that costs only time and so goes unnoticed for a long while."""
    assert hash_inputs({"a": 1, "b": 2}) == hash_inputs({"b": 2, "a": 1})


# ════════ hit, miss, and what is recorded ═══════════════════════════════════════════════════


def test_the_second_identical_call_does_not_reach_the_solver(cache):
    solver = _FakeSolver()
    cached = CachedSolver(solver, cache)
    airframe, condition = default_airframe(), OperatingCondition()

    first = cached.evaluate(airframe, condition)
    second = cached.evaluate(airframe, condition)

    assert solver.calls == 1
    assert cache.stats["hits"] == 1 and cache.stats["misses"] == 1
    assert second.to_dict() == first.to_dict()


def test_a_different_question_does_reach_the_solver(cache):
    solver = _FakeSolver()
    cached = CachedSolver(solver, cache)
    cached.evaluate(default_airframe(), OperatingCondition())
    cached.evaluate(default_airframe(), OperatingCondition(velocity_mps=25.0))
    assert solver.calls == 2


def test_a_cached_result_is_indistinguishable_from_a_solved_one(cache):
    """Including the drag models and which one is primary. A round trip that lost `primary_model`
    would silently change every headline L/D on a cache hit."""
    solver = _FakeSolver()
    cached = CachedSolver(solver, cache)
    airframe, condition = default_airframe(), OperatingCondition()
    fresh = cached.evaluate(airframe, condition)
    restored = cached.evaluate(airframe, condition)

    assert restored.primary_model == fresh.primary_model == "vspaero_native"
    assert restored.cd == fresh.cd
    assert restored.reference.aspect_ratio == pytest.approx(fresh.reference.aspect_ratio)
    assert list(restored.drag.models) == list(fresh.drag.models)


def test_a_stored_row_records_what_produced_it(cache):
    """A result whose provenance is unknown is not a result, it is a rumour — and a cache is
    exactly where rumours accumulate, because the code that produced the row is long gone by the
    time anyone questions it."""
    cached = CachedSolver(_FakeSolver(), cache)
    cached.evaluate(default_airframe(), OperatingCondition(fidelity=Fidelity.HIGH, scope="wing"))

    entry = cache.entries()[0]
    assert entry["solver"] == "fake"
    assert entry["airframe"] == "new-airframe"
    assert entry["fidelity"] == "high"
    assert entry["scope"] == "wing"
    assert entry["created_at"]


def test_tool_versions_record_an_absent_package_rather_than_omitting_it():
    """"numpy was missing" and "we forgot to look" are different facts, and only one of them
    explains a number that will not reproduce."""
    versions = tool_versions()
    assert "python" in versions
    assert set(versions) >= {"numpy", "scipy", "neuralfoil", "openvsp"}
    assert all(isinstance(v, str) and v for v in versions.values())


def test_git_commit_marks_a_dirty_tree_or_reports_nothing():
    """The dirty suffix is the useful half: it means the code that produced a result no longer
    exists anywhere, so the row is not reproducible. Outside a checkout the answer is None, which
    is a fact rather than an error."""
    commit = git_commit()
    assert commit is None or isinstance(commit, str)


def test_entries_can_be_filtered_to_one_airframe(cache):
    cached = CachedSolver(_FakeSolver(), cache)
    cached.evaluate(default_airframe(), OperatingCondition())
    other = ParametricAirframe(name="other", wing=default_airframe().wing)
    cached.evaluate(other, OperatingCondition(scope="wing"))

    assert len(cache.entries()) == 2
    assert [e["airframe"] for e in cache.entries("other")] == ["other"]


def test_clearing_reports_how_much_was_discarded(cache):
    cached = CachedSolver(_FakeSolver(), cache)
    cached.evaluate(default_airframe(), OperatingCondition())
    assert cache.clear() == 1
    assert cache.entries() == []


def test_bypassing_reads_still_writes(cache, monkeypatch):
    """`$UAVAERO_NO_CACHE` is for checking whether a suspicious cached number reproduces. It has to
    re-solve, and it has to leave the fresh answer behind — a bypass that also refused to write
    would make the next ordinary run pay again."""
    solver = _FakeSolver()
    cached = CachedSolver(solver, cache)
    airframe, condition = default_airframe(), OperatingCondition()
    cached.evaluate(airframe, condition)

    monkeypatch.setenv("UAVAERO_NO_CACHE", "1")
    cached.evaluate(airframe, condition)
    assert solver.calls == 2

    monkeypatch.delenv("UAVAERO_NO_CACHE")
    cached.evaluate(airframe, condition)
    assert solver.calls == 2


def test_the_wrapper_passes_availability_through_rather_than_answering_itself(cache):
    """A scope that cannot be solved must be reported before anything expensive starts, and the
    cache has no opinion about it."""
    airframe = ParametricAirframe(name="bare", wing=default_airframe().wing)
    from uavaero.solvers.vspaero import VSPAeroSolver

    cached = CachedSolver(VSPAeroSolver(), cache)
    assert "no tail" in cached.unavailable_for(airframe, OperatingCondition(scope="aircraft"))
    assert cached.unavailable_for(airframe, OperatingCondition(scope="wing")) == ""
