"""A scope names what goes into the mesh, and it has to stay answerable as geometry changes.

The component builder can delete a tail or a fuselage from a saved aircraft. The operating
condition does not know that happened, so a file authored as `aircraft` and then stripped of its
fuselage becomes one that cannot be solved at all — and the only symptom is a refusal at the moment
Solve is pressed, tens of seconds into a session, which reads as the tool being broken rather than
as one setting that needs changing.

`widest_runnable_scope` is what keeps the two in step. The property worth protecting is that it
**only ever narrows**: promoting a `wing_tail` scope back to `aircraft` because a fuselage
reappeared would silently change the question being asked, and comparing what the tail costs in
drag against a body-free baseline is exactly why scopes exist in the first place.
"""
from __future__ import annotations

import copy

import pytest

from uavaero.geometry import authoring
from uavaero.schema.airframe import default_airframe
from uavaero.schema.condition import SCOPES
from uavaero.solvers.vspaero import missing_for_scope, widest_runnable_scope


@pytest.fixture
def aircraft():
    return default_airframe()


def _without(airframe, *kinds):
    stripped = copy.deepcopy(airframe)
    for kind in kinds:
        authoring.remove_component(stripped, kind)
    return stripped


def test_a_complete_aircraft_keeps_the_widest_scope(aircraft):
    assert widest_runnable_scope(aircraft) == "aircraft"


def test_losing_the_fuselage_drops_the_scope_to_wing_tail(aircraft):
    assert widest_runnable_scope(_without(aircraft, "fuselage")) == "wing_tail"


def test_losing_the_tails_as_well_drops_it_to_the_wing(aircraft):
    assert widest_runnable_scope(_without(aircraft, "fuselage", "htail", "vtail")) == "wing"


def test_one_remaining_surface_still_counts_as_a_tail(aircraft):
    """`wing_tail` needs *a* tail, not both of them - a V-tail aircraft has no separate fin and is
    still perfectly solvable at that scope."""
    assert widest_runnable_scope(_without(aircraft, "fuselage", "vtail")) == "wing_tail"


def test_it_never_widens(aircraft):
    """The load-bearing one. A narrower scope is a question someone chose to ask; restoring the
    geometry does not un-ask it."""
    for scope in SCOPES:
        assert widest_runnable_scope(aircraft, scope) == scope


def test_whatever_it_returns_actually_runs(aircraft):
    """The result is only useful if it is solvable, so that is asserted directly rather than by
    matching a remembered table of names."""
    for stripped in (
        aircraft,
        _without(aircraft, "fuselage"),
        _without(aircraft, "fuselage", "vtail"),
        _without(aircraft, "fuselage", "htail", "vtail"),
    ):
        scope = widest_runnable_scope(stripped)
        assert missing_for_scope(stripped, scope) == "", scope


def test_a_refusal_names_the_scope_that_would_work(aircraft):
    """"Not available" without a way forward reads as the tool refusing rather than as a question
    this airframe cannot answer."""
    assert "wing_tail" in missing_for_scope(_without(aircraft, "fuselage"), "aircraft")
    assert "wing" in missing_for_scope(_without(aircraft, "htail", "vtail"), "wing_tail")
