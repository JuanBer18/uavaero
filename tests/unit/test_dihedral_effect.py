"""`Cl_beta` and `CY_beta` have to agree about where the side force is.

A rolling moment from sideslip is a side force times a moment arm, so a derivative set states its
own geometry: `z = Cl_beta / CY_beta · b` is the height that side force must act at. That makes the
pair checkable against the aircraft, and it is the only check available — there is no third code
here to vote, and both values look entirely reasonable on their own.

Measured on the reference aircraft, VSPAERO and AVL disagreed on `Cl_beta` by 28x (−0.0046 against
−0.1273) and the arithmetic said why: VSPAERO's pair puts the side force at z = 0.006 m, below a
fin whose root is at 0.050 m. Both a fin above the CG and positive wing dihedral push that height
*up*, so nothing available can produce a value below the root. AVL's lands at 0.165 m, inside the
fin. That is a rejection on geometry, not a preference between codes.

**The tolerance is deliberately generous**, and these tests pin that too. A fuselage carries side
force near z = 0 and drags the weighted average down, so the height is not the fin's own centre and
a value somewhat below the root is legitimate. Only a value that is essentially zero is called
impossible.

**It is not a claim that AVL is the better code.** On `Cn_beta` VSPAERO is the one to trust: it has
a fuselage in the solve and AVL is a vortex lattice with no bodies at all. The substitution happens
only when one value fails a physical check and the other passes the same one.
"""
from __future__ import annotations

import copy
import dataclasses

import pytest

from uavaero.schema.airframe import default_airframe
from uavaero.schema.stability import StabilityResult
from uavaero.solvers.stability import fin_height_band, implausible_dihedral_effect

#: (CY_beta, Cl_beta) as the two codes reported them on 2026-09-02, and the span they were
#: measured at. The span matters: `z = Cl_beta / CY_beta · b`, so the same coefficient pair means a
#: different height on a different aircraft, and reading these against a 2.4 m reference wing makes
#: an impossible pair look merely small. That mistake is why this constant carries its own bref.
VSPAERO_PAIR = (-0.4481, -0.0046)
AVL_PAIR = (-0.4635, -0.1273)
MEASURED_BREF_M = 0.6


@pytest.fixture
def aircraft():
    """A small airframe whose fin sits well above the CG and whose wing has no dihedral, so the
    fin is the only thing that can produce a rolling moment from sideslip."""
    air = default_airframe()
    air.vtail.origin = (air.vtail.origin[0], 0.0, 0.05)
    air.wing.sections = [
        dataclasses.replace(section, dihedral_deg=0.0) for section in air.wing.sections
    ]
    return air


def _result(aircraft, pair, bref_m: float | None = None) -> StabilityResult:
    cy, cl = pair
    return StabilityResult(
        CY_beta=cy, Cl_beta=cl,
        bref_m=MEASURED_BREF_M if bref_m is None else bref_m,
    )


# ════════ the arithmetic states the geometry ════════════════════════════════════════════════


def test_the_pair_implies_a_height(aircraft):
    """`z = Cl_beta / CY_beta · b`, which is the whole basis of the check."""
    result = _result(aircraft, AVL_PAIR)
    expected = AVL_PAIR[1] / AVL_PAIR[0] * MEASURED_BREF_M
    assert result.side_force_height_m == pytest.approx(expected, rel=1e-12)


def test_a_set_with_no_side_force_has_no_implied_height(aircraft):
    """Dividing by a zero `CY_beta` would produce an infinity that reads like a very tall fin."""
    assert StabilityResult(CY_beta=0.0, Cl_beta=-0.1, bref_m=2.4).side_force_height_m is None


def test_the_fin_band_is_measured_above_the_cg(aircraft):
    root, tip = fin_height_band(aircraft)
    assert root == pytest.approx(0.05)
    assert tip == pytest.approx(0.05 + aircraft.vtail.span())


# ════════ the rejection, on the measured numbers ════════════════════════════════════════════


def test_the_measured_vspaero_pair_is_rejected(aircraft):
    """The case this exists for. Not "disagrees with AVL" - impossible on its own terms."""
    complaint = implausible_dihedral_effect(_result(aircraft, VSPAERO_PAIR), aircraft)
    assert complaint
    assert "below the fin root" in complaint
    assert "0.0046" in complaint and "0.4481" in complaint, "say which numbers, not just that"


def test_the_measured_avl_pair_is_accepted(aircraft):
    assert implausible_dihedral_effect(_result(aircraft, AVL_PAIR), aircraft) == ""


def test_a_value_somewhat_below_the_fin_root_is_allowed(aircraft):
    """The fuselage genuinely does pull the average height down. A check that rejected every value
    below the fin's own centre would fire on healthy solves and be turned off."""
    root, _ = fin_height_band(aircraft)
    bref = MEASURED_BREF_M
    cy = -0.45
    just_under = StabilityResult(
        CY_beta=cy, Cl_beta=cy * (0.6 * root) / bref, bref_m=bref,
    )
    assert just_under.side_force_height_m < root
    assert implausible_dihedral_effect(just_under, aircraft) == ""


# ════════ it declines to judge what it cannot ═══════════════════════════════════════════════


def test_an_aircraft_with_no_fin_is_not_judged(aircraft):
    """With no fin there is no floor to be below, and a flying wing's `Cl_beta` comes from dihedral
    alone."""
    tailless = copy.deepcopy(aircraft)
    tailless.vtail = None
    assert implausible_dihedral_effect(_result(tailless, VSPAERO_PAIR), tailless) == ""


def test_an_anhedral_wing_is_not_judged(aircraft):
    """Anhedral rolls the other way and legitimately cancels the fin, so a near-zero `Cl_beta` is
    exactly what it should produce. The check's whole argument is that nothing *can* push the
    height down - that stops being true here."""
    anhedral = copy.deepcopy(aircraft)
    anhedral.wing.sections = [
        dataclasses.replace(section, dihedral_deg=-5.0) for section in anhedral.wing.sections
    ]
    assert implausible_dihedral_effect(_result(anhedral, VSPAERO_PAIR), anhedral) == ""


def test_a_fin_at_or_below_the_cg_is_not_judged(aircraft):
    """A ventral fin produces the opposite sign, so the floor argument does not apply."""
    ventral = copy.deepcopy(aircraft)
    ventral.vtail.origin = (ventral.vtail.origin[0], 0.0, -0.10)
    assert implausible_dihedral_effect(_result(ventral, VSPAERO_PAIR), ventral) == ""


def test_a_set_with_no_reference_span_is_not_judged(aircraft):
    assert implausible_dihedral_effect(
        StabilityResult(CY_beta=-0.45, Cl_beta=-0.004, bref_m=0.0), aircraft
    ) == ""
