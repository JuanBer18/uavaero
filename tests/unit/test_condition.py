"""The operating condition, the atmosphere it implies, and the fidelity ladder.

Two properties matter here and both are the kind that fail silently.

The first is that **there is one atmosphere**. Reynolds number appears in three unrelated places —
the reference Reynolds VSPAERO is handed, the local Reynolds each strip uses to pick a section
polar, and the Reynolds a stall check compares Cl_max against. In the previous generation these
were served by a hard-coded density function in the sizing tree and two separate hard-coded
viscosity constants, which is a disagreement waiting to happen. They now all come off one
`Atmosphere`, and the tests below pin its numbers against the constants they replaced.

The second is that **the fidelity rungs are actually different, and ordered**. A ladder whose
rungs quietly resolve to the same mesh is worse than no ladder: it makes screening runs cost what
finalist runs cost while claiming otherwise, and nothing about the results looks wrong.
"""
from __future__ import annotations

import math

import pytest

from uavaero.schema.condition import (
    SEA_LEVEL_DENSITY,
    Atmosphere,
    Fidelity,
    OperatingCondition,
    air_density,
    dynamic_viscosity,
    speed_of_sound,
)

# ════════ the atmosphere ════════════════════════════════════════════════════════════════════


def test_sea_level_reproduces_the_constants_it_replaced():
    """The previous generation hard-coded MU_SEA_LEVEL = 1.789e-5 and SPEED_OF_SOUND = 340.3 next
    to a separate ISA density function. Sutherland's law and the ideal-gas speed of sound have to
    land on those same numbers, or the substitution silently moved every Reynolds number in the
    tool."""
    assert air_density(0.0) == pytest.approx(1.225)
    assert dynamic_viscosity(0.0) == pytest.approx(1.789e-5, rel=2e-3)
    assert speed_of_sound(0.0) == pytest.approx(340.3, rel=1e-3)


def test_density_falls_and_viscosity_falls_with_altitude():
    """Both decrease in the troposphere. Viscosity falling is the less obvious one and it is the
    half that a hard-coded sea-level constant got wrong."""
    assert air_density(2000) < air_density(0)
    assert dynamic_viscosity(2000) < dynamic_viscosity(0)


def test_density_below_sea_level_does_not_extrapolate():
    """A negative altitude is a user typo, not a request for a sub-sea-level ISA. Returning the
    sea-level value keeps a stray minus sign from producing a plausible-looking wrong answer."""
    assert air_density(-500) == SEA_LEVEL_DENSITY


def test_reynolds_of_a_degenerate_strip_is_zero_rather_than_an_exception():
    """The strip integrator walks stations that can legitimately have zero chord at a pointed tip.
    A raise here would turn an ordinary geometry into a solver crash."""
    assert Atmosphere(0).reynolds(20.0, 0.0) == 0.0
    assert Atmosphere(0).reynolds(20.0, -1.0) == 0.0


def test_reynolds_matches_the_definition():
    air = Atmosphere(0)
    expected = air.density * 20.0 * 0.25 / air.viscosity
    assert air.reynolds(20.0, 0.25) == pytest.approx(expected)


def test_mach_is_referenced_to_the_local_speed_of_sound():
    """Not to the sea-level one. At altitude the two differ by several percent, and Mach is what
    decides whether a compressibility correction is even asked for."""
    fast = Atmosphere(3000)
    assert fast.mach(30.0) == pytest.approx(30.0 / speed_of_sound(3000))
    assert fast.mach(30.0) > Atmosphere(0).mach(30.0)


# ════════ the fidelity ladder ═══════════════════════════════════════════════════════════════


def test_every_rung_is_strictly_finer_than_the_one_below():
    """The ordering is the whole contract. If QUICK ever resolved to the same mesh as STANDARD the
    coarse-to-fine pipeline would still run, still report, and cost the same as running everything
    at full fidelity."""
    quick, standard, high = (f.settings for f in (Fidelity.QUICK, Fidelity.STANDARD, Fidelity.HIGH))
    for attribute in ("alpha_points", "wake_iterations", "tess_w", "sect_tess_u"):
        assert getattr(quick, attribute) < getattr(standard, attribute) < getattr(high, attribute), attribute


def test_the_screening_rung_still_meshes_finely_enough_to_be_trusted():
    """QUICK is deliberately not as coarse as VSPAERO will accept. A screening rung that
    occasionally produces a degenerate mesh does not fail — it returns zeros, which an optimiser
    reads as a spectacularly low-drag aircraft and promotes."""
    assert Fidelity.QUICK.settings.tess_w >= 17
    assert Fidelity.QUICK.settings.sect_tess_u >= 6


def test_the_sweep_takes_its_point_count_from_the_rung_unless_told_otherwise():
    assert len(OperatingCondition(fidelity=Fidelity.QUICK).alphas()) == 3
    assert len(OperatingCondition(fidelity=Fidelity.HIGH).alphas()) == 17
    assert len(OperatingCondition(fidelity=Fidelity.QUICK, alpha_points=11).alphas()) == 11


def test_a_sweep_always_has_at_least_two_points():
    """A one-point sweep has no slope, so CL_alpha and the CD0/e fit both become meaningless. The
    floor is here rather than in every consumer."""
    assert len(OperatingCondition(alpha_points=1).alphas()) == 2


# ════════ the condition itself ══════════════════════════════════════════════════════════════


def test_a_single_beta_means_a_symmetric_solve():
    """One sideslip value is the normal case and must not become a degenerate two-point sweep."""
    assert list(OperatingCondition().betas()) == [0.0]


def test_a_beta_range_is_what_the_lateral_derivatives_are_differenced_over():
    """OpenVSP 3.51 has no stability-analysis mode at all — verified against the installed
    package's VSPAEROSweep inputs, which carry Beta{Start,End,Npts} and no StabilityType. So
    Cn_beta and Cl_beta have to come from a finite difference over a beta sweep, and that sweep
    has to be expressible here."""
    condition = OperatingCondition(beta_start_deg=-5.0, beta_end_deg=5.0, beta_points=5)
    assert list(condition.betas()) == [-5.0, -2.5, 0.0, 2.5, 5.0]


def test_an_unknown_scope_is_refused_at_construction():
    """A typo'd scope would otherwise reach the adapter and silently solve the wrong set of
    components — a result that looks entirely normal and answers a different question."""
    with pytest.raises(ValueError, match="scope must be one of"):
        OperatingCondition(scope="whole_plane")


def test_a_condition_survives_a_round_trip():
    """It is half of the cache key and it crosses a process boundary into the solver worker as
    JSON. A field that does not survive serialisation is a field the cache cannot see."""
    original = OperatingCondition(
        velocity_mps=18.0, altitude_m=500.0, fidelity=Fidelity.HIGH, scope="wing_tail",
        thick=True, xyz_ref=(0.3, 0.0, 0.05), control_deflections={"elevator": -3.0},
    )
    assert OperatingCondition.from_dict(original.to_dict()) == original


def test_the_serialised_form_is_json_plain():
    """The worker reads it back through `json.loads`, so an Enum or a tuple surviving into the
    payload is a crash at the process boundary rather than at the call site."""
    import json

    payload = OperatingCondition(fidelity=Fidelity.HIGH).to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["fidelity"] == "high"
    assert isinstance(payload["xyz_ref"], list)


def test_wake_iterations_come_from_the_rung():
    """Threading it through the condition rather than asking every call site to look it up is what
    makes one value at the call site change the whole cost of a solve."""
    assert OperatingCondition(fidelity=Fidelity.QUICK).wake_iterations == 3
    assert OperatingCondition(fidelity=Fidelity.HIGH).wake_iterations == 9


def test_the_atmosphere_follows_the_conditions_altitude():
    condition = OperatingCondition(altitude_m=1500.0)
    assert condition.atmosphere.altitude_m == 1500.0
    assert condition.atmosphere.density == pytest.approx(air_density(1500.0))
    assert condition.mach() == pytest.approx(20.0 / speed_of_sound(1500.0))
    assert not math.isnan(condition.mach())
