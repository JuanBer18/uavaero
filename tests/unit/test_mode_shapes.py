"""Turning an eigenvector into motion, so a mode can be watched rather than read.

The property this protects is that **phase survives**. `Mode.participation` keeps the eigenvector's
magnitudes, which is enough to name a mode and not enough to draw one: the characteristic thing
about Dutch roll is that the roll leads the yaw by roughly a quarter cycle, and a magnitude-only
description cannot express "leads" at all. Two modes with identical participation and opposite
phase look the same in the table and completely different in the air.

The known answer used here is exactly that quarter-cycle lead. It falls out of the arithmetic
rather than being asserted into it, so if the phase were ever dropped - by taking `abs()` a step
too early, say - this test fails and the mode table does not.

The second property is that **the decay is shown**. Animating at constant amplitude makes a
divergent spiral indistinguishable from a well-damped roll subsidence, which is the one distinction
somebody watching an animation most needs.
"""
from __future__ import annotations

import math

import pytest

from uavaero.dynamics import modes as mode_analysis
from uavaero.dynamics import shapes, statespace
from uavaero.schema.condition import OperatingCondition
from uavaero.schema.mass import MassProperties
from uavaero.schema.stability import StabilityResult


@pytest.fixture
def flight():
    derivatives = StabilityResult(
        x_cg_m=0.30, cref_m=0.25, sref_m2=0.6, bref_m=2.4,
        CL=0.45, CL_alpha=5.0, CD=0.03, Cm_alpha=-1.0, Cm_q=-12.0,
        CY_beta=-0.4, Cl_beta=-0.12, Cn_beta=0.08,
        Cl_p=-0.5, Cn_r=-0.1, Cl_r=0.1, Cn_p=-0.03,
        neutral_point_x_m=0.35, alpha_deg=3.0,
    )
    mass = MassProperties(mass_kg=2.5, ixx=0.3, iyy=0.4, izz=0.6)
    condition = OperatingCondition(velocity_mps=20.0)
    trim = statespace.trim_level_flight(derivatives, mass, condition)
    longitudinal = mode_analysis.analyse(
        statespace.longitudinal_matrix(derivatives, mass, trim), ("u", "w", "q", "theta")
    )
    lateral = mode_analysis.analyse(
        statespace.lateral_matrix(derivatives, mass, trim), ("v", "p", "r", "phi"), lateral=True
    )
    return longitudinal + lateral


def _named(collection, name):
    return next((item for item in collection if item.name == name), None)


# ════════ the eigenvector survives ══════════════════════════════════════════════════════════


def test_every_mode_keeps_its_eigenvector(flight):
    """Participation is the magnitudes normalised. Keeping only that is what made the phase
    unavailable, and phase is the whole difference between a mode named and a mode drawn."""
    for mode in flight:
        assert mode.vector, mode.name
        assert set(mode.vector) == set(mode.states)


# ════════ the known answer ══════════════════════════════════════════════════════════════════


def test_dutch_roll_has_the_roll_leading_the_yaw_by_about_a_quarter_cycle(flight):
    """The textbook signature, and the reason to animate at all: the aircraft wallows because the
    two axes are out of step, not because either is large."""
    dutch = _named(shapes.shapes_for(flight, samples=240), "dutch roll")
    assert dutch is not None
    assert "leading" in dutch.caption

    roll_peak = max(abs(p.roll_deg) for p in dutch.poses)
    yaw_peak = max(abs(p.yaw_deg) for p in dutch.poses)
    assert roll_peak > 0 and yaw_peak > 0, "both axes must move in Dutch roll"

    # The instants of peak roll and peak yaw are separated by a quarter period, give or take.
    roll_at = max(dutch.poses, key=lambda p: abs(p.roll_deg)).time_s
    yaw_at = max(dutch.poses, key=lambda p: abs(p.yaw_deg)).time_s
    quarter = dutch.period_s / 4.0
    assert abs(abs(roll_at - yaw_at) - quarter) < 0.6 * quarter


def test_the_short_period_is_a_pitching_mode_and_the_roll_modes_are_not(flight):
    """A sanity check that the axis mapping did not get transposed, which would be invisible in
    the numbers and obvious the moment anybody watched it."""
    drawn = {s.name: s for s in shapes.shapes_for(flight, samples=90)}

    short = drawn["short period"]
    assert max(abs(p.pitch_deg) for p in short.poses) > max(
        abs(p.roll_deg) for p in short.poses
    )

    roll = drawn["roll subsidence"]
    assert max(abs(p.roll_deg) for p in roll.poses) > max(
        abs(p.pitch_deg) for p in roll.poses
    )


def test_a_pitch_rate_still_moves_the_aircraft_when_the_angle_barely_does(flight):
    """A short period is mostly `q`. Animating only `theta` would show an almost stationary
    aeroplane for the mode people most want to see, so rates are integrated into the angle they
    produce over the mode's own timescale."""
    short = _named(shapes.shapes_for(flight, samples=90), "short period")
    assert max(abs(p.pitch_deg) for p in short.poses) > 1.0


# ════════ the decay is visible ══════════════════════════════════════════════════════════════


def test_a_stable_mode_dies_away_over_the_animation(flight):
    short = _named(shapes.shapes_for(flight, samples=90), "short period")
    assert short.stable
    assert short.poses[0].envelope == pytest.approx(1.0)
    assert short.poses[-1].envelope < 0.2


def test_the_envelope_follows_the_modes_own_real_part(flight):
    """`exp(sigma·t)`, not a decoration - so an unstable mode grows on screen by exactly as much as
    it grows in the state space."""
    mode = _named(flight, "phugoid")
    shape = shapes.mode_shape(mode, duration_s=4.0, samples=41)
    last = shape.poses[-1]
    assert last.envelope == pytest.approx(math.exp(mode.eigenvalue.real * 4.0), rel=1e-9)


def test_an_unstable_mode_grows_rather_than_being_clipped(flight):
    """Whatever the aircraft, a divergent mode must read as divergent."""
    mode = _named(flight, "phugoid")
    mode.eigenvalue = complex(+0.4, mode.eigenvalue.imag or 1.0)
    shape = shapes.mode_shape(mode, duration_s=3.0, samples=31)
    assert shape.poses[-1].envelope > 3.0
    assert not shape.stable


# ════════ the amplitude is honest about being arbitrary ═════════════════════════════════════


def test_the_amplitude_is_normalised_and_the_caption_says_so(flight):
    """An eigenvector has no scale. Showing degrees without saying they are normalised invites
    reading them as a prediction of how far the aircraft actually rolls."""
    for shape in shapes.shapes_for(flight, samples=30, amplitude_deg=10.0):
        peak = max(
            max(abs(p.roll_deg), abs(p.pitch_deg), abs(p.yaw_deg)) for p in shape.poses
        )
        assert peak <= 10.0 + 1e-6
        assert "normalised" in shape.caption


def test_the_ratio_between_axes_is_preserved_when_the_amplitude_changes(flight):
    """The absolute size is a display choice; the ratio is the physics. Doubling one must double
    the other exactly."""
    mode = _named(flight, "dutch roll")
    small = shapes.mode_shape(mode, duration_s=2.0, samples=41, amplitude_deg=6.0)
    large = shapes.mode_shape(mode, duration_s=2.0, samples=41, amplitude_deg=12.0)
    for a, b in zip(small.poses, large.poses):
        assert b.roll_deg == pytest.approx(2.0 * a.roll_deg, abs=1e-9)
        assert b.yaw_deg == pytest.approx(2.0 * a.yaw_deg, abs=1e-9)
