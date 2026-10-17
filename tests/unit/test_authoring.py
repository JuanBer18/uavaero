"""Building a whole aircraft from the numbers an engineer quotes.

The property this module exists to protect: **the arm you ask for is the arm you get**.

A tail is quoted as area and arm because tail volume — area times arm, against the wing — is what
decides whether the aircraft is controllable. If `build` placed a tail by its root leading edge
instead of by its aerodynamic centre, the arm would come out short by a quarter of the tail chord.
That is a few percent of Vh, it drifts as tail taper changes, and nothing about the resulting
aircraft looks wrong — you would simply be flying a slightly different aeroplane than the one whose
numbers you checked.

The second property is that **napkin numbers are a view, never a second copy**. `infer_tail_sizing`
is the inverse of the placement, so the editor can show area and arm for a tail that was loaded
from a file or hand-edited station by station, without anything storing those numbers separately
and going stale.
"""
from __future__ import annotations

import math

import pytest

from uavaero.geometry.authoring import (
    TailSizing,
    add_default_component,
    aerodynamic_centre_x,
    build,
    horizontal_tail_volume,
    infer_tail_sizing,
    apply_wing,
    is_v_tail,
    set_surface_height,
    surface_height,
    remove_component,
    summary,
    tail_arm,
    vertical_tail_volume,
)
from uavaero.geometry.planform import Planform


def _wing(**kwargs) -> Planform:
    base = dict(span=2.40, aspect_ratio=9.60, taper=0.667, sweep_mode="straight_qc",
                dihedral_deg=3.0, tip_twist_deg=-2.0)
    base.update(kwargs)
    return Planform(**base)


def _aircraft(**kwargs):
    defaults = dict(
        htail=TailSizing(area_m2=0.098, aspect_ratio=2.50, arm_m=0.880, taper=0.75,
                         incidence_deg=-1.0),
        vtail=TailSizing(area_m2=0.034, aspect_ratio=1.85, arm_m=0.880, taper=0.70),
        fuselage_style="conventional", wing_position="high",
    )
    defaults.update(kwargs)
    return build("test-aircraft", _wing(), **defaults)


# ════════ the arm you ask for ═══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("arm", [0.60, 0.880, 1.25])
def test_the_horizontal_tail_lands_at_the_arm_it_was_given(arm):
    aircraft = _aircraft(htail=TailSizing(area_m2=0.098, aspect_ratio=2.5, arm_m=arm))
    assert tail_arm(aircraft, aircraft.htail) == pytest.approx(arm, abs=1e-9)


def test_the_arm_holds_when_the_tail_is_reshaped():
    """The regression that catches placing by root leading edge. Taper changes where a surface's
    aerodynamic centre sits inside it, so a placement that ignored the MAC would drift here while
    continuing to look entirely reasonable."""
    for taper in (0.4, 0.7, 1.0):
        aircraft = _aircraft(htail=TailSizing(area_m2=0.098, aspect_ratio=2.5, arm_m=0.9, taper=taper))
        assert tail_arm(aircraft, aircraft.htail) == pytest.approx(0.9, abs=1e-9), taper


def test_the_arm_holds_when_the_tail_is_swept():
    """Sweep moves the MAC aft inside the surface. Same failure mode, different cause."""
    aircraft = _aircraft(
        htail=TailSizing(area_m2=0.098, aspect_ratio=2.5, arm_m=0.9, sweep_deg=20.0)
    )
    assert tail_arm(aircraft, aircraft.htail) == pytest.approx(0.9, abs=1e-9)


def test_the_arm_is_measured_from_the_wings_aerodynamic_centre_not_its_root():
    """A swept wing's AC is well aft of its root leading edge. If the arm were measured from the
    root, sweeping the wing would silently lengthen every tail arm and inflate both tail volumes."""
    straight = build("s", _wing(sweep_mode="manual", sweep_deg=0.0),
                     htail=TailSizing(0.098, 2.5, 0.9))
    swept = build("w", _wing(sweep_mode="manual", sweep_deg=25.0),
                  htail=TailSizing(0.098, 2.5, 0.9))
    assert aerodynamic_centre_x(swept.wing) > aerodynamic_centre_x(straight.wing)
    assert tail_arm(swept, swept.htail) == pytest.approx(0.9, abs=1e-9)
    assert tail_arm(straight, straight.htail) == pytest.approx(0.9, abs=1e-9)


# ════════ the tail volumes ══════════════════════════════════════════════════════════════════


def test_horizontal_tail_volume_matches_its_definition():
    aircraft = _aircraft()
    wing = aircraft.wing
    expected = (
        aircraft.htail.area() * tail_arm(aircraft, aircraft.htail)
        / (wing.area() * wing.mean_aerodynamic_chord())
    )
    assert horizontal_tail_volume(aircraft) == pytest.approx(expected)
    assert horizontal_tail_volume(aircraft) == pytest.approx(0.567, abs=0.005)


def test_vertical_tail_volume_is_referenced_to_the_full_wing_span():
    """`LiftingSurface.span()` is one side. Halving the reference span doubles Vv, which makes an
    under-finned aircraft look adequate — a quiet failure in the flattering direction."""
    aircraft = _aircraft()
    wing = aircraft.wing
    full_span = wing.span() * 2.0
    expected = aircraft.vtail.area() * tail_arm(aircraft, aircraft.vtail) / (wing.area() * full_span)
    assert vertical_tail_volume(aircraft) == pytest.approx(expected)
    assert vertical_tail_volume(aircraft) == pytest.approx(0.021, abs=0.003)


def test_a_fins_area_is_not_doubled():
    """A vertical surface is single-sided, which is also how fin area is quoted. Doubling it would
    double Vv."""
    aircraft = _aircraft(vtail=TailSizing(area_m2=0.034, aspect_ratio=1.85, arm_m=0.88))
    assert aircraft.vtail.symmetric is False
    assert aircraft.vtail.area() == pytest.approx(0.034, rel=1e-6)


def test_tail_volumes_are_none_without_a_tail():
    """None rather than zero. A tailless aircraft does not have a tail volume of nought — it does
    not have one, and a readout showing 0.000 invites the reader to conclude it is dangerously
    under-tailed."""
    tailless = build("flying-wing", _wing(), htail=None, vtail=None, fuselage_style="none")
    assert horizontal_tail_volume(tailless) is None
    assert vertical_tail_volume(tailless) is None


def test_optional_components_can_be_added_to_and_removed_from_a_wing_only_aircraft():
    """The geometry UI is a component builder, so an absent part must be real model state rather
    than a disabled tab that leaves hidden default geometry behind."""
    aircraft = build("wing-only", _wing(), htail=None, vtail=None, fuselage_style="none")

    assert add_default_component(aircraft, "htail") == "htail"
    assert aircraft.htail is not None
    assert add_default_component(aircraft, "vtail") == "vtail"
    assert aircraft.vtail is not None
    assert add_default_component(aircraft, "fuselage") == "fuselage"
    assert aircraft.fuselages

    remove_component(aircraft, "fuselage")
    remove_component(aircraft, "vtail")
    remove_component(aircraft, "htail")
    assert aircraft.htail is None
    assert aircraft.vtail is None
    assert aircraft.fuselages == []


def test_the_v_tail_add_action_builds_one_symmetric_dihedral_surface():
    """A V-tail is one mirrored lifting surface with large dihedral, never a horizontal plus a
    vertical tail hidden behind a configuration label."""
    aircraft = build("wing-only", _wing(), htail=None, vtail=None, fuselage_style="none")

    assert add_default_component(aircraft, "v_tail") == "htail"
    assert aircraft.htail is not None
    assert aircraft.htail.symmetric
    assert aircraft.htail.sections[-1].dihedral_deg == pytest.approx(38.0)
    assert aircraft.vtail is None


def test_a_v_tail_is_recognisable_as_one_after_the_fact():
    """Storing a V-tail as a high-dihedral horizontal tail keeps every downstream consumer free of
    a second configuration to handle, and the price is that the screen would otherwise label it
    "Horizontal tail" — which reads as the add action having done the wrong thing."""
    assert not is_v_tail(_aircraft())

    vee = build("vee", _wing(), htail=None, vtail=None, fuselage_style="none")
    add_default_component(vee, "v_tail")
    assert is_v_tail(vee)


def test_an_aircraft_with_no_horizontal_tail_is_not_a_v_tail():
    assert not is_v_tail(build("bare", _wing(), htail=None, vtail=None, fuselage_style="none"))


def test_height_is_measured_from_the_aircraft_datum_not_from_the_wing():
    """`TailSizing.z_m` lands in `origin[2]` unchanged, so it is an absolute height above z = 0.

    The field's own comment said "above the wing root" until 2026-09-02, which is a different
    number the moment the wing is not at the datum — and a T-tail placed by that reading would sit
    wherever the wing happened to be.
    """
    aircraft = _aircraft()
    set_surface_height(aircraft, "wing", 0.20)
    set_surface_height(aircraft, "htail", 0.35)

    assert aircraft.wing.origin[2] == pytest.approx(0.20)
    assert aircraft.htail.origin[2] == pytest.approx(0.35)
    assert infer_tail_sizing(aircraft, aircraft.htail, vertical=False).z_m == pytest.approx(0.35)


def test_moving_a_surface_vertically_disturbs_nothing_else():
    """Height is independent of the arm, which is measured along x, and of the tail volume, which
    is areas and lengths. If moving a wing from low to high re-sized the tail, the two editors
    would be fighting each other."""
    aircraft = _aircraft()
    before = summary(aircraft)

    set_surface_height(aircraft, "wing", 0.18)
    set_surface_height(aircraft, "htail", 0.40)
    after = summary(aircraft)

    for key in ("htail_arm_m", "vtail_arm_m", "horizontal_tail_volume", "vertical_tail_volume",
                "wing_area_m2", "mac_m"):
        assert after[key] == pytest.approx(before[key]), key


def test_a_height_survives_reshaping_the_wing():
    """`apply_wing` carries `origin[2]` through, so editing span or taper must not drop the
    aircraft back to the datum."""
    aircraft = _aircraft()
    set_surface_height(aircraft, "wing", 0.15)
    apply_wing(aircraft, _wing(span=3.0))
    assert surface_height(aircraft, "wing") == pytest.approx(0.15)


def test_asking_for_the_height_of_a_component_that_is_gone_is_not_an_error():
    """The geometry screen can remove a tail, and a readout must not raise for asking."""
    bare = build("bare", _wing(), htail=None, vtail=None, fuselage_style="none")
    assert surface_height(bare, "htail") is None
    assert surface_height(bare, "wing") is not None
    with pytest.raises(ValueError, match="no htail"):
        set_surface_height(bare, "htail", 0.1)


# ════════ what the aircraft comes out as ════════════════════════════════════════════════════


def test_a_built_aircraft_is_complete():
    """Wing, both tails and a fuselage — so every solve scope is reachable. The previous generation
    could only produce a wing, which made the whole-aircraft scope permanently unavailable."""
    aircraft = _aircraft()
    assert [s.name for s in aircraft.lifting_surfaces()] == ["wing", "htail", "vtail"]
    assert aircraft.fuselages
    assert not aircraft.is_tailless()


def test_the_wing_comes_out_at_the_span_and_aspect_ratio_asked_for():
    aircraft = _aircraft()
    facts = summary(aircraft)
    assert facts["wing_span_m"] == pytest.approx(2.40, rel=1e-6)
    assert facts["wing_aspect_ratio"] == pytest.approx(9.60, rel=1e-3)
    assert facts["wing_area_m2"] == pytest.approx(2.40**2 / 9.60, rel=1e-3)


def test_derived_values_follow_the_dependent_variable_relationships():
    """S = b²/AR, c_root = 2S/(b(1+λ)), c_tip = λ·c_root. The screen shows these beside the inputs,
    so they have to be the real relationships rather than a remembered constant."""
    span, ar, taper = 2.40, 9.60, 0.667
    facts = summary(_aircraft())
    area = span**2 / ar
    root = 2 * area / (span * (1 + taper))
    assert facts["wing_area_m2"] == pytest.approx(area, rel=1e-3)
    assert facts["root_chord_m"] == pytest.approx(root, rel=1e-3)
    assert facts["tip_chord_m"] == pytest.approx(taper * root, rel=1e-3)


def test_incidence_is_applied_to_the_whole_tail_not_just_the_tip():
    """Incidence on a tail is the surface's setting angle. Applying it only to the tip station
    would silently turn it into washout, which is a different thing entirely."""
    aircraft = _aircraft(htail=TailSizing(0.098, 2.5, 0.88, incidence_deg=-2.5))
    assert [s.twist_deg for s in aircraft.htail.sections] == [-2.5, -2.5]


def test_an_unknown_fuselage_style_is_refused_by_name():
    with pytest.raises(ValueError, match="Unknown fuselage style"):
        build("x", _wing(), fuselage_style="spaceship")


def test_an_unknown_wing_position_is_refused_by_name():
    with pytest.raises(ValueError, match="Unknown wing position"):
        build("x", _wing(), wing_position="sideways")


# ════════ the inverse ═══════════════════════════════════════════════════════════════════════


def test_reading_the_tail_numbers_back_recovers_what_was_asked_for():
    """The property that lets napkin numbers be a view rather than a stored copy."""
    asked = TailSizing(area_m2=0.11, aspect_ratio=3.2, arm_m=0.95, taper=0.6, incidence_deg=-1.5)
    aircraft = build("x", _wing(), htail=asked)
    got = infer_tail_sizing(aircraft, aircraft.htail, vertical=False)
    assert got.area_m2 == pytest.approx(asked.area_m2, rel=1e-6)
    assert got.aspect_ratio == pytest.approx(asked.aspect_ratio, rel=1e-6)
    assert got.arm_m == pytest.approx(asked.arm_m, abs=1e-9)
    assert got.taper == pytest.approx(asked.taper, rel=1e-6)
    assert got.incidence_deg == pytest.approx(asked.incidence_deg)


def test_reading_a_fin_back_recovers_it_too():
    asked = TailSizing(area_m2=0.042, aspect_ratio=1.6, arm_m=0.91, taper=0.55)
    aircraft = build("x", _wing(), vtail=asked)
    got = infer_tail_sizing(aircraft, aircraft.vtail, vertical=True)
    assert got.area_m2 == pytest.approx(asked.area_m2, rel=1e-6)
    assert got.aspect_ratio == pytest.approx(asked.aspect_ratio, rel=1e-6)
    assert got.arm_m == pytest.approx(asked.arm_m, abs=1e-9)


def test_every_derived_value_is_finite_for_a_normal_aircraft():
    """A NaN in this block reaches the screen as a readout, and a NaN that is displayed is a number
    someone will believe."""
    for key, value in summary(_aircraft()).items():
        assert value is not None, key
        assert math.isfinite(value), key
