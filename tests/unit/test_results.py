"""The result types, and the arithmetic of comparing drag models.

`DragBreakdown` is the piece of this tool that most easily degrades into something dishonest, and
these tests pin the two ways it must not.

The first is that **a reference is never mixed into a prediction**. Once a CFD run or a tunnel
campaign exists, the tempting move is to fold it in — average it, or replace the predictions with
it. Both destroy the only thing the comparison was for: you can no longer say which method was
reliable where, because nothing survives to be scored. References are scored *against*, never
averaged *in*.

The second is that **CD0 is a fitted intercept, not a solver's parasite column**. Those are two
different quantities and one of them is named misleadingly. Reporting a component buildup under
the name CD0 puts two incompatible numbers in the same row of the same table.
"""
from __future__ import annotations

import numpy as np
import pytest

from uavaero.schema.results import (
    AeroResult,
    DragBreakdown,
    DragModel,
    MeshDiagnostics,
    Polar,
    ReferenceValues,
    SpanLoad,
)


def _polar(cl, cd, alpha=(-4.0, 0.0, 4.0, 8.0)) -> Polar:
    n = len(alpha)
    return Polar(
        alpha_deg=np.array(alpha), CL=np.array(cl), CD=np.array(cd), CM=np.zeros(n),
        source="test",
    )


def _breakdown() -> DragBreakdown:
    """Three predictions that disagree, plus one measurement — the shape the comparison exists
    for."""
    breakdown = DragBreakdown(cd_induced=[0.002, 0.006, 0.012, 0.020])
    breakdown.add(DragModel("vspaero_native", "VSPAERO native", [0.030, 0.032, 0.034, 0.041]))
    breakdown.add(DragModel("parasite_buildup", "Parasite buildup", [0.034, 0.036, 0.038, 0.045]))
    breakdown.add(DragModel("strip_neuralfoil", "Strip-corrected", [0.032, 0.034, 0.036, 0.043]))
    breakdown.add(DragModel("tunnel", "Wind tunnel", [0.033, 0.035, 0.037, 0.048], reference=True))
    return breakdown


# ════════ predictions versus references ═════════════════════════════════════════════════════


def test_a_reference_is_not_counted_as_a_prediction():
    breakdown = _breakdown()
    assert [m.key for m in breakdown.predictions()] == [
        "vspaero_native", "parasite_buildup", "strip_neuralfoil",
    ]
    assert [m.key for m in breakdown.references()] == ["tunnel"]


def test_the_spread_measures_disagreement_between_models_not_error_against_truth():
    """A measurement inside the spread would turn "how much do the methods disagree" into "how
    wrong are they", which is a different quantity with a different use — and at index 3 here the
    tunnel value is the outlier, so including it would inflate the spread by a factor of two."""
    breakdown = _breakdown()
    assert breakdown.spread(3) == pytest.approx(0.045 - 0.041)
    assert breakdown.spread(3) < 0.048 - 0.041  # what it would be if the tunnel were folded in


def test_relative_spread_is_the_number_worth_thresholding():
    """An absolute spread of 0.004 is negligible on a brick and enormous on a sailplane."""
    breakdown = _breakdown()
    mean = (0.030 + 0.034 + 0.032) / 3
    assert breakdown.relative_spread(0) == pytest.approx((0.034 - 0.030) / mean)


def test_a_lone_model_has_no_spread_rather_than_a_misleading_zero_of_its_own():
    """One model does not agree with itself — there is simply nothing to compare. Zero is the
    honest answer only because the caller is expected to notice there is one prediction."""
    lone = DragBreakdown()
    lone.add(DragModel("only", "Only", [0.03, 0.03]))
    assert lone.spread(0) == 0.0
    assert lone.relative_spread(0) == 0.0


def test_error_against_a_reference_is_signed_so_over_prediction_is_visible():
    """Positive means the model predicts more drag than was measured. An unsigned error would hide
    the systematic bias that makes a model useful to correct."""
    errors = _breakdown().error_against("tunnel", 0)
    assert errors["vspaero_native"] == pytest.approx((0.030 - 0.033) / 0.033)
    assert errors["vspaero_native"] < 0        # under-predicts
    assert errors["parasite_buildup"] > 0      # over-predicts
    assert "tunnel" not in errors              # a reference is not scored against itself


def test_adding_a_model_never_removes_one():
    """Later validation must not overwrite the predictions it is judging."""
    breakdown = _breakdown()
    before = list(breakdown.models)
    breakdown.add(DragModel("cfd", "CFD", [0.031, 0.033, 0.035, 0.044], reference=True))
    assert list(breakdown.models) == before + ["cfd"]


# ════════ the result ════════════════════════════════════════════════════════════════════════


def test_the_primary_model_is_what_lift_to_drag_reports():
    """Which model is primary changes every headline number, so it is recorded on the result
    rather than decided by whoever plots it."""
    result = AeroResult(
        polar=_polar([0.0, 0.4, 0.8, 1.1], [0.030, 0.032, 0.034, 0.041]),
        drag=_breakdown(), primary_model="strip_neuralfoil",
    )
    assert result.cd == [0.032, 0.034, 0.036, 0.043]
    assert result.lift_to_drag[1] == pytest.approx(0.4 / 0.034)


def test_a_result_with_no_primary_model_falls_back_to_the_polars_own_drag():
    """Rather than raising. A solver that registered no drag models still produced a polar, and
    that polar is a usable, if uncorrected, answer."""
    result = AeroResult(polar=_polar([0.0, 0.4, 0.8, 1.1], [0.03, 0.032, 0.034, 0.041]))
    assert result.cd == [0.03, 0.032, 0.034, 0.041]


def test_best_lift_to_drag_reports_the_angle_it_happens_at():
    """The angle matters as much as the ratio: an aircraft whose best L/D sits at 10 degrees is
    telling you something different from one whose best sits at 2, even at the same ratio."""
    result = AeroResult(polar=_polar([0.0, 0.4, 0.8, 1.1], [0.030, 0.020, 0.060, 0.120]))
    ratio, alpha = result.best_lift_to_drag
    assert alpha == 0.0
    assert ratio == pytest.approx(0.4 / 0.020)


def test_an_empty_sweep_reports_zero_rather_than_raising():
    """A caller formatting a failed result should not have to guard against an IndexError."""
    empty = AeroResult(polar=Polar(np.array([]), np.array([]), np.array([]), np.array([])))
    assert empty.best_lift_to_drag == (0.0, 0.0)


def test_cd0_is_the_fitted_intercept_and_the_fit_recovers_what_built_it():
    """Constructed from a known CD = CD0 + CL^2/(pi AR e), then fitted back. This is the property
    that makes CD0 comparable across solvers: it is a fit to the drag polar, never a solver's own
    parasite buildup column, which is a wetted-area estimate of a different quantity."""
    cd0_true, e_true, ar = 0.0250, 0.85, 9.0
    cl = np.array([0.0, 0.3, 0.6, 0.9])
    cd = cd0_true + cl**2 / (np.pi * ar * e_true)
    result = AeroResult(
        polar=Polar(np.array([-2.0, 1.0, 4.0, 7.0]), cl, cd, np.zeros(4)),
        reference=ReferenceValues(sref_m2=1.0, bref_m=3.0, cref_m=0.333),
    )
    assert result.reference.aspect_ratio == pytest.approx(ar)
    cd0_fit, e_fit = result.fit_cd0_e()
    assert cd0_fit == pytest.approx(cd0_true, rel=1e-6)
    assert e_fit == pytest.approx(e_true, rel=1e-6)


def test_cd0_is_never_negative():
    """A noisy three-point sweep can fit a slightly negative intercept. Reporting one would be
    reporting an aircraft that produces thrust at zero lift."""
    cl = np.array([0.5, 0.7, 0.9])
    result = AeroResult(polar=Polar(np.array([0.0, 2.0, 4.0]), cl, np.array([0.02, 0.03, 0.05]),
                                    np.zeros(3)),
                        reference=ReferenceValues(1.0, 3.0, 0.333))
    assert result.fit_cd0_e()[0] >= 0.0


def test_aspect_ratio_uses_the_full_span():
    """`ReferenceValues.bref_m` is the FULL span — the doubling `LiftingSurface.span()` does not
    do. Passing a half span here quadruples the aspect ratio, and the only symptom is an Oswald
    efficiency above 1, which is easy to read as a merely surprising result."""
    assert ReferenceValues(sref_m2=0.6, bref_m=2.4).aspect_ratio == pytest.approx(9.6)
    assert ReferenceValues(sref_m2=0.6, bref_m=1.2).aspect_ratio == pytest.approx(2.4)


def test_an_aspect_ratio_with_no_reference_area_is_zero_rather_than_a_division_error():
    assert ReferenceValues(sref_m2=0.0, bref_m=2.4).aspect_ratio == 0.0


# ════════ persistence ═══════════════════════════════════════════════════════════════════════


def test_a_result_survives_a_round_trip_through_json():
    """It is stored in the cache as a JSON blob and read back as though freshly solved. A field
    that does not survive is a field that silently becomes a default on every cache hit."""
    import json

    original = AeroResult(
        polar=_polar([0.0, 0.4, 0.8, 1.1], [0.030, 0.032, 0.034, 0.041]),
        drag=_breakdown(),
        span_loads=[SpanLoad(alpha_deg=4.0, y_m=[0.1, 0.3], chord_m=[0.3, 0.2],
                             cl_local=[0.5, 0.4], root_bending_moment_coefficient=0.21)],
        reference=ReferenceValues(0.6, 2.4, 0.253),
        mesh=MeshDiagnostics(nodes=187, loops=300, kutta_nodes=11, surfaces=1),
        reynolds=346877.0, mach=0.0588, solver="vspaero", fidelity="quick", scope="wing",
        elapsed_s=9.5, primary_model="strip_neuralfoil",
    )
    restored = AeroResult.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.to_dict() == original.to_dict()
    assert restored.primary_model == "strip_neuralfoil"
    assert [m.key for m in restored.drag.predictions()] == [
        "vspaero_native", "parasite_buildup", "strip_neuralfoil",
    ]
    assert restored.drag.models["tunnel"].reference is True
    assert restored.span_loads[0].root_bending_moment_coefficient == pytest.approx(0.21)


def test_a_mesh_with_no_kutta_nodes_is_not_ok():
    """This is the degenerate-mesh signature. It does not fail the solve — it returns zeros, which
    reads as a spectacularly low-drag aircraft rather than as an error, and that is exactly why it
    has to be caught by a flag rather than noticed by a human."""
    assert MeshDiagnostics(nodes=200, loops=300, kutta_nodes=0).ok is False
    assert MeshDiagnostics(nodes=200, loops=300, kutta_nodes=11).ok is True
    assert MeshDiagnostics().ok is False


def test_a_polar_round_trips_its_optional_lateral_columns():
    """CY/Cl/Cn are None on a symmetric solve and populated on a beta sweep. None has to survive
    as None, because an array of zeros means "measured, and zero"."""
    plain = _polar([0.0, 0.4, 0.8, 1.1], [0.03, 0.032, 0.034, 0.041])
    assert Polar.from_dict(plain.to_dict()).CY is None

    lateral = _polar([0.0, 0.4, 0.8, 1.1], [0.03, 0.032, 0.034, 0.041])
    lateral.CY = np.array([0.0, -0.01, -0.02, -0.03])
    assert Polar.from_dict(lateral.to_dict()).CY == pytest.approx(lateral.CY)
