"""What the tool tells you while it works, and what it tells you when it stops.

Both of these were real usability failures rather than cosmetic ones.

**A solve looked identical whether it was running or hung.** The output was already being tailed —
`run_isolated` watches the solver's log file for exactly that reason — and every line went into
`statusBar().showMessage(line[:160])`, where it replaced its predecessor and vanished. A real
solve emits 483 lines; you could see one of them at a time, truncated.

**A failure reported three lines of context.** Almost always the wrong three: VSPAERO announces a
problem part-way up its output and then keeps printing, so the tail is the aftermath rather than
the cause.

`parse_progress` is tested against strings taken verbatim from a real solver log, because a
progress parser tested against invented strings is a parser tested against its own author's memory
of the format.
"""
from __future__ import annotations

import pytest

from uavaero.solvers.vspaero import diagnose, parse_progress

#: Lines copied from a real `vspaero.log`, not written from memory.
REAL_LINES = {
    "case": "case: 3 of 7 ",
    "solving": "Solving... Mach: 0.058773 ... Alpha: 4.000000 .",
    "wake": "    Wake Iter:     2 / 3     ... GMRES Iter:     2 ",
    "noise": "Creating quad tree velocity survey data structures... ",
}


# ════════ progress ══════════════════════════════════════════════════════════════════════════


def test_the_case_count_gives_an_exact_fraction():
    """VSPAERO states its own case count. That is better than counting `Solving...` lines, which
    drifts whenever cases and angles differ — a stability run perturbs about a single angle and
    still reports seven cases."""
    fraction, what = parse_progress(REAL_LINES["case"], alpha_points=3)
    assert fraction == pytest.approx(3 / 7)
    assert what == "case 3 of 7"


def test_the_case_count_wins_over_the_angle_count():
    """Both can appear in one log. The exact one has to take priority or the bar jumps between two
    disagreeing measures of the same run."""
    parse_progress.seen = 0
    both = "case: 2 of 4 ... Solving... Mach: 0.05 ... Alpha: 4.000000"
    fraction, what = parse_progress(both, alpha_points=3)
    assert what.startswith("case")
    assert fraction == pytest.approx(0.5)


def test_an_angle_line_advances_the_sweep():
    parse_progress.seen = 0
    fraction, what = parse_progress(REAL_LINES["solving"], alpha_points=3)
    assert fraction == pytest.approx(1 / 3)
    assert "4.000000" in what


def test_a_wake_iteration_is_indeterminate_rather_than_a_fraction():
    """Its total is not known in advance in a way that maps onto the whole solve, so it drives a
    busy indicator instead of a number. Reporting it as a percentage would make the bar run
    backwards at the start of every angle."""
    fraction, what = parse_progress(REAL_LINES["wake"], alpha_points=3)
    assert fraction == -1.0
    assert what == "wake iteration 2/3"


def test_ordinary_chatter_is_not_progress():
    assert parse_progress(REAL_LINES["noise"], alpha_points=3) is None
    assert parse_progress("", alpha_points=3) is None


# ════════ failure diagnostics ═══════════════════════════════════════════════════════════════


def test_a_failure_carries_the_whole_capture_not_the_last_three_lines(tmp_path):
    captured = [f"line {i}" for i in range(1, 60)]
    message = diagnose("the worker exited without writing a result", captured, tmp_path)

    assert "the worker exited without writing a result" in message
    assert "line 59" in message
    assert "line 30" in message, "the middle of the output is where the cause usually is"


def test_the_run_directory_is_named_because_that_is_where_the_evidence_is(tmp_path):
    """`job.json`, `vspaero.log` and the `.vsp3` are the actual evidence. The message is a summary
    of them, and it has to say where they are."""
    message = diagnose("it failed", ["something"], tmp_path)
    assert str(tmp_path) in message
    assert "run directory" in message


def test_the_solver_log_is_included_when_there_is_one(tmp_path):
    (tmp_path / "vspaero.log").write_text(
        "\n".join(["setting up"] + [f"iterating {i}" for i in range(50)] + ["FATAL: mesh is bad"]),
        encoding="utf-8",
    )
    message = diagnose("it failed", [], tmp_path)
    assert "FATAL: mesh is bad" in message
    assert "vspaero.log" in message


def test_a_failure_with_nothing_captured_still_says_where_to_look(tmp_path):
    """A native crash can leave no output at all. The message must still be actionable."""
    message = diagnose("the worker exited 3221226505", [], tmp_path)
    assert "3221226505" in message
    assert str(tmp_path) in message


def test_the_message_survives_being_carried_as_one_string(tmp_path):
    """It crosses a worker boundary and rides on `SolveFailed`, so it has to be a single string —
    the console splits it back into lines at the other end."""
    message = diagnose("headline", ["a", "b"], tmp_path)
    assert isinstance(message, str)
    assert message.splitlines()[0] == "headline"


# ════════ the console's own behaviour ═══════════════════════════════════════════════════════


def test_the_console_trims_a_long_block_from_the_front():
    """The end of a traceback names the failure; the start names whichever frame was outermost.
    Trimming the wrong end is how a path bug presented as fifteen identical messages reading
    "return io.open(self"."""
    from uavaero.app.widgets.console import Console

    kept: list[str] = []

    class _Stub(Console):
        def __init__(self):  # noqa: D107 - no Qt in a unit test
            pass

        def append(self, text, level="info"):
            kept.append(text)

    _Stub().block("\n".join(f"line {i}" for i in range(1, 101)), limit=10)
    assert any("90 earlier line(s) omitted" in line for line in kept)
    assert any("line 100" in line for line in kept)
    assert not any("line 5" == line.strip() for line in kept)
