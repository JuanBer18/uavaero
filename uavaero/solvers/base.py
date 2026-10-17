"""The solver interface.

Everything above this line — the drag comparison, the dynamics bridge, the optimiser, the UI —
talks to `AeroSolver`. Nothing above it imports a backend module directly.

The previous generation deliberately had no such interface, on the grounds that a duck-typed
`run_alpha_sweep(airframe, settings, workdir)` naming convention across three backends was cheaper
than an abstraction. That was a reasonable call for a tool with one solver wired into one screen.
It stops being reasonable here: three drag models, two section backends and an optimiser that must
swap fidelity and backend per candidate all need one call signature, and "they happen to be spelled
the same" is not one.

**Failure is an exception, not a sentinel.** `evaluate` raises `SolveFailed`. An optimiser wants
`except SolveFailed: return penalty` — a solver failure is an ordinary and expected event when
something is driving geometry into strange corners, and a `None` that must be checked at every call
site eventually is not. The subprocess isolation underneath still means a *native crash* is a
non-zero exit code rather than a dead interpreter; `SolveFailed` is what that exit code becomes.

**`unavailable_for` is separate from failing.** Asking for a whole-aircraft solve on an airframe
with no fuselage is not a solver error, it is a question that cannot be asked yet — and the honest
answer is a sentence saying what is missing, checked *before* anything expensive starts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from uavaero.schema.airframe import ParametricAirframe
from uavaero.schema.condition import OperatingCondition
from uavaero.schema.results import AeroResult


class SolveFailed(Exception):
    """A solve that was asked for and did not produce a usable answer.

    Carries the solver's own message. A degenerate mesh raises this too: a solve that returns
    zeros is a failure even though the process exited cleanly, and treating it as a result is how
    an impossible aircraft wins an optimisation.
    """

    def __init__(self, message: str, solver: str = "") -> None:
        super().__init__(message)
        self.solver = solver


@runtime_checkable
class AeroSolver(Protocol):
    """One aerodynamic solve: this shape, this condition, one canonical result."""

    #: Stable key recorded on every result this solver produces.
    name: str

    def unavailable_for(self, airframe: ParametricAirframe, condition: OperatingCondition) -> str:
        """Why this solve cannot be run, or "" if it can.

        Checked before the expensive part. The string is shown to the user verbatim, so it says
        what is missing rather than which assertion tripped.
        """
        ...

    def evaluate(
        self,
        airframe: ParametricAirframe,
        condition: OperatingCondition,
        *,
        workdir: Path | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> AeroResult:
        """Solve, or raise `SolveFailed`.

        `on_output` receives solver progress line by line as it arrives. A thirty-second solve
        behind a silent interface is indistinguishable from a hang.
        """
        ...
