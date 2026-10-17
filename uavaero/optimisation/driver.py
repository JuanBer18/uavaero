"""Running the search: screen, solve, gate, score — and remembering everything it saw.

Differential evolution, because the design space is not smooth. A candidate can be rejected by
screening, fail to mesh, or produce an untrustworthy result, and all three are cliffs rather than
slopes. A gradient method needs derivatives that do not exist across a cliff; DE only needs to be
able to compare two points, which survives the whole thing.

**Coarse to fine, using the fidelity ladder built in Phase 0.** The search runs at QUICK, where a
solve is seconds and the answer is consistently coarse — consistent being the operative word, since
ranking is all a search needs. The finalists are then re-solved at STANDARD or HIGH, and *that* is
where the numbers you quote come from. Optimising directly at HIGH would spend an overnight run to
reach the same shape.

**Every evaluation is remembered, accepted or not.** The rejection ledger is the diagnostic that
tells you the bounds are wrong, and the accepted history is what a convergence plot is drawn from.
A run that discards its failures can only tell you it did not work.

Parallelism is threads over subprocesses. Each solve already runs in its own OpenVSP process, so
the threads spend their lives waiting on those and the GIL never enters into it.

**Parallel OpenVSP is measurably unreliable, and a crash biases the search.** Measured on this
machine over the same sixteen candidates: serial accepted 14 of 16 with zero solver failures, while
four workers accepted 9 and lost five to native crashes with no log and no result. The geometry was
identical and fine — the crashes are contention, not candidates.

That is worse than it sounds. A crashed candidate scores `REJECTED_SCORE`, so a *good* aircraft
that happened to crash is indistinguishable from a bad one, and the population learns to avoid a
region for no reason. So a failed solve is retried once, alone, before it is believed: a genuine
failure fails twice, and a contention artefact usually does not. It costs one serial solve per
crash and removes the bias.
"""
from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from uavaero.optimisation.objectives import (
    REJECTED_SCORE,
    Candidate,
    StabilityConstraints,
    trust,
)
from uavaero.optimisation.screening import Limits, RejectionLedger, screen
from uavaero.optimisation.space import DesignSpace
from uavaero.paths import RUNS
from uavaero.schema.condition import Fidelity, OperatingCondition
from uavaero.solvers.base import SolveFailed


@dataclass
class OptimisationResult:
    """What a run produced, and what it threw away getting there."""

    best: Candidate | None
    history: list[Candidate] = field(default_factory=list)
    ledger: RejectionLedger = field(default_factory=RejectionLedger)
    evaluations: int = 0
    elapsed_s: float = 0.0
    converged: bool = False
    message: str = ""
    #: Solves that failed once and were retried alone. High against `evaluations` means the worker
    #: count is above what this machine's OpenVSP tolerates.
    retries: int = 0

    @property
    def accepted(self) -> list[Candidate]:
        return [c for c in self.history if c.accepted]

    @property
    def acceptance_rate(self) -> float:
        return len(self.accepted) / len(self.history) if self.history else 0.0

    def best_so_far(self) -> list[float]:
        """The convergence curve: the best score seen up to each accepted evaluation."""
        curve: list[float] = []
        running = REJECTED_SCORE
        for candidate in self.accepted:
            running = min(running, candidate.score)
            curve.append(running)
        return curve

    def summary(self) -> str:
        if self.best is None:
            return (
                f"No candidate survived out of {self.evaluations}. "
                f"{self.ledger.summary()}"
            )
        lines = [
            f"Best score {self.best.score:.4f} after {self.evaluations} evaluations "
            f"in {self.elapsed_s / 60:.1f} min",
            f"Accepted {len(self.accepted)} of {len(self.history)} "
            f"({self.acceptance_rate * 100:.0f}%)",
        ]
        if self.ledger.total:
            worst = self.ledger.worst
            if worst:
                lines.append(f"Most common rejection: {worst[1]}x {worst[0]}")
        return "\n".join(lines)


class Optimiser:
    """One search over one design space."""

    def __init__(
        self,
        space: DesignSpace,
        condition: OperatingCondition,
        objective: Callable,
        limits: Limits | None = None,
        constraints: StabilityConstraints | None = None,
        search_fidelity: Fidelity = Fidelity.QUICK,
        workers: int = 4,
        workdir: Path | None = None,
        require_all_models: bool = True,
    ) -> None:
        self.space = space
        self.condition = condition
        self.objective = objective
        self.limits = limits or Limits()
        self.constraints = constraints
        self.search_fidelity = search_fidelity
        self.workers = max(1, workers)
        self.workdir = Path(workdir) if workdir else RUNS / "optimisation"
        self.require_all_models = require_all_models

        self.history: list[Candidate] = []
        self.ledger = RejectionLedger()
        self.retries = 0
        self._cache: dict[tuple, Candidate] = {}
        # Held during a retry so the retry runs alone. A contention failure retried under the same
        # contention proves nothing.
        self._retry_lock = threading.Lock()

    # ── evaluating one candidate ───────────────────────────────────────────────────────

    def evaluate(self, vector: np.ndarray) -> Candidate:
        """Screen, solve, gate, score. Never raises — a failure is a rejected candidate."""
        key = tuple(np.round(np.asarray(vector, dtype=float), 6))
        if key in self._cache:
            return self._cache[key]

        started = time.time()
        values = self.space.describe(vector)

        def reject(reason: str) -> Candidate:
            self.ledger.record(reason)
            candidate = Candidate(
                vector=np.asarray(vector, dtype=float), values=values,
                score=REJECTED_SCORE, accepted=False, reason=reason,
                elapsed_s=time.time() - started,
            )
            self._cache[key] = candidate
            self.history.append(candidate)
            return candidate

        try:
            airframe = self.space.build(vector)
        except Exception as exc:  # noqa: BLE001 - a geometry that will not build is a rejection
            return reject(f"geometry could not be built: {type(exc).__name__}: {exc}")

        verdict = screen(airframe, self.condition, self.limits)
        if not verdict:
            return reject(verdict.reason)

        # The directory is derived from the candidate itself rather than a counter. A counter
        # is a read-modify-write across worker threads, so two candidates can take the same number
        # and then clobber each other's job and result files; a hash cannot collide by accident and
        # has the useful side effect that re-evaluating a candidate reuses its directory.
        tag = hashlib.sha256(str(key).encode()).hexdigest()[:10]
        condition = _at_fidelity(self.condition, self.search_fidelity)
        try:
            result = self._solve_with_retry(airframe, condition, tag)
        except SolveFailed as exc:
            return reject(f"solve failed twice: {exc}")
        except Exception as exc:  # noqa: BLE001
            return reject(f"solve raised: {type(exc).__name__}: {exc}")

        gate = trust(result, require_all_models=self.require_all_models)
        if not gate:
            return reject(gate.reason)

        static_margin = None
        if self.constraints is not None:
            try:
                derivatives = self._stability(airframe, condition, tag)
            except Exception as exc:  # noqa: BLE001
                return reject(f"stability solve failed: {type(exc).__name__}: {exc}")
            allowed = self.constraints.check(derivatives)
            if not allowed:
                return reject(allowed.reason)
            static_margin = derivatives.static_margin

        score = float(self.objective(result))
        if not np.isfinite(score):
            return reject("the objective came back non-finite")

        best_ld, _ = result.best_lift_to_drag
        candidate = Candidate(
            vector=np.asarray(vector, dtype=float), values=values, score=score, accepted=True,
            lift_to_drag=best_ld, cd=float(np.min(result.cd)), static_margin=static_margin,
            elapsed_s=time.time() - started, notes=list(result.notes),
        )
        self._cache[key] = candidate
        self.history.append(candidate)
        return candidate

    def _solve_with_retry(self, airframe, condition, tag: str):
        """Solve, and if it fails, try once more with the other workers held off.

        Parallel OpenVSP crashes under contention. Retrying alone separates "this candidate is
        impossible" from "the machine was busy", and only the first is a fact about the aircraft.
        """
        try:
            return self._solve(airframe, condition, tag)
        except Exception as first:  # noqa: BLE001
            self.retries += 1
            with self._retry_lock:
                try:
                    return self._solve(airframe, condition, f"{tag}-retry")
                except Exception:  # noqa: BLE001 - the second failure is the believable one
                    raise first from None

    def _solve(self, airframe, condition, tag: str):
        from uavaero.solvers.vspaero import VSPAeroSolver
        from uavaero.store.cache import CachedSolver, ResultCache

        solver = CachedSolver(VSPAeroSolver(), ResultCache())
        return solver.evaluate(airframe, condition, workdir=self.workdir / tag)

    def _stability(self, airframe, condition, tag: str):
        from uavaero.solvers import stability

        return stability.solve(
            airframe, condition, workdir=self.workdir / tag / "stability",
            cross_check=False,
        )

    # ── the search ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        population: int = 12,
        generations: int = 20,
        seed: int | None = 0,
        on_progress: Callable[[int, Candidate], None] | None = None,
    ) -> OptimisationResult:
        """Differential evolution over the space.

        `population` is a multiplier on the number of variables, which is how SciPy counts it — a
        twelve-variable space at population 12 is 144 candidates per generation. Start smaller than
        feels right; at tens of seconds a solve, the arithmetic gets away from you quickly.
        """
        from scipy.optimize import differential_evolution

        started = time.time()
        evaluated = 0

        def cost(vector: np.ndarray) -> float:
            nonlocal evaluated
            candidate = self.evaluate(vector)
            evaluated += 1
            if on_progress is not None:
                on_progress(evaluated, candidate)
            return candidate.score

        outcome = differential_evolution(
            cost,
            bounds=self.space.bounds,
            popsize=population,
            maxiter=generations,
            seed=seed,
            polish=False,   # polishing runs a gradient method, which the cliffs defeat
            workers=1,      # parallelism is inside `evaluate`, not here
            updating="immediate",
            tol=0.01,
        )

        accepted = [c for c in self.history if c.accepted]
        best = min(accepted, key=lambda c: c.score) if accepted else None
        return OptimisationResult(
            best=best, history=list(self.history), ledger=self.ledger,
            evaluations=evaluated, elapsed_s=time.time() - started,
            converged=bool(outcome.success), message=str(outcome.message),
            retries=self.retries,
        )

    def explore(
        self,
        samples: int = 32,
        seed: int | None = 0,
        on_progress: Callable[[int, Candidate], None] | None = None,
    ) -> OptimisationResult:
        """A parallel random sweep of the space, for seeing its shape before searching it.

        Worth running first. It costs a known amount of time, it exercises the screening bounds,
        and if it accepts nothing then the search would have spent an hour discovering the same
        thing one candidate at a time.
        """
        rng = np.random.default_rng(seed)
        low = np.array([b[0] for b in self.space.bounds])
        high = np.array([b[1] for b in self.space.bounds])
        vectors = rng.uniform(low, high, size=(samples, len(low)))

        started = time.time()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for index, candidate in enumerate(pool.map(self.evaluate, vectors), start=1):
                if on_progress is not None:
                    on_progress(index, candidate)

        accepted = [c for c in self.history if c.accepted]
        best = min(accepted, key=lambda c: c.score) if accepted else None
        return OptimisationResult(
            best=best, history=list(self.history), ledger=self.ledger,
            evaluations=samples, elapsed_s=time.time() - started,
            converged=False, message="random exploration",
            retries=self.retries,
        )

    def refine(self, candidate: Candidate, fidelity: Fidelity = Fidelity.STANDARD):
        """Re-solve one candidate at a higher rung.

        The search runs at QUICK because ranking only needs consistency. The number you quote has
        to come from somewhere better, and this is where — a finalist re-solved at STANDARD or HIGH,
        once, rather than the whole population.
        """
        airframe = self.space.build(candidate.vector)
        condition = _at_fidelity(self.condition, fidelity)
        tag = hashlib.sha256(str(tuple(np.round(candidate.vector, 6))).encode()).hexdigest()[:10]
        result = self._solve(airframe, condition, f"{tag}-refined")
        return airframe, result, trust(result, require_all_models=self.require_all_models)


def _at_fidelity(condition: OperatingCondition, fidelity: Fidelity) -> OperatingCondition:
    from dataclasses import replace

    return replace(condition, fidelity=fidelity)
