"""Never solve the same thing twice.

A VSPAERO sweep costs tens of seconds. Design work re-asks the same question constantly — you
change a tail, look at the wing polar again, change it back — and an optimiser re-asks it by
construction, because a population that has converged is a population full of near-duplicates. So
caching is not an optimisation of this tool, it is a precondition for the way it is meant to be
used, and it is built now rather than bolted on at Phase 5.

**The key is everything that could change the answer.** Geometry, airfoils, atmosphere, speed,
angle range, sideslip, control deflections, reference values, scope, thin-vs-thick, and the
fidelity rung — which is what sets mesh density and wake iterations. Plus the solver's own name.
Miss one and the cache serves a confidently wrong answer, which is far worse than no cache: a stale
result is indistinguishable from a fresh one at the call site. Both dataclasses serialise
completely, so the key is taken over the *whole* serialised form rather than a hand-listed subset —
a new field on `OperatingCondition` is then covered the day it is added, with nothing to remember.

**Provenance travels with the result.** A cached row records the tool versions and the git commit
that produced it. A result whose provenance is unknown is not a result, it is a rumour — and a
cache is precisely where rumours accumulate, because the thing that produced the row is long gone.

Storage is SQLite with the result as a JSON blob. The arrays involved are span loads and polars —
tens of kilobytes — and a blob is the simplest thing that works. When surface pressure fields
arrive they will not fit that description; the sidecar to write then is an `.npz` beside the
database, keyed by the same hash. Not before.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from uavaero.paths import CACHE
from uavaero.schema.airframe import ParametricAirframe
from uavaero.schema.condition import OperatingCondition
from uavaero.schema.results import AeroResult
from uavaero.store.serial import canonical

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    key          TEXT PRIMARY KEY,
    solver       TEXT NOT NULL,
    airframe     TEXT NOT NULL,
    fidelity     TEXT NOT NULL,
    scope        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    elapsed_s    REAL NOT NULL,
    commit_hash  TEXT,
    versions     TEXT NOT NULL,
    condition    TEXT NOT NULL,
    result       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS results_airframe ON results (airframe);
"""


def hash_inputs(*parts: Any) -> str:
    """Stable hash over anything JSON-serialisable.

    Hashed through `canonical()`, so the key is taken over the payload *as it will be persisted*.
    Without that, saving an airframe changed its key — the file stored a float rounded for
    readability while the hash saw the full-precision one — which orphaned every cached solve and
    made every recorded result look stale as soon as the file was reopened.

    `sort_keys` matters too: dict ordering is an implementation detail of however the payload was
    built, and without it the same question asked twice hashes differently and the cache silently
    never hits. `default=str` keeps an unexpected type from raising — it degrades to a distinct
    key rather than to a crash.
    """
    encoded = json.dumps(canonical(parts), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def tool_versions() -> dict[str, str]:
    """Versions of everything that could change a number. Absent packages are recorded as absent
    rather than omitted, because "numpy was missing" and "we forgot to look" are different."""
    import importlib.metadata as md

    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "neuralfoil", "aerosandbox", "openvsp"):
        try:
            versions[package] = md.version(package)
        except Exception:  # noqa: BLE001 - a missing package is a fact, not an error
            versions[package] = "absent"
    return versions


def git_commit(repo: Path | None = None) -> str | None:
    """Short commit hash, suffixed `-dirty` when the tree has uncommitted changes.

    The suffix is the useful half: an uncommitted tree means the code that produced this result no
    longer exists anywhere, so the row is not reproducible and should be read with that in mind.
    """
    cwd = str(repo or Path(__file__).resolve().parent)
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True,
            timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True,
            timeout=5, check=True,
        ).stdout.strip()
        return f"{head}-dirty" if dirty else head
    except Exception:  # noqa: BLE001 - not a git checkout, or no git; neither is an error here
        return None


class ResultCache:
    """SQLite-backed store of solved `AeroResult`s, keyed by everything that produced them."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else CACHE / "results.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        """A connection that survives concurrent writers.

        The optimiser runs several solves at once and every one of them writes here when it
        finishes. SQLite's default is to fail immediately if the database is locked, which on
        Windows surfaces as "the process cannot access the file because it is being used by another
        process" — reported by the driver as a *solve* failure, because that is where it was
        raised. The candidate was fine; the cache was busy.

        `timeout` makes a writer wait its turn instead. WAL lets readers carry on while one writes,
        which is the common case here: many cache lookups, occasional inserts.
        """
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def key(
        airframe: ParametricAirframe, condition: OperatingCondition, solver_name: str
    ) -> str:
        return hash_inputs(solver_name, airframe.to_dict(), condition.to_dict())

    def get(
        self, airframe: ParametricAirframe, condition: OperatingCondition, solver_name: str
    ) -> AeroResult | None:
        key = self.key(airframe, condition, solver_name)
        with self._connect() as db:
            row = db.execute("SELECT result FROM results WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return AeroResult.from_dict(json.loads(row[0]))

    def put(
        self,
        airframe: ParametricAirframe,
        condition: OperatingCondition,
        result: AeroResult,
    ) -> str:
        key = self.key(airframe, condition, result.solver)
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO results "
                "(key, solver, airframe, fidelity, scope, created_at, elapsed_s, commit_hash, "
                " versions, condition, result) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key, result.solver, airframe.name, result.fidelity, result.scope,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    result.elapsed_s, git_commit(), json.dumps(tool_versions()),
                    json.dumps(condition.to_dict()), json.dumps(result.to_dict()),
                ),
            )
        return key

    def entries(self, airframe_name: str | None = None) -> list[dict[str, Any]]:
        """Everything stored, newest first, without deserialising the results themselves."""
        sql = (
            "SELECT key, solver, airframe, fidelity, scope, created_at, elapsed_s, commit_hash "
            "FROM results"
        )
        args: tuple = ()
        if airframe_name is not None:
            sql += " WHERE airframe = ?"
            args = (airframe_name,)
        sql += " ORDER BY created_at DESC"
        columns = ["key", "solver", "airframe", "fidelity", "scope", "created_at", "elapsed_s",
                   "commit_hash"]
        with self._connect() as db:
            return [dict(zip(columns, row)) for row in db.execute(sql, args)]

    def clear(self) -> int:
        with self._connect() as db:
            n = db.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            db.execute("DELETE FROM results")
        return int(n)

    @property
    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits, "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "path": str(self.path),
        }


class CachedSolver:
    """Any `AeroSolver`, with the cache in front of it.

    A wrapper rather than a flag on each backend: caching is the same for every solver, and a
    backend that has to remember to check a cache is a backend that will eventually forget. It
    also means the optimiser can hold a cached solver and a bare one and choose, without either
    backend knowing a cache exists.

    Set `$UAVAERO_NO_CACHE=1` to bypass reads while still writing — which is what you want when
    checking whether a suspicious cached number reproduces.
    """

    def __init__(self, solver, cache: ResultCache | None = None) -> None:
        self.solver = solver
        self.cache = cache if cache is not None else ResultCache()

    @property
    def name(self) -> str:
        return self.solver.name

    def unavailable_for(
        self, airframe: ParametricAirframe, condition: OperatingCondition
    ) -> str:
        return self.solver.unavailable_for(airframe, condition)

    def evaluate(
        self,
        airframe: ParametricAirframe,
        condition: OperatingCondition,
        *,
        workdir: Path | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> AeroResult:
        if not os.environ.get("UAVAERO_NO_CACHE"):
            cached = self.cache.get(airframe, condition, self.solver.name)
            if cached is not None:
                if on_output is not None:
                    on_output(f"cache hit ({self.cache.key(airframe, condition, self.solver.name)})")
                return cached
        result = self.solver.evaluate(
            airframe, condition, workdir=workdir, on_output=on_output,
        )
        self.cache.put(airframe, condition, result)
        return result
