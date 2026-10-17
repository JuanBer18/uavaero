"""A stored library of XFOIL polars, so XFOIL runs once per section rather than once per solve.

XFOIL is the reason this exists. A NeuralFoil polar is milliseconds and needs no store; an XFOIL
polar is seconds, fails unpredictably, and is worth keeping for as long as the airfoil coordinates
and the flow conditions have not changed.

**Convergence state is stored, not inferred.** A row records how many angles were asked for and how
many came back, so a polar that only half-solved is identifiable later as a partial answer rather
than mistaken for a well-behaved section with a narrow envelope. `SectionLookup` then refuses to
interpolate across the holes — see `base.MAX_CL_GAP`.

**A total failure is stored too.** An airfoil XFOIL cannot solve at a given Reynolds is a fact
about that pair, and re-deriving it costs the same timeout every time. The row holds no polar and
`get` reports it as a known failure rather than a miss.

The Reynolds grid below is the one to precompute over before an optimisation run: the optimiser
must not be the thing that discovers an airfoil takes forty seconds to characterise.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from uavaero.paths import CACHE
from uavaero.schema.results import Polar

#: Plan §10's grid. Spaced tightly at the bottom because that is where low-Reynolds behaviour
#: changes fastest — the difference between Re 50k and 75k is a different airfoil, and the
#: difference between 750k and 1M is a rounding error.
STANDARD_RE_GRID: tuple[float, ...] = (
    50_000, 75_000, 100_000, 125_000, 150_000, 200_000, 250_000,
    300_000, 400_000, 500_000, 750_000, 1_000_000,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS polars (
    key           TEXT PRIMARY KEY,
    airfoil       TEXT NOT NULL,
    reynolds      REAL NOT NULL,
    n_crit        REAL NOT NULL,
    mach          REAL NOT NULL,
    created_at    TEXT NOT NULL,
    n_requested   INTEGER NOT NULL,
    n_converged   INTEGER NOT NULL,
    cl_max        REAL,
    failed        INTEGER NOT NULL DEFAULT 0,
    note          TEXT,
    polar         TEXT
);
CREATE INDEX IF NOT EXISTS polars_airfoil ON polars (airfoil);
"""


def _key(airfoil: str, reynolds: float, n_crit: float, mach: float) -> str:
    return f"{airfoil.lower()}|{reynolds:.0f}|{n_crit:g}|{mach:g}"


class PolarDatabase:
    """SQLite store of section polars, keyed by airfoil and flow condition."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else CACHE / "polars.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # Same concurrency exposure as the result cache; same treatment. See store/cache.py.
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def get(
        self, airfoil: str, reynolds: float, n_crit: float = 9.0, mach: float = 0.0
    ) -> Polar | None:
        """The stored polar, or None if it is not there.

        A stored *failure* also returns None, but without counting as a miss — the caller is not
        being told "go and compute this", it is being told "this was computed and it did not work".
        Use `known_failure` to tell the two apart.
        """
        with self._connect() as db:
            row = db.execute(
                "SELECT polar, failed FROM polars WHERE key = ?",
                (_key(airfoil, reynolds, n_crit, mach),),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        blob, failed = row
        if failed or not blob:
            return None
        self.hits += 1
        return Polar.from_dict(json.loads(blob))

    def known_failure(
        self, airfoil: str, reynolds: float, n_crit: float = 9.0, mach: float = 0.0
    ) -> str | None:
        """The recorded reason this pair could not be solved, if it was tried and failed."""
        with self._connect() as db:
            row = db.execute(
                "SELECT failed, note FROM polars WHERE key = ?",
                (_key(airfoil, reynolds, n_crit, mach),),
            ).fetchone()
        if row is None or not row[0]:
            return None
        return row[1] or "XFOIL did not converge"

    def put(
        self,
        airfoil: str,
        reynolds: float,
        polar: Polar,
        n_crit: float = 9.0,
        mach: float = 0.0,
    ) -> None:
        meta = polar.meta or {}
        cl_values = np.asarray(polar.CL, dtype=float)
        self._write(
            airfoil, reynolds, n_crit, mach,
            n_requested=int(meta.get("n_requested", len(cl_values))),
            n_converged=int(meta.get("n_converged", len(cl_values))),
            cl_max=float(np.max(cl_values)) if len(cl_values) else None,
            failed=False, note=None, blob=json.dumps(polar.to_dict()),
        )

    def put_failure(
        self,
        airfoil: str,
        reynolds: float,
        note: str,
        n_requested: int = 0,
        n_crit: float = 9.0,
        mach: float = 0.0,
    ) -> None:
        self._write(
            airfoil, reynolds, n_crit, mach, n_requested=n_requested, n_converged=0,
            cl_max=None, failed=True, note=note, blob=None,
        )

    def _write(
        self, airfoil: str, reynolds: float, n_crit: float, mach: float, *,
        n_requested: int, n_converged: int, cl_max: float | None,
        failed: bool, note: str | None, blob: str | None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO polars (key, airfoil, reynolds, n_crit, mach, created_at,"
                " n_requested, n_converged, cl_max, failed, note, polar)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    _key(airfoil, reynolds, n_crit, mach), airfoil.lower(), float(reynolds),
                    float(n_crit), float(mach),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    n_requested, n_converged, cl_max, int(failed), note, blob,
                ),
            )

    def entries(self, airfoil: str | None = None) -> list[dict[str, Any]]:
        """What is stored, without deserialising the polars. `n_converged` against `n_requested`
        is the health of each row."""
        columns = ["airfoil", "reynolds", "n_crit", "mach", "created_at",
                   "n_requested", "n_converged", "cl_max", "failed", "note"]
        sql = f"SELECT {', '.join(columns)} FROM polars"
        args: tuple = ()
        if airfoil is not None:
            sql += " WHERE airfoil = ?"
            args = (airfoil.lower(),)
        sql += " ORDER BY airfoil, reynolds"
        with self._connect() as db:
            return [dict(zip(columns, row)) for row in db.execute(sql, args)]

    def clear(self) -> int:
        with self._connect() as db:
            n = db.execute("SELECT COUNT(*) FROM polars").fetchone()[0]
            db.execute("DELETE FROM polars")
        return int(n)


def precompute(
    airfoils: list[str],
    database: PolarDatabase | None = None,
    reynolds_grid: tuple[float, ...] = STANDARD_RE_GRID,
    n_crit: float = 9.0,
    on_progress=None,
) -> dict[str, int]:
    """Fill the database ahead of time. Returns {airfoil: rows written}.

    Worth running before an optimisation: the optimiser should not be the thing that discovers a
    section takes forty seconds to characterise, and a failure found here is a design decision
    rather than a mysterious stall in a long run.
    """
    from uavaero.solvers.sections.base import ALPHA_GRID_DEG
    from uavaero.solvers.sections.xfoil import run_polar

    database = database or PolarDatabase()
    written: dict[str, int] = {}
    for airfoil in airfoils:
        count = 0
        for reynolds in reynolds_grid:
            if database.get(airfoil, reynolds, n_crit) is not None:
                continue
            if database.known_failure(airfoil, reynolds, n_crit) is not None:
                continue
            try:
                polar = run_polar(airfoil, ALPHA_GRID_DEG, reynolds, n_crit=n_crit)
                database.put(airfoil, reynolds, polar, n_crit=n_crit)
                count += 1
            except Exception as exc:  # noqa: BLE001 - a failure is a row, not a stop
                database.put_failure(
                    airfoil, reynolds, f"{type(exc).__name__}: {exc}",
                    n_requested=len(ALPHA_GRID_DEG), n_crit=n_crit,
                )
            if on_progress is not None:
                on_progress(airfoil, reynolds)
        written[airfoil] = count
    return written
