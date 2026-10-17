"""The unit of work: one aircraft, the condition it is being studied at, and every solve it has.

One file, hand-editable YAML, in a flat library. There is no project directory, no iteration chain,
no workflow state and nothing upstream of it — the airframe is authored directly and this is where
it lives.

**Results are kept inside the file, beside the aircraft that produced them.** Reopening an airframe
gives back every solve you have run on it, which is what makes the drag comparison worth anything:
the whole point is holding several answers side by side over time, and an answer that vanishes when
you close the window cannot participate. The cache in `cache.py` is a speed-up and is disposable;
this file is the record and is not.

Each solve is stored with the condition that produced it and a key over the airframe *as it was at
the time*. That distinction matters: edit the wing and the old solve does not silently start
claiming to describe the new one — it is still findable, still labelled, and simply no longer
matches the current key. Nothing is deleted behind your back.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uavaero.paths import AIRFRAMES
from uavaero.schema.airframe import ParametricAirframe, default_airframe
from uavaero.schema.condition import OperatingCondition
from uavaero.schema.mass import MassProperties
from uavaero.schema.results import AeroResult
from uavaero.store.cache import ResultCache
from uavaero.store.serial import dump_yaml, load_yaml

SCHEMA_VERSION = 1
SUFFIX = ".yaml"


class AirframeFileError(Exception):
    """Something wrong with a file on disk, phrased for the person who has to fix it."""


def slug(name: str) -> str:
    """A filename from a display name. Collapses anything awkward rather than rejecting it —
    naming an aircraft should not be a validation exercise."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return cleaned or "airframe"


@dataclass
class Solve:
    """One recorded answer, with the question that produced it."""

    key: str
    condition: OperatingCondition
    result: AeroResult
    recorded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "recorded_at": self.recorded_at,
            "condition": self.condition.to_dict(),
            "result": self.result.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Solve":
        return cls(
            key=payload.get("key", ""),
            condition=OperatingCondition.from_dict(payload["condition"]),
            result=AeroResult.from_dict(payload["result"]),
            recorded_at=payload.get("recorded_at", ""),
        )


@dataclass
class AirframeFile:
    """One aircraft and everything known about it."""

    name: str = "airframe"
    airframe: ParametricAirframe = field(default_factory=default_airframe)
    condition: OperatingCondition = field(default_factory=OperatingCondition)
    #: What it weighs and how that mass is distributed. Stated, never sized — see schema/mass.py.
    #: Empty by default, because an aerodynamic study needs none of it and only the dynamics do.
    mass: MassProperties = field(default_factory=MassProperties)
    solves: list[Solve] = field(default_factory=list)
    created_at: str = ""
    modified_at: str = ""
    notes: str = ""
    #: Where it was loaded from. Deliberately NOT serialised — a file that carried its own path
    #: would claim the wrong one the moment it was copied or renamed on disk.
    path: Path | None = None

    def record(self, result: AeroResult, condition: OperatingCondition | None = None) -> Solve:
        """Store a solve, replacing any earlier answer to the identical question.

        Replacing rather than appending keeps the file from filling with near-duplicates as you
        re-solve the same thing — and the replaced one is not lost, because an identical question
        has an identical answer. A *different* question keeps its own entry, which is what makes
        several fidelities or scopes readable side by side.
        """
        condition = self.condition if condition is None else condition
        key = ResultCache.key(self.airframe, condition, result.solver)
        entry = Solve(
            key=key, condition=condition, result=result,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.solves = [s for s in self.solves if s.key != key] + [entry]
        return entry

    def current_solves(self) -> list[Solve]:
        """Solves that describe the aircraft as it is right now.

        A solve whose key no longer matches was measured on a shape you have since changed. It is
        kept — deleting someone's results because they nudged a chord would be indefensible — but
        it must never be presented as current, because a stale polar looks exactly like a fresh one.
        """
        return [
            s for s in self.solves
            if s.key == ResultCache.key(self.airframe, s.condition, s.result.solver)
        ]

    def stale_solves(self) -> list[Solve]:
        current = {s.key for s in self.current_solves()}
        return [s for s in self.solves if s.key not in current]

    def latest(self, solver: str | None = None) -> Solve | None:
        """Most recently recorded solve that still describes the current aircraft."""
        candidates = [
            s for s in self.current_solves()
            if solver is None or s.result.solver == solver
        ]
        return candidates[-1] if candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "notes": self.notes,
            "condition": self.condition.to_dict(),
            "mass": self.mass.to_dict(),
            "airframe": self.airframe.to_dict(),
            "solves": [s.to_dict() for s in self.solves],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AirframeFile":
        version = payload.get("schema_version", SCHEMA_VERSION)
        if version > SCHEMA_VERSION:
            raise AirframeFileError(
                f"This airframe was written by a newer version of the tool "
                f"(file format {version}, this build understands {SCHEMA_VERSION}). "
                f"Update rather than opening it — saving would discard whatever is new."
            )
        return cls(
            name=payload.get("name", "airframe"),
            airframe=ParametricAirframe.from_dict(payload["airframe"]),
            condition=OperatingCondition.from_dict(payload.get("condition") or {}),
            mass=MassProperties.from_dict(payload.get("mass") or {}),
            solves=[Solve.from_dict(s) for s in payload.get("solves") or []],
            created_at=payload.get("created_at", ""),
            modified_at=payload.get("modified_at", ""),
            notes=payload.get("notes", ""),
        )


# ════════ the library ═══════════════════════════════════════════════════════════════════════

def library_dir() -> Path:
    return AIRFRAMES


def path_for(name: str, library: Path | None = None) -> Path:
    return (library or library_dir()) / f"{slug(name)}{SUFFIX}"


def names(library: Path | None = None) -> list[str]:
    directory = library or library_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob(f"*{SUFFIX}"))


def create(
    name: str,
    airframe: ParametricAirframe | None = None,
    condition: OperatingCondition | None = None,
) -> AirframeFile:
    """A new airframe, not yet on disk. Defaults to the complete conventional starter — wing,
    both tails and a fuselage — because an aircraft you cannot solve every scope of is a worse
    starting point than one you can edit."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    built = airframe if airframe is not None else default_airframe()
    built.name = slug(name)
    return AirframeFile(
        name=name,
        airframe=built,
        condition=condition or OperatingCondition(),
        created_at=now,
        modified_at=now,
    )


def save(file: AirframeFile, path: Path | None = None) -> Path:
    target = Path(path) if path is not None else (file.path or path_for(file.name))
    target.parent.mkdir(parents=True, exist_ok=True)
    file.modified_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    target.write_text(dump_yaml(file.to_dict()), encoding="utf-8")
    file.path = target
    return target


def load(name_or_path: str | Path, library: Path | None = None) -> AirframeFile:
    """Accepts a library name or a path to a file anywhere."""
    candidate = Path(name_or_path)
    target = candidate if candidate.suffix == SUFFIX else path_for(str(name_or_path), library)
    if not target.is_file():
        raise AirframeFileError(f"No airframe at {target}")
    try:
        payload = load_yaml(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a malformed file is a message, not a traceback
        raise AirframeFileError(f"{target.name} could not be read: {exc}") from exc
    if not payload:
        raise AirframeFileError(f"{target.name} is empty")
    loaded = AirframeFile.from_dict(payload)
    loaded.path = target
    return loaded


def delete(name: str, library: Path | None = None) -> None:
    path_for(name, library).unlink(missing_ok=True)
