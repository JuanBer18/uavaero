"""Airfoil name resolution — the spec stores plain lowercase name strings ("naca2412", "sd7062").

NACA 4-digit sections are generated analytically here (pure numpy, no dependency); anything else
resolves through AeroSandbox's bundled UIUC database via a lazy import, so Stage-1 spec code stays
importable without aerosandbox installed.

`write_openvsp_dat` is the OpenVSP-specific path: it writes a Selig .dat with an open (blunt)
trailing edge explicitly closed, because blunt-TE coordinate files were confirmed (in AEROAPP) to
degenerate VSPAERO's panel mesh at small model scale. NACA names never go through the file path —
the OpenVSP adapter builds those parametrically (XS_FOUR_SERIES).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import aerosandbox as asb

#: Below this TE gap the loop is already closed — nothing to rewrite.
_TE_GAP_CLOSE_TOLERANCE = 1e-9

_logger = logging.getLogger(__name__)

#: Small, reliability-first shortlist for the airfoil-sweep study (core/studies/airfoil_sweep.py)
#: — well-documented low-Re sections with real wind-tunnel/UIUC pedigree, not a large or
#: speculative list. e387 is the most extensively wind-tunnel-tested low-Re airfoil in the UIUC
#: database (a de facto validation standard); sd7062/s4022/s5010/s7075 are Selig-Donovan low-Re
#: park-flyer/sailplane sections; e423 is a well-characterized Eppler high-lift low-Re section;
#: ag35 is a Drela AG-series sailplane section; mh32 is a well-documented Hepperle low-Re section;
#: the two NACA-4 references are analytically generated (no database dependency) and already used
#: as this app's own defaults (default_airframe()).
CURATED_UAS_AIRFOILS: list[str] = [
    "sd7062", "e387", "e423", "s4022", "s5010", "s7075", "ag35", "mh32", "naca2412", "naca0012",
]

#: Named starting sets for the Airfoil Explorer's preset selection mode. Entries are candidate
#: names, not guarantees — resolve_airfoil_preset() intersects with list_available_airfoils() so a
#: database layout change degrades to a smaller preset, never an error. "Sweep-all" mode ignores
#: these entirely and takes the whole database through the geometry/trust filters instead.
AIRFOIL_PRESETS: dict[str, list[str]] = {
    "Curated UAS": list(CURATED_UAS_AIRFOILS),
    "Low-Re sailplane / park flyer": [
        "sd7062", "sd7037", "sd7032", "sd7080", "s4022", "s5010", "s7075", "sa7035", "sa7036",
        "ag35", "ag24", "ag25", "ag26", "ag27", "mh32", "e387", "e205", "e214", "e216", "rg15",
    ],
    "High lift": [
        "e423", "s1223", "s1210", "e214", "e216", "fx63137", "e64", "goe417a", "ch10sm",
    ],
    "Thick / structural": [
        "naca2415", "naca2418", "naca4415", "naca4418", "naca0015", "naca0018", "clarky",
        "fx63137", "e562", "goe398",
    ],
    "Reflexed / flying wing": [
        "mh45", "mh60", "mh61", "mh64", "mh78", "e184", "e186", "e325", "e340", "s5020",
    ],
    "NACA 4-digit classics": [
        "naca0009", "naca0012", "naca1408", "naca1412", "naca2408", "naca2410", "naca2412",
        "naca2415", "naca4412", "naca4415", "naca6409",
    ],
}

#: name-prefix -> designer family, longest prefix wins (so "sd" beats "s", "ag" beats "a").
_FAMILY_PREFIXES: list[tuple[str, str]] = [
    ("naca", "NACA"),
    ("goe", "Gottingen"),
    ("clark", "Clark"),
    ("usa", "USA"),
    ("sd", "Selig-Donovan"),
    ("sa", "Selig-Ashok"),
    ("ag", "Drela AG"),
    ("ht", "Drela HT"),
    ("mh", "Hepperle MH"),
    ("fx", "Wortmann FX"),
    ("rg", "RG"),
    ("dae", "Drela DAE"),
    ("ah", "Althaus AH"),
    ("nlf", "NASA NLF"),
    ("ls(", "NASA LS"),
    ("e", "Eppler"),
    ("s", "Selig"),
]


def airfoil_family(name: str) -> str:
    """Designer family for a database name ("e387" -> "Eppler"), by longest matching prefix —
    prefixes with trailing digits are how the UIUC database is actually organized. Unrecognized
    names group under "Other"."""
    lowered = name.lower()
    for prefix, family in sorted(_FAMILY_PREFIXES, key=lambda pair: -len(pair[0])):
        if lowered.startswith(prefix):
            return family
    return "Other"


def resolve_airfoil_preset(preset: str) -> list[str]:
    """Preset entries that actually resolve on this machine (intersection with the database +
    NACA names), preserving preset order. Raises KeyError for an unknown preset name."""
    available = set(list_available_airfoils())
    return [name for name in AIRFOIL_PRESETS[preset] if name in available]


@lru_cache(maxsize=1)
def list_available_airfoils() -> list[str]:
    """Every airfoil name resolvable via get_asb_airfoil(): AeroSandbox's bundled UIUC .dat
    directory (the full candidate set for a wide airfoil-sweep study) plus CURATED_UAS_AIRFOILS
    (NACA-4 names aren't files on disk, so wouldn't otherwise appear). Falls back to
    CURATED_UAS_AIRFOILS alone, with a logged warning, if the database directory can't be found —
    keeps the airfoil-sweep UI usable even if AeroSandbox's internal package layout changes."""
    import aerosandbox

    db_dir = Path(aerosandbox.__file__).parent / "geometry" / "airfoil" / "airfoil_database"
    if not db_dir.is_dir():
        _logger.warning("AeroSandbox airfoil database not found at %s; falling back to the curated list", db_dir)
        return list(CURATED_UAS_AIRFOILS)

    names = {p.stem.lower() for p in db_dir.glob("*.dat")}
    names.update(CURATED_UAS_AIRFOILS)
    return sorted(names)


def parse_naca4(name: str) -> tuple[float, float, float] | None:
    """(max camber, camber position, thickness) as chord fractions for a 4-digit NACA name
    ("naca2412" or bare "2412"), or None if the name isn't a 4-digit NACA designation."""
    digits = name.lower().removeprefix("naca")
    if len(digits) != 4 or not digits.isdigit():
        return None
    return int(digits[0]) / 100.0, int(digits[1]) / 10.0, int(digits[2:4]) / 100.0


def naca4_coordinates(name: str, n_points: int = 100) -> np.ndarray:
    """(N, 2) unit-chord coordinate loop for a NACA 4-digit section, closed, ordered upper
    surface TE->LE then lower surface LE->TE (UIUC/Selig convention). Cosine-spaced stations.
    Ported from AEROAPP engine/aero_airfoil.py (proven against XFOIL there)."""
    parsed = parse_naca4(name)
    if parsed is None:
        raise ValueError(f"Not a 4-digit NACA designation: '{name}'")
    m, p, tt = parsed

    beta = np.linspace(0.0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2.0  # LE (0) to TE (1)

    yt = 5 * tt * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            m / p**2 * (2 * p * x - x**2),
            m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2),
        )
        dyc_dx = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    theta = np.arctan(dyc_dx)

    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)

    x_loop = np.concatenate([xu[::-1], xl[1:]])
    y_loop = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack([x_loop, y_loop])


@lru_cache(maxsize=128)
def get_asb_airfoil(name: str) -> "asb.Airfoil":
    """Resolve a spec airfoil name to an asb.Airfoil with real coordinates. NACA 4-digit names are
    generated by AeroSandbox itself; other names come from its bundled UIUC database. Raises
    ValueError for a name neither can resolve, rather than returning a coordinate-less shell.
    Cached (treated as read-only downstream) — the live viewport loop resolves the same few names
    on every keystroke."""
    import aerosandbox as asb  # lazy: keeps uavaero.geometry importable without aerosandbox

    airfoil = asb.Airfoil(name)
    if airfoil.coordinates is None:
        raise ValueError(f"Unknown airfoil '{name}' (not NACA-4 and not in the UIUC database)")
    return airfoil


def airfoil_coordinates(name: str, n_points: int = 100) -> np.ndarray:
    """(N, 2) unit-chord Selig-ordered loop for any spec airfoil name. NACA-4 analytically,
    everything else through AeroSandbox's database."""
    if parse_naca4(name) is not None:
        return naca4_coordinates(name, n_points=n_points)
    return np.asarray(get_asb_airfoil(name).coordinates, dtype=float)


def thickness_camber(name: str, fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(thickness, camber) of an airfoil at the given chord fractions, both as chord fractions —
    used by the blended-body fuselage generator to shape a center body like an airfoil. The
    coordinate loop is split at its minimum-x point (the LE) into upper/lower surfaces, each
    interpolated onto the requested stations."""
    coords = airfoil_coordinates(name, n_points=200)
    i_le = int(np.argmin(coords[:, 0]))
    upper = coords[: i_le + 1][::-1]  # LE -> TE
    lower = coords[i_le:]  # LE -> TE
    fractions = np.clip(np.asarray(fractions, dtype=float), 0.0, 1.0)
    yu = np.interp(fractions, upper[:, 0], upper[:, 1])
    yl = np.interp(fractions, lower[:, 0], lower[:, 1])
    return yu - yl, (yu + yl) / 2.0


def close_trailing_edge(coordinates: np.ndarray) -> np.ndarray:
    """Snap an open (blunt) trailing edge closed by moving both loop endpoints to their shared
    midpoint. Returns the input unchanged when the loop is already closed."""
    coords = np.asarray(coordinates, dtype=float)
    if len(coords) < 2:
        return coords
    gap = float(np.hypot(*(coords[0] - coords[-1])))
    if gap <= _TE_GAP_CLOSE_TOLERANCE:
        return coords
    closed = coords.copy()
    midpoint = (coords[0] + coords[-1]) / 2.0
    closed[0] = closed[-1] = midpoint
    return closed


@lru_cache(maxsize=2048)
def airfoil_geometry_stats(name: str) -> dict[str, float]:
    """{max_thickness, max_camber (both chord fractions, camber signed), n_points} — measured
    from the resolved coordinate loop, cached because the sweep-all screening filter asks for
    every database airfoil. Raises for a name that doesn't resolve."""
    coords = airfoil_coordinates(name, n_points=150)
    fractions = np.linspace(0.01, 0.99, 80)
    thickness, camber = thickness_camber(name, fractions)
    max_camber = float(camber[np.argmax(np.abs(camber))])
    return {
        "max_thickness": float(np.max(thickness)),
        "max_camber": max_camber,
        "n_points": float(len(coords)),
    }


def airfoil_validity_issue(name: str) -> str | None:
    """None when the section's geometry is usable; otherwise a short reason. This is the
    pre-sweep gate that keeps degenerate database .dat files (few/broken points, non-normalized
    or non-physical shapes) out of the screening batch — bad geometry in, garbage polars out."""
    try:
        coords = airfoil_coordinates(name)
        stats = airfoil_geometry_stats(name)
    except Exception as exc:
        return f"unresolvable: {exc}"
    if len(coords) < 20:
        return f"only {len(coords)} coordinate points"
    if not np.all(np.isfinite(coords)):
        return "non-finite coordinates"
    x = coords[:, 0]
    if x.min() < -0.05 or x.max() > 1.05 or (x.max() - x.min()) < 0.9:
        return "coordinates not normalized to unit chord"
    if stats["max_thickness"] < 0.01:
        return f"degenerate thickness ({stats['max_thickness']:.3f} t/c)"
    if stats["max_thickness"] > 0.5:
        return f"implausible thickness ({stats['max_thickness']:.3f} t/c)"
    return None


def write_openvsp_dat(name: str, path: Path, n_points: int = 100) -> Path:
    """Write a TE-closed Selig .dat for OpenVSP's ReadFileAirfoil. Only used for non-NACA
    airfoils — the OpenVSP adapter builds NACA sections parametrically."""
    coords = close_trailing_edge(airfoil_coordinates(name, n_points=n_points))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name} (TE-closed)"] + [f"{x:.6f} {y:.6f}" for x, y in coords]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
