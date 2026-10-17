"""Tier 1 cross-check — REAL XFOIL 2D section polars, via AeroSandbox's `asb.XFoil` wrapper.

NeuralFoil (core/aero/neuralfoil_backend.py) is a surrogate trained ON XFOIL, so this backend is
its ground truth: the screening funnel ranks with NeuralFoil (milliseconds per polar) and
verifies finalists here (a real subprocess per run, seconds per polar). Same relationship as
native-VLM vs AVL at Tier 2 — an independent implementation to check against, not a replacement.

Two honesty notes baked into the contract:
- XFOIL silently DROPS unconverged alpha points — the returned Polar's alpha array is the
  converged subset of what was requested (verified empirically: naca2412 @ Re=2e5 returned 7 of
  8 requested points). `meta["n_requested"]`/`meta["n_converged"]` record the attrition; heavy
  attrition is itself a signal the section is hard to analyze at that Reynolds.
- `asb.XFoil` manages its own subprocess with timeout+kill (like `asb.AVL`), so a hung XFOIL run
  kills only that child — no extra watchdog wrapper needed here.

Binary resolution mirrors avl_backend.avl_binary_path(): env override -> bundled external/xfoil/
xfoil.exe (not committed) -> bare "xfoil" on PATH.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from uavaero.paths import external_binary
from uavaero.schema.results import Polar
from uavaero.geometry.airfoils import get_asb_airfoil


def xfoil_binary_path() -> str:
    """Resolution order: $UAVDESIGNER_XFOIL_PATH -> bundled external/xfoil/xfoil.exe -> "xfoil"
    on PATH. A missing binary surfaces as a normal job failure (FileNotFoundError from the
    subprocess launch), never a hang."""
    env_override = os.environ.get("UAVDESIGNER_XFOIL_PATH")
    if env_override:
        return env_override
    bundled = external_binary("xfoil", "xfoil.exe")
    if bundled.is_file():
        return str(bundled)
    return "xfoil"


def run_polar(
    airfoil_name: str,
    alpha_deg: np.ndarray,
    reynolds: float,
    mach: float = 0.0,
    n_crit: float = 9.0,
    timeout_s: float = 60.0,
) -> Polar:
    """Real-XFOIL polar with the same signature shape as neuralfoil_backend.run_polar, so the
    two are directly comparable point-for-point. The returned alpha array is XFOIL's CONVERGED
    subset of the request (see module docstring)."""
    import aerosandbox as asb  # lazy: keeps uavaero.solvers importable without aerosandbox

    airfoil = get_asb_airfoil(airfoil_name)
    requested = np.asarray(alpha_deg, dtype=float)

    xfoil = asb.XFoil(
        airfoil=airfoil,
        Re=reynolds,
        mach=mach,
        n_crit=n_crit,
        xfoil_command=xfoil_binary_path(),
        timeout=timeout_s,
    )
    result = xfoil.alpha(requested)

    converged = np.asarray(result["alpha"], dtype=float)
    if len(converged) == 0:
        raise RuntimeError(
            f"XFOIL converged on none of the {len(requested)} requested alphas for "
            f"'{airfoil_name}' at Re={reynolds:.0f} — the section is likely pathological there"
        )
    return Polar(
        alpha_deg=converged,
        CL=np.asarray(result["CL"], dtype=float),
        CD=np.asarray(result["CD"], dtype=float),
        CM=np.asarray(result["CM"], dtype=float),
        source="xfoil",
        meta={
            "airfoil": airfoil_name, "reynolds": reynolds, "mach": mach, "n_crit": n_crit,
            "n_requested": int(len(requested)), "n_converged": int(len(converged)),
            "top_xtr": np.asarray(result["Top_Xtr"], dtype=float).tolist(),
            "bot_xtr": np.asarray(result["Bot_Xtr"], dtype=float).tolist(),
        },
    )


def compare_with_neuralfoil(
    airfoil_name: str,
    alpha_deg: np.ndarray,
    reynolds: float,
    mach: float = 0.0,
    n_crit: float = 9.0,
    timeout_s: float = 60.0,
    model_size: str = "large",
) -> dict:
    """One airfoil, both solvers, plus point-matched deltas (NeuralFoil interpolated onto
    XFOIL's converged alphas). This is the empirical answer to "is NeuralFoil good enough for
    this section at this Reynolds" — JSON-plain for job transport."""
    from uavaero.solvers.sections.neuralfoil import run_polar as run_neuralfoil

    xfoil_polar = run_polar(
        airfoil_name, alpha_deg, reynolds, mach=mach, n_crit=n_crit, timeout_s=timeout_s,
    )
    neural_polar = run_neuralfoil(
        airfoil_name, np.asarray(alpha_deg, dtype=float), reynolds, mach=mach,
        model_size=model_size,
    )

    # Compare on XFOIL's converged alphas only — interpolating NeuralFoil (smooth, dense) onto
    # them is well-posed; the reverse is not.
    nf_cl = np.interp(xfoil_polar.alpha_deg, neural_polar.alpha_deg, neural_polar.CL)
    nf_cd = np.interp(xfoil_polar.alpha_deg, neural_polar.alpha_deg, neural_polar.CD)
    delta_cl = nf_cl - xfoil_polar.CL
    delta_cd = nf_cd - xfoil_polar.CD

    return {
        "airfoil": airfoil_name,
        "reynolds": reynolds,
        "xfoil": xfoil_polar.to_dict(),
        "neuralfoil": neural_polar.to_dict(),
        "max_abs_delta_cl": float(np.max(np.abs(delta_cl))),
        "max_abs_delta_cd": float(np.max(np.abs(delta_cd))),
        "mean_abs_delta_cd": float(np.mean(np.abs(delta_cd))),
        "n_requested": xfoil_polar.meta["n_requested"],
        "n_converged": xfoil_polar.meta["n_converged"],
    }


class XFoilSource:
    """XFOIL behind the common section interface, backed by the polar database.

    Not the default. XFOIL is the ground truth NeuralFoil is calibrated against and the right
    source for a handful of finalists — but it fails unpredictably at low Reynolds, and a solver
    that sometimes returns nothing is a bad thing to put underneath an optimiser.

    Every solve is stored, including the failures: an airfoil XFOIL cannot handle at a given
    Reynolds is a fact about that pair, and rediscovering it costs the same timeout each time.
    """

    name = "xfoil"

    def __init__(
        self,
        database: "PolarDatabase | None" = None,
        n_crit: float = 9.0,
        timeout_s: float = 60.0,
    ) -> None:
        from uavaero.solvers.sections.polar_db import PolarDatabase

        self.database = database if database is not None else PolarDatabase()
        self.n_crit = n_crit
        self.timeout_s = timeout_s

    def polar(
        self, airfoil: str, alpha_deg: np.ndarray, reynolds: float, mach: float = 0.0
    ) -> Polar:
        stored = self.database.get(airfoil, reynolds, self.n_crit, mach)
        if stored is not None:
            return stored

        known = self.database.known_failure(airfoil, reynolds, self.n_crit, mach)
        if known is not None:
            raise RuntimeError(f"XFOIL previously failed here and it was recorded: {known}")

        try:
            polar = run_polar(
                airfoil, alpha_deg, reynolds, mach=mach,
                n_crit=self.n_crit, timeout_s=self.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised for the caller
            self.database.put_failure(
                airfoil, reynolds, f"{type(exc).__name__}: {exc}",
                n_requested=len(np.asarray(alpha_deg)), n_crit=self.n_crit, mach=mach,
            )
            raise
        self.database.put(airfoil, reynolds, polar, n_crit=self.n_crit, mach=mach)
        return polar
