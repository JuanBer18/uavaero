"""Tier 1 — NeuralFoil 2D section polars, replacing XFOIL-per-case.

Calls AeroSandbox's bundled NeuralFoil wrapper (`asb.Airfoil.get_aero_from_neuralfoil`), which is
already fully vectorized over numpy alpha arrays — a single call, no batching machinery needed.
Fast enough (milliseconds) that no JobType wraps this; call it directly.
"""
from __future__ import annotations

import numpy as np

from uavaero.schema.results import Polar
from uavaero.geometry.airfoils import get_asb_airfoil


def run_polar(
    airfoil_name: str,
    alpha_deg: np.ndarray,
    reynolds: float,
    mach: float = 0.0,
    model_size: str = "large",
) -> Polar:
    airfoil = get_asb_airfoil(airfoil_name)
    alpha = np.asarray(alpha_deg, dtype=float)
    result = airfoil.get_aero_from_neuralfoil(
        alpha=alpha, Re=reynolds, mach=mach, model_size=model_size,
    )
    return Polar(
        alpha_deg=alpha,
        CL=np.asarray(result["CL"], dtype=float),
        CD=np.asarray(result["CD"], dtype=float),
        CM=np.asarray(result["CM"], dtype=float),
        source="neuralfoil",
        meta={
            "airfoil": airfoil_name, "reynolds": reynolds, "mach": mach, "model_size": model_size,
            # NeuralFoil's own out-of-distribution signal (0..1 per alpha) — the primary input to
            # the airfoil-screening trust layer (core/studies/airfoil_metrics.py). An exotic or
            # badly-digitized section far from the training set reads low here even when the
            # CL/CD numbers look (deceptively) plausible.
            "analysis_confidence": np.asarray(result["analysis_confidence"], dtype=float).tolist(),
            # Upper/lower transition locations (x/c) — diagnostic for laminar-bucket behavior.
            "top_xtr": np.asarray(result["Top_Xtr"], dtype=float).tolist(),
            "bot_xtr": np.asarray(result["Bot_Xtr"], dtype=float).tolist(),
        },
    )


class NeuralFoilSource:
    """NeuralFoil behind the common section interface — the production path.

    It cannot fail to converge, which is exactly why it is the default: inside an optimiser a
    section solver that occasionally returns nothing turns into an aircraft that occasionally
    scores as drag-free. It is also fast enough that no store is worth keeping.
    """

    name = "neuralfoil"

    def __init__(self, model_size: str = "large") -> None:
        self.model_size = model_size

    def polar(
        self, airfoil: str, alpha_deg: np.ndarray, reynolds: float, mach: float = 0.0
    ) -> Polar:
        return run_polar(airfoil, alpha_deg, reynolds, mach=mach, model_size=self.model_size)
