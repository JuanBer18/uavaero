"""Disposable VSPAERO worker: `python -m uavaero.solvers.vspaero_worker <job.json>`.

Same reasoning as `uavaero/geometry/adapters/vsp_worker`, one step further along the pipeline:
OpenVSP's state is process-global and its native crashes kill the interpreter with no Python
exception to catch (rule 5). Running the solve here means a crash is a non-zero exit code in the
parent and nothing worse — and the parent already knows how to report that as a finding.

Everything it prints goes to the parent's captured stdout, so progress is visible in a terminal
while a long thick-panel solve runs.

job.json:    {"mode": "sweep" | "stability", "spec": <ParametricAirframe dict>,
              "condition": <OperatingCondition dict>, "workdir": str, "result_out": str}
             `mode` defaults to "sweep". A stability run is a different analysis on the same
             geometry, so it shares this worker rather than starting a second OpenVSP process.
result_out:  {"ok": bool, "result": <VspAeroResult dict>|null, "error": str|null}
Exit 0 iff ok.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# Runnable both as `-m` and as a bare script path. The application root is *found* rather
# than counted: `parents[N]` is exactly the kind of bootstrap that breaks silently when a
# file moves depth, and this one already did when these workers were promoted into this tree.
_here = Path(__file__).resolve()
for _candidate in _here.parents:
    if (_candidate / 'uavaero' / '__init__.py').is_file():
        sys.path.insert(0, str(_candidate))
        break

from uavaero.schema.airframe import ParametricAirframe  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m uavaero.solvers.vspaero_worker <job.json>", file=sys.stderr)
        return 2

    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    result_out = Path(job["result_out"])
    payload: dict = {"ok": False, "result": None, "error": None}

    try:
        from uavaero.schema.condition import OperatingCondition
        from uavaero.solvers.vspaero import run_alpha_sweep

        airframe = ParametricAirframe.from_dict(job["spec"])
        condition = OperatingCondition.from_dict(job["condition"])

        if job.get("mode") == "stability":
            import openvsp as vsp

            from uavaero.solvers.stability import run_stability

            print(
                f"VSPAERO stability: '{airframe.name}' at {condition.alpha_start_deg:g} deg, "
                f"beta {condition.beta_start_deg:g}, scope {condition.scope}, "
                f"CG x {condition.xyz_ref[0]:g} m...",
                flush=True,
            )
            stability = run_stability(vsp, airframe, condition, Path(job["workdir"]))
            payload["result"] = stability.to_dict()
            payload["ok"] = True
            margin = stability.static_margin
            print(
                f"VSPAERO stability: done. static margin "
                f"{'%.1f%%' % (margin * 100) if margin is not None else 'unknown'}, "
                f"Cm_alpha {stability.Cm_alpha:+.3f}/rad, Cn_beta {stability.Cn_beta:+.4f}/rad.",
                flush=True,
            )
            result_out.parent.mkdir(parents=True, exist_ok=True)
            result_out.write_text(json.dumps(payload), encoding="utf-8")
            return 0
        print(
            f"VSPAERO: '{airframe.name}', {len(condition.alphas())} alpha points "
            f"{condition.alpha_start_deg:g} to {condition.alpha_end_deg:g} deg, "
            f"{condition.fidelity.value} fidelity, scope {condition.scope}, "
            f"{'thick panels' if condition.thick else 'thin surfaces'}...",
            flush=True,
        )

        result = run_alpha_sweep(airframe, condition, Path(job["workdir"]))

        # A mesh with no Kutta nodes returns zeros rather than failing, which is the failure mode
        # that wasted the most time historically. Catch it here so it surfaces as an error rather
        # than as a suspiciously flat polar in the UI.
        if not result.mesh.ok:
            raise RuntimeError(
                "VSPAERO meshed the model but produced no Kutta nodes, so the solve carries no "
                "lift. This is the degenerate-mesh signature: check for very thin sections or an "
                "extreme dihedral/twist combination."
            )

        # A healthy mesh is not a healthy solve. Panel mode can converge to a physically
        # impossible answer with every mesh diagnostic green.
        implausible = result.implausible()
        if implausible:
            raise RuntimeError(
                f"VSPAERO produced a result that cannot be right: {implausible}. "
                f"On the reference aircraft this is what the thick/panel treatment does; the "
                f"thin treatment solves the same geometry correctly."
            )

        payload["result"] = result.to_dict()
        payload["ok"] = True
        best_ld, best_alpha = result.best_lift_to_drag
        print(
            f"VSPAERO: done. CD0 {result.cd0:.4f}, e {result.oswald_e:.3f}, "
            f"best L/D {best_ld:.1f} at {best_alpha:g} deg.",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    result_out.parent.mkdir(parents=True, exist_ok=True)
    result_out.write_text(json.dumps(payload), encoding="utf-8")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
