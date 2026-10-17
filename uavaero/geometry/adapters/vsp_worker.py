"""Disposable OpenVSP worker process: `python -m uavaero.geometry.adapters.vsp_worker <job.json>`.

OpenVSP's state is process-global and its native crashes kill the interpreter with no Python
exception (the exact failure mode that left AEROAPP jobs stuck "running" forever) — so every
OpenVSP build/check runs in one of these, launched by OpenVSPExportJob. A native crash here is a
nonzero exit code in the parent, nothing more.

job.json: {"spec": <ParametricAirframe dict>, "vsp3_out": str, "run_compgeom": bool,
           "result_out": str}
Writes result_out: {"ok": bool, "vsp3": str, "geom_ids": {...}, "compgeom": {...}|null,
                    "error": str|null}. Exit 0 iff ok.
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

from uavaero.schema.airframe import ParametricAirframe


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m uavaero.geometry.adapters.vsp_worker <job.json>", file=sys.stderr)
        return 2

    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    result_out = Path(job["result_out"])
    result: dict = {"ok": False, "vsp3": None, "geom_ids": None, "compgeom": None, "error": None}

    try:
        from uavaero.geometry.adapters import openvsp as vsp_adapter

        airframe = ParametricAirframe.from_dict(job["spec"])
        expected = len(airframe.lifting_surfaces()) + len(airframe.fuselages)

        print(f"building '{airframe.name}' ({expected} components) in OpenVSP...", flush=True)
        vsp3_out = Path(job["vsp3_out"])
        geom_ids = vsp_adapter.write_vsp3(airframe, vsp3_out)
        result["vsp3"] = str(vsp3_out)
        result["geom_ids"] = geom_ids
        print(f"wrote {vsp3_out}", flush=True)

        if job.get("run_compgeom", True):
            print("running CompGeom mesh check...", flush=True)
            compgeom = vsp_adapter.run_compgeom(expected_components=expected)
            result["compgeom"] = compgeom.to_dict()
            result["ok"] = compgeom.ok
            print(
                f"CompGeom: ok={compgeom.ok} tris={compgeom.total_num_tris} "
                f"wetted={compgeom.total_wetted_area:.4f}",
                flush=True,
            )
            for message in compgeom.messages:
                print(f"  ! {message}", flush=True)
        else:
            result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    result_out.parent.mkdir(parents=True, exist_ok=True)
    result_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
