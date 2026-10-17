# uavaero

A preliminary-design tool for UAV aerodynamics and flight dynamics, written in Python. You author an airframe directly and the tool tells you how it flies. It is a personal project by Juan Bernardo Pastuszak, still in development.

## What it does

- **Parametric airframe.** One airframe description (wing, tails, fuselage) drives everything else.
- **Aerodynamics.** OpenVSP and VSPAERO give the 3D solution. Several independent drag estimates are kept side by side so the differences stay visible, including a component buildup and a strip-based viscous correction with 2D section data from NeuralFoil and XFOIL.
- **Stability and dynamic modes.** VSPAERO static derivatives are combined with AVL rate derivatives. The tool checks that they agree, builds longitudinal and lateral state-space models and names the dynamic modes (short period, phugoid, Dutch roll, roll, spiral).
- **Design-space search.** A differential-evolution search over a small set of engineering variables. It rejects failed or implausible results (for example non-positive drag) before a design is scored.
- **Interface.** A PySide6 desktop app with a plan-view drawing and a 3D viewport.

## What it does not do (yet)

- Sizing, mission analysis, structures and custom CFD are out of scope.
- Results are numerical predictions. They have not been validated against flight data.
- Mass is stated by the user and inertia is estimated. One unsteady derivative is omitted, and the tool flags both limits in its output.
- A physical 1 m test aircraft is planned. It is not built.

## Structure

```
uavaero/        the package, with no Qt imports
  schema/       airframe, stability, condition and results data classes
  geometry/     planform, fuselage and tail generation, OpenVSP adapter
  solvers/      VSPAERO, AVL, section sources (NeuralFoil, XFOIL), parasite drag
  aero/         strip correction and stall checks
  dynamics/     state-space models, eigenmodes, report
  optimisation/ search space, screening, objectives, driver
  store/        result cache and airframe file format
  app/          the Qt interface (calls into the package, never the reverse)
tests/          unit tests and reference cases that run a real solver
```

OpenVSP and VSPAERO run in a disposable subprocess, because a crash in OpenVSP kills the Python process and cannot be caught.

## Requirements

- Python 3.13 and the pinned packages in `pyproject.toml`.
- OpenVSP 3.51 with its Python API. Without it, the VSPAERO solves and the reference cases skip.
- AVL and XFOIL binaries (not included) and the airfoil data, resolved through `uavaero/paths.py`.

Run the tests with `pytest`. Tests marked `slow` run a real solver.

## Licence

MIT. See `LICENSE`.
