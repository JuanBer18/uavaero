"""Is there a usable OpenGL context on this machine? Asked in a process we can afford to lose.

Creating a VTK render window without a valid pixel format does not raise — it **segfaults**, taking
the whole app with it. A `try/except` cannot catch that, so the question has to be asked somewhere
a crash is only an exit code. Same reasoning as the OpenVSP workers (CLAUDE.md rule 6): the answer
to an uncatchable native crash is a disposable process, not a bigger try block.

**The probe must construct the real widget.** An offscreen `vtkRenderWindow` comes up fine in
environments where the on-screen `QtInteractor` dies, so a cheaper probe answers "yes" and the app
then crashes doing the thing the probe never tried. It builds a `QtInteractor` inside a real
`QApplication`, exactly what the caller is about to do.

Run directly, it exits 0 when a render window came up:

    python -m uavaero.app.viewport.probe
"""
from __future__ import annotations

import subprocess
import sys

#: Cached across calls: the answer cannot change while the app runs, and the subprocess costs about
#: a second. None means "not asked yet".
_AVAILABLE: bool | None = None

PROBE_TIMEOUT_S = 30.0


def opengl_available(force: bool = False) -> bool:
    """True when a VTK render window can actually be created here.

    Never raises and never crashes the caller: a timeout, a missing interpreter and a segfault in
    the probe all mean the same thing — do not build a 3D view in this process.
    """
    global _AVAILABLE
    if _AVAILABLE is not None and not force:
        return _AVAILABLE
    try:
        import tempfile

        from uavaero.solvers.vspaero import _worker_environment

        # `cwd` and `env` travel together: `-m` resolves through the working directory, so one
        # without the other is either an import failure or a process writing into whatever
        # directory the app happens to be standing in. This probe writes nothing today, but the
        # invariant is only useful if it has no exceptions to remember.
        completed = subprocess.run(
            [sys.executable, "-m", "uavaero.app.viewport.probe"],
            cwd=tempfile.gettempdir(), env=_worker_environment(),
            capture_output=True, timeout=PROBE_TIMEOUT_S,
        )
        _AVAILABLE = completed.returncode == 0
    except Exception:  # noqa: BLE001 - timeout, OSError, anything: the answer is still "no"
        _AVAILABLE = False
    return _AVAILABLE


def _probe() -> int:
    """Runs in the disposable process. A segfault here is exactly the answer we wanted.

    The parent's `QT_QPA_PLATFORM` is inherited, so this tests the platform actually in use.
    """
    try:
        from PySide6.QtWidgets import QApplication
        from pyvistaqt import QtInteractor

        app = QApplication.instance() or QApplication([])
        interactor = QtInteractor()
        interactor.render()
        ok = bool(interactor.render_window.SupportsOpenGL())
        interactor.close()
        del app
        return 0 if ok else 1
    except Exception:  # noqa: BLE001
        return 1


if __name__ == "__main__":
    raise SystemExit(_probe())
