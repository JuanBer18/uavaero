"""Cross-thread Qt signals must have a receiver QObject, or they are not cross-thread at all.

This guards one specific, silent, and genuinely dangerous mistake.

Qt decides between a direct and a queued call by comparing the emitting thread with the
*receiver's* thread. A signal connected to a bound method of a QObject has a receiver, so a
solve worker's output arrives on the GUI thread. A signal connected to a **lambda** has no
receiver object — Qt falls back to the sender as context, the threads therefore always match,
and AutoConnection resolves to a *direct* call that runs the slot in the worker thread.

Measured here, so it is not folklore:

    worker.out.connect(lambda line: ...)   ->  runs in WORKER
    worker.out.connect(window.slot)        ->  runs in GUI

`main_window` connected the solver's `output` signal to a lambda, so `console.append` mutated a
QPlainTextEdit from the solve thread — around 3800 times per aircraft-scope sweep. Qt widgets are
not thread-safe. The window stopped repainting about a second after Solve was pressed, Windows
marked it "not responding", and the process eventually died natively with no Python traceback,
which is exactly the signature of memory corruption rather than a bug with a stack trace.

There is nothing to see in review: `connect(lambda line: self._log(line, DETAIL))` looks like it
does what a bound method does. So the check is mechanical and lives here.

This is asserted against the source rather than by running Qt because the whole test suite is
deliberately Qt-free (CLAUDE.md rule 1), and reproducing it behaviourally would need a real
MainWindow on a real display.
"""
from __future__ import annotations

import ast

from uavaero.paths import APP_ROOT

#: Modules that move work onto a QThread and therefore have cross-thread connections.
THREADED_MODULES = [
    APP_ROOT / "uavaero" / "app" / "main_window.py",
    APP_ROOT / "uavaero" / "app" / "screens" / "optimise.py",
]


def _worker_connections(tree: ast.AST) -> list[ast.Call]:
    """Every `self._worker.<signal>.connect(...)` call in the module."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if not (isinstance(callee, ast.Attribute) and callee.attr == "connect"):
            continue
        signal = callee.value
        if not isinstance(signal, ast.Attribute):
            continue
        owner = signal.value
        if isinstance(owner, ast.Attribute) and owner.attr == "_worker":
            found.append(node)
    return found


def test_no_worker_signal_is_connected_to_a_lambda():
    """The regression itself. A lambda receiver silently makes the connection direct, and the slot
    then touches widgets from the solve thread."""
    for module in THREADED_MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        connections = _worker_connections(tree)
        assert connections, f"{module.name}: expected worker connections, found none"
        for call in connections:
            receiver = call.args[0] if call.args else None
            assert not isinstance(receiver, ast.Lambda), (
                f"{module.name}:{call.lineno} connects a worker signal to a lambda. A lambda has "
                f"no receiver QObject, so Qt runs it in the worker thread — use a bound method."
            )


def test_every_worker_signal_goes_to_a_method_on_self():
    """A bound method of the window is what gives Qt a receiver whose thread it can compare. Any
    other callable — a module function, a nested closure — has the same defect as the lambda."""
    for module in THREADED_MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for call in _worker_connections(tree):
            receiver = call.args[0] if call.args else None
            assert isinstance(receiver, ast.Attribute) and isinstance(receiver.value, ast.Name), (
                f"{module.name}:{call.lineno} connects a worker signal to something other than a "
                f"bound method"
            )
            assert receiver.value.id == "self", (
                f"{module.name}:{call.lineno} receiver is not a method on self"
            )
